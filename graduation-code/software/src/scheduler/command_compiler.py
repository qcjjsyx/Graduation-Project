from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from src.command_codec import (
    COMMAND_RECORD_BYTES,
    COMMAND_SCHEMA_VERSION,
    COMPLETION_RECORD_BYTES,
    DESCRIPTOR_RECORD_BYTES,
    NONE,
    PVECTOR_ELEMENT_BYTES,
    PVECTOR_INIT_VALUE,
    SCALE_AUTO,
    Command,
    Completion,
    DataFormat,
    DataLayout,
    Descriptor,
    DescriptorType,
    KernelBackend,
    Opcode,
    PivotPolicy,
    SolveDirection,
    StatusCode,
    encode_commands,
    encode_completions,
    encode_descriptors,
    encode_pivot_threshold,
    validate_command_batch,
)
from src.config import CommandCompilerConfig, GemmConfig, PivotPolicyConfig
from src.dataStruct import MapTableEntry, MemoryRegion, NodeCompileRecord
from src.memory.planner import plan_regions
from src.scheduler.task_queue import sibling_friendly_order


class CommandCompileError(ValueError):
    pass


# How many nodes may interleave in the command stream at once. Two matches the
# dual-front device: two independent fronts alternate command-by-command, and
# wider forests are emitted pair by pair (documented command-order constraint).
INTERLEAVE_WIDTH = 2


@dataclass(frozen=True)
class CompiledCommandArtifact:
    image: bytes
    manifest: dict[str, Any]
    commands: tuple[Command, ...]
    descriptors: tuple[Descriptor, ...]
    completion_templates: tuple[Completion, ...]
    regions: dict[str, MemoryRegion]


@dataclass(frozen=True)
class _RegionSpec:
    name: str
    rows: int
    cols: int
    data_format: DataFormat
    initial_data: bytes

    @property
    def element_size(self) -> int:
        return 8 if self.data_format == DataFormat.FP64 else 4

    @property
    def row_stride(self) -> int:
        return self.cols * self.element_size if self.rows and self.cols else 0

    @property
    def size(self) -> int:
        return self.rows * self.row_stride


@dataclass(frozen=True)
class _DescriptorSpec:
    name: str
    descriptor_type: DescriptorType
    fields: dict[str, Any]
    payload: bytes = b""


@dataclass(frozen=True)
class _CommandSpec:
    opcode: Opcode
    node_id: int
    descriptor_name: str | None
    waits: tuple[int, ...]
    signal_token: int


def compile_command_artifact(
    *,
    nodes: Sequence[NodeCompileRecord],
    map_tables: Sequence[Sequence[MapTableEntry]],
    local_fronts: Sequence[np.ndarray],
    permutation: Sequence[int],
    rhs: np.ndarray,
    config: CommandCompilerConfig,
    pivot: PivotPolicyConfig,
    gemm: GemmConfig = GemmConfig(),
) -> CompiledCommandArtifact:
    """Compile symbolic/front IR into a self-contained Command schema v2 image."""

    ordered_nodes, parent, matrix_dim = _validate_inputs(
        nodes, map_tables, local_fronts, permutation, rhs, config
    )
    region_specs = _build_data_regions(
        ordered_nodes, local_fronts, permutation, rhs, matrix_dim
    )
    descriptor_specs = _build_static_descriptors(
        ordered_nodes, map_tables, config.tile_size, pivot, gemm
    )
    command_specs, node_command_ids, node_tokens = _build_commands(
        ordered_nodes, map_tables, parent, config.emit_solve_commands
    )

    wait_descriptor_names: dict[tuple[int, ...], str] = {}
    for command in command_specs:
        if command.waits and command.waits not in wait_descriptor_names:
            name = f"dependency.{len(wait_descriptor_names)}"
            wait_descriptor_names[command.waits] = name
            descriptor_specs.append(
                _DescriptorSpec(
                    name=name,
                    descriptor_type=DescriptorType.DEPENDENCY_DESC,
                    fields={"tokens": command.waits},
                    payload=_pack_u32(command.waits),
                )
            )

    descriptor_ids = {
        descriptor.name: index for index, descriptor in enumerate(descriptor_specs)
    }
    if len(descriptor_ids) != len(descriptor_specs):
        raise CommandCompileError("duplicate descriptor name")

    payload_offsets: dict[str, int] = {}
    payload_blob = bytearray()
    for descriptor in descriptor_specs:
        if descriptor.payload:
            padding = (-len(payload_blob)) & 3
            payload_blob.extend(b"\x00" * padding)
            payload_offsets[descriptor.name] = len(payload_blob)
            payload_blob.extend(descriptor.payload)

    completion_templates = tuple(
        Completion(
            command_id=command_id,
            node_id=spec.node_id,
            status_code=StatusCode.OK,
        )
        for command_id, spec in enumerate(command_specs)
    )

    layout_specs: list[tuple[str, int]] = [
        ("command_buffer", len(command_specs) * COMMAND_RECORD_BYTES),
        ("descriptor_table", len(descriptor_specs) * DESCRIPTOR_RECORD_BYTES),
        ("descriptor_payload", len(payload_blob)),
        ("completion_status", len(completion_templates) * COMPLETION_RECORD_BYTES),
    ]
    layout_specs.extend((region.name, region.size) for region in region_specs)
    regions, total_bytes = plan_regions(layout_specs, alignment=config.alignment)

    descriptors = tuple(
        _resolve_descriptor(
            spec,
            descriptor_ids=descriptor_ids,
            data_regions=regions,
            region_specs={region.name: region for region in region_specs},
            payload_base=regions["descriptor_payload"].offset,
            payload_offset=payload_offsets.get(spec.name),
        )
        for spec in descriptor_specs
    )
    commands = tuple(
        Command(
            opcode=spec.opcode,
            flags=0,
            command_id=command_id,
            node_id=spec.node_id,
            descriptor_id=(
                NONE
                if spec.descriptor_name is None
                else descriptor_ids[spec.descriptor_name]
            ),
            wait_list_id=(
                NONE
                if not spec.waits
                else descriptor_ids[wait_descriptor_names[spec.waits]]
            ),
            signal_token=spec.signal_token,
        )
        for command_id, spec in enumerate(command_specs)
    )

    image = bytearray(total_bytes)
    _copy_region(image, regions["command_buffer"], encode_commands(commands))
    _copy_region(image, regions["descriptor_table"], encode_descriptors(descriptors))
    _copy_region(image, regions["descriptor_payload"], bytes(payload_blob))
    _copy_region(
        image,
        regions["completion_status"],
        encode_completions(completion_templates),
    )
    for spec in region_specs:
        _copy_region(image, regions[spec.name], spec.initial_data)

    validate_command_batch(
        commands,
        descriptors,
        bytes(image),
        token_count=len(commands),
        max_wait_tokens=config.max_wait_tokens,
    )

    manifest = _build_manifest(
        config=config,
        pivot=pivot,
        gemm=gemm,
        total_bytes=total_bytes,
        regions=regions,
        descriptor_ids=descriptor_ids,
        command_specs=command_specs,
        completion_templates=completion_templates,
        ordered_nodes=ordered_nodes,
        node_command_ids=node_command_ids,
        node_tokens=node_tokens,
        parent=parent,
        permutation=permutation,
        map_tables=map_tables,
    )
    return CompiledCommandArtifact(
        image=bytes(image),
        manifest=manifest,
        commands=commands,
        descriptors=descriptors,
        completion_templates=completion_templates,
        regions=regions,
    )


def _validate_inputs(
    nodes: Sequence[NodeCompileRecord],
    map_tables: Sequence[Sequence[MapTableEntry]],
    local_fronts: Sequence[np.ndarray],
    permutation: Sequence[int],
    rhs: np.ndarray,
    config: CommandCompilerConfig,
) -> tuple[list[NodeCompileRecord], list[int], int]:
    if not nodes:
        raise CommandCompileError("command compiler requires at least one node")
    if len(nodes) != len(map_tables) or len(nodes) != len(local_fronts):
        raise CommandCompileError("nodes, map tables, and local fronts must match")

    ids = [node.node_id for node in nodes]
    if len(ids) != len(set(ids)):
        raise CommandCompileError("duplicate node ID")
    if sorted(ids) != list(range(len(nodes))):
        raise CommandCompileError("node IDs must form a dense 0..N-1 range")
    if ids != list(range(len(nodes))):
        raise CommandCompileError("nodes must be supplied in ascending node-ID order")
    ordered = sorted(nodes, key=lambda node: node.node_id)
    parent = [node.parent_id for node in ordered]
    for node in ordered:
        if node.parent_id < -1 or node.parent_id >= len(nodes):
            raise CommandCompileError(f"node {node.node_id} has invalid parent ID")
        if node.parent_id == node.node_id:
            raise CommandCompileError(f"node {node.node_id} cannot parent itself")
    try:
        sibling_friendly_order(parent)
    except ValueError as exc:
        raise CommandCompileError(f"invalid elimination forest: {exc}") from exc

    matrix_dim = len(permutation)
    if matrix_dim == 0:
        raise CommandCompileError("permutation must not be empty")
    if sorted(int(value) for value in permutation) != list(range(matrix_dim)):
        raise CommandCompileError("permutation is not a valid 0..N-1 permutation")
    rhs_array = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if rhs_array.size == 0:
        raise CommandCompileError("RHS must not be empty")
    if rhs_array.size != matrix_dim:
        raise CommandCompileError("RHS length does not match matrix dimension")
    if not np.all(np.isfinite(rhs_array)):
        raise CommandCompileError("RHS contains non-finite values")

    for node, front in zip(ordered, local_fronts):
        if node.pivot_dim <= 0 or node.total_dim < node.pivot_dim:
            raise CommandCompileError(f"node {node.node_id} has invalid dimensions")
        if node.total_dim > config.max_front_size:
            raise CommandCompileError(
                f"node {node.node_id} front size {node.total_dim} exceeds "
                f"limit {config.max_front_size}"
            )
        expected_prefix = tuple(range(node.node_range.start, node.node_range.end))
        if node.front_indices[: node.pivot_dim] != expected_prefix:
            raise CommandCompileError(
                f"node {node.node_id} front pivot prefix does not match node range"
            )
        if len(set(node.front_indices)) != node.total_dim or any(
            index < 0 or index >= matrix_dim for index in node.front_indices
        ):
            raise CommandCompileError(f"node {node.node_id} front indices are invalid")
        values = np.asarray(front, dtype=np.float64)
        if values.shape != (node.total_dim, node.total_dim):
            raise CommandCompileError(f"node {node.node_id} local front shape mismatch")
        if not np.all(np.isfinite(values)):
            raise CommandCompileError(
                f"node {node.node_id} local front contains non-finite values"
            )

    seen_children: set[int] = set()
    for parent_id, entries in enumerate(map_tables):
        expected_children = sorted(
            node.node_id for node in ordered if node.parent_id == parent_id
        )
        actual_children = sorted(entry.child_id for entry in entries)
        if actual_children != expected_children:
            raise CommandCompileError(
                f"node {parent_id} contribution maps do not match its children"
            )
        parent_front_size = ordered[parent_id].total_dim
        for entry in entries:
            if entry.child_id in seen_children:
                raise CommandCompileError("duplicate child contribution map")
            seen_children.add(entry.child_id)
            child = ordered[entry.child_id]
            if len(entry.row_map) != len(entry.col_map):
                raise CommandCompileError("contribution row/column maps differ in length")
            if entry.row_map != list(range(child.update_dim)):
                raise CommandCompileError(
                    f"child {entry.child_id} map does not cover its update"
                )
            if len(set(entry.col_map)) != len(entry.col_map) or any(
                column < 0 or column >= parent_front_size for column in entry.col_map
            ):
                raise CommandCompileError(
                    f"child {entry.child_id} parent-front map is invalid"
                )
    return ordered, parent, matrix_dim


def _build_data_regions(
    nodes: Sequence[NodeCompileRecord],
    local_fronts: Sequence[np.ndarray],
    permutation: Sequence[int],
    rhs: np.ndarray,
    matrix_dim: int,
) -> list[_RegionSpec]:
    regions = [
        _RegionSpec(
            "permutation_data",
            1,
            matrix_dim,
            DataFormat.INT32,
            np.asarray(permutation, dtype="<i4").tobytes(order="C"),
        ),
        _RegionSpec(
            "rhs_data",
            1,
            matrix_dim,
            DataFormat.FP32,
            np.asarray(rhs, dtype="<f4").reshape(-1).tobytes(order="C"),
        ),
        _RegionSpec(
            "solution_data",
            1,
            matrix_dim,
            DataFormat.FP32,
            bytes(matrix_dim * 4),
        ),
    ]
    for node, front in zip(nodes, local_fronts):
        prefix = f"node.{node.node_id}"
        update = node.update_dim
        regions.extend(
            [
                _RegionSpec(
                    f"{prefix}.front",
                    node.total_dim,
                    node.total_dim,
                    DataFormat.FP32,
                    np.asarray(front, dtype="<f4").tobytes(order="C"),
                ),
                _RegionSpec(
                    f"{prefix}.update",
                    update,
                    update,
                    DataFormat.FP32,
                    bytes(update * update * 4),
                ),
                _RegionSpec(
                    f"{prefix}.factor_l",
                    node.total_dim,
                    node.pivot_dim,
                    DataFormat.FP32,
                    bytes(node.total_dim * node.pivot_dim * 4),
                ),
                _RegionSpec(
                    f"{prefix}.factor_u",
                    node.pivot_dim,
                    node.total_dim,
                    DataFormat.FP32,
                    bytes(node.pivot_dim * node.total_dim * 4),
                ),
                _RegionSpec(
                    f"{prefix}.p_vector",
                    1,
                    node.pivot_dim,
                    DataFormat.INT32,
                    struct.pack(
                        f"<{node.pivot_dim}i",
                        *([PVECTOR_INIT_VALUE] * node.pivot_dim),
                    ),
                ),
                _RegionSpec(
                    f"{prefix}.solve_workspace",
                    2,
                    node.pivot_dim,
                    DataFormat.FP32,
                    bytes(2 * node.pivot_dim * 4),
                ),
            ]
        )
    for region in regions:
        if len(region.initial_data) != region.size:
            raise CommandCompileError(f"{region.name} initialization size mismatch")
    return regions


def _build_static_descriptors(
    nodes: Sequence[NodeCompileRecord],
    map_tables: Sequence[Sequence[MapTableEntry]],
    tile_size: int,
    pivot: PivotPolicyConfig,
    gemm: GemmConfig = GemmConfig(),
) -> list[_DescriptorSpec]:
    descriptors: list[_DescriptorSpec] = []
    policy_word = int(PivotPolicy[pivot.policy])
    threshold_word = encode_pivot_threshold(pivot.threshold_ratio)

    def region(name: str) -> str:
        descriptor_name = f"region:{name}"
        descriptors.append(
            _DescriptorSpec(
                descriptor_name,
                DescriptorType.REGION_DESC,
                {"region_name": name},
            )
        )
        return descriptor_name

    region("permutation_data")
    region("rhs_data")
    region("solution_data")
    for node in nodes:
        prefix = f"node.{node.node_id}"
        for suffix in (
            "front",
            "update",
            "factor_l",
            "factor_u",
            "p_vector",
            "solve_workspace",
        ):
            region(f"{prefix}.{suffix}")

    for node in nodes:
        prefix = f"node.{node.node_id}"
        descriptors.append(
            _DescriptorSpec(
                f"{prefix}.factor",
                DescriptorType.FACTOR_DESC,
                {
                    "l": f"region:{prefix}.factor_l",
                    "u": f"region:{prefix}.factor_u",
                    "p": f"region:{prefix}.p_vector",
                    "total_dim": node.total_dim,
                    "pivot_dim": node.pivot_dim,
                    "format": DataFormat.FP32,
                    "workspace": f"region:{prefix}.solve_workspace",
                },
            )
        )
        contribution_names = [
            f"{prefix}.contribution.{entry.child_id}"
            for entry in map_tables[node.node_id]
        ]
        descriptors.append(
            _DescriptorSpec(
                f"{prefix}.front",
                DescriptorType.FRONT_DESC,
                {
                    "front": f"region:{prefix}.front",
                    "total_dim": node.total_dim,
                    "pivot_dim": node.pivot_dim,
                    "tile_size": tile_size,
                    "contribution": (
                        contribution_names[0] if len(contribution_names) == 1 else None
                    ),
                    "factor": f"{prefix}.factor",
                    "p": f"region:{prefix}.p_vector",
                    "workspace": f"region:{prefix}.solve_workspace",
                    "pivot_policy": policy_word,
                    "pivot_threshold": threshold_word,
                },
            )
        )
        kernel_shapes = {
            "trsm_left": (node.pivot_dim, node.update_dim, node.pivot_dim),
            "trsm_right": (node.update_dim, node.pivot_dim, node.pivot_dim),
            "gemm": (node.update_dim, node.update_dim, node.pivot_dim),
        }
        for kernel_name, (m, n, k) in kernel_shapes.items():
            int32_gemm = gemm.int32_gemm_enabled and kernel_name == "gemm"
            descriptors.append(
                _DescriptorSpec(
                    f"{prefix}.kernel.{kernel_name}",
                    DescriptorType.KERNEL_DESC,
                    {
                        "a": f"{prefix}.front",
                        "b": f"{prefix}.front",
                        "c": f"{prefix}.front",
                        "m": m,
                        "n": n,
                        "k": k,
                        "backend": (
                            KernelBackend.SYSTEMC_INT32_GEMM_MODEL
                            if int32_gemm
                            else KernelBackend.SYSTEMC_FP32_DEVICE_MODEL
                        ),
                        "format": DataFormat.FP32,
                        "scale": (
                            f"{prefix}.kernel.gemm.scale" if int32_gemm else None
                        ),
                        "tile_size": tile_size,
                    },
                )
            )
            if int32_gemm:
                # One SCALE_DESC per node GEMM kernel. It never propagates to
                # the front/DDR data format or to any other kernel.
                descriptors.append(
                    _DescriptorSpec(
                        f"{prefix}.kernel.gemm.scale",
                        DescriptorType.SCALE_DESC,
                        {
                            "exponent_a": gemm.scale_desc.exponent_a,
                            "exponent_b": gemm.scale_desc.exponent_b,
                            "rounding": int(gemm.scale_desc.rounding),
                            "saturation": int(gemm.scale_desc.saturation),
                            "overflow": int(gemm.scale_desc.overflow),
                        },
                    )
                )
        for direction, label in (
            (SolveDirection.FORWARD, "solve_forward"),
            (SolveDirection.BACKWARD, "solve_backward"),
        ):
            descriptors.append(
                _DescriptorSpec(
                    f"{prefix}.{label}",
                    DescriptorType.SOLVE_DESC,
                    {
                        "factor": f"{prefix}.factor",
                        "rhs": "region:rhs_data",
                        "solution": "region:solution_data",
                        "p": f"region:{prefix}.p_vector",
                        "workspace": f"region:{prefix}.solve_workspace",
                        "rhs_count": 1,
                        "direction": direction,
                        "format": DataFormat.FP32,
                    },
                )
            )

    for parent_id, entries in enumerate(map_tables):
        parent = nodes[parent_id]
        for entry in entries:
            child = nodes[entry.child_id]
            payload = _pack_u32((*entry.row_map, *entry.col_map))
            descriptors.append(
                _DescriptorSpec(
                    f"node.{parent_id}.contribution.{entry.child_id}",
                    DescriptorType.CONTRIBUTION_DESC,
                    {
                        "source": f"region:node.{entry.child_id}.update",
                        "target": f"region:node.{parent_id}.front",
                        "child_id": entry.child_id,
                        "parent_id": parent_id,
                        "row_count": len(entry.row_map),
                        "col_count": len(entry.col_map),
                        "source_stride": child.update_dim * 4,
                        "target_stride": parent.total_dim * 4,
                        "format": DataFormat.FP32,
                    },
                    payload=payload,
                )
            )
    return descriptors


@dataclass(frozen=True)
class _CommandTemplate:
    """Command template with symbolic dependency references (node_id, token_name)."""

    opcode: Opcode
    node_id: int
    descriptor_name: str | None
    wait_refs: tuple[tuple[int, str], ...]
    token_name: str | None


def _node_command_plan(
    nodes: Sequence[NodeCompileRecord],
    map_tables: Sequence[Sequence[MapTableEntry]],
    node: NodeCompileRecord,
) -> list[_CommandTemplate]:
    node_id = node.node_id
    prefix = f"node.{node_id}"

    def child_token(child_id: int) -> str:
        return "update_stored" if nodes[child_id].update_dim else "commit"

    children = sorted(map_tables[node_id], key=lambda item: item.child_id)
    # Admission: a parent front waits for every child to complete (its
    # NODE_COMMIT token) before it may request residency.
    admission_refs = tuple((entry.child_id, "commit") for entry in children)
    plan: list[_CommandTemplate] = [
        # Admission dependency: a parent front may only start once its
        # required children completed. The device must never infer the tree
        # from host side state; it only sees these dependency tokens.
        _CommandTemplate(
            Opcode.NODE_BEGIN, node_id, f"{prefix}.front", admission_refs, "begin"
        ),
        _CommandTemplate(
            Opcode.LOAD_FRONT,
            node_id,
            f"{prefix}.front",
            ((node_id, "begin"),),
            "loaded",
        ),
    ]
    current = "loaded"
    for entry in children:
        assembled = f"assembled_child_{entry.child_id}"
        plan.append(
            _CommandTemplate(
                Opcode.ASSEMBLE_EXTEND_ADD,
                node_id,
                f"{prefix}.contribution.{entry.child_id}",
                ((node_id, current), (entry.child_id, child_token(entry.child_id))),
                assembled,
            )
        )
        current = assembled
    plan.append(
        _CommandTemplate(
            Opcode.PANEL_LU, node_id, f"{prefix}.front", ((node_id, current),), "panel"
        )
    )
    if node.update_dim:
        plan.extend(
            [
                _CommandTemplate(
                    Opcode.TRSM_LEFT,
                    node_id,
                    f"{prefix}.kernel.trsm_left",
                    ((node_id, "panel"),),
                    "trsm_left",
                ),
                _CommandTemplate(
                    Opcode.TRSM_RIGHT,
                    node_id,
                    f"{prefix}.kernel.trsm_right",
                    ((node_id, "panel"),),
                    "trsm_right",
                ),
                _CommandTemplate(
                    Opcode.GEMM_SCHUR,
                    node_id,
                    f"{prefix}.kernel.gemm",
                    ((node_id, "trsm_left"), (node_id, "trsm_right")),
                    "schur",
                ),
                _CommandTemplate(
                    Opcode.STORE_FACTOR,
                    node_id,
                    f"{prefix}.factor",
                    ((node_id, "trsm_left"), (node_id, "trsm_right")),
                    "factor_stored",
                ),
                _CommandTemplate(
                    Opcode.STORE_UPDATE,
                    node_id,
                    f"region:{prefix}.update",
                    ((node_id, "schur"),),
                    "update_stored",
                ),
                _CommandTemplate(
                    Opcode.NODE_COMMIT,
                    node_id,
                    f"{prefix}.front",
                    ((node_id, "factor_stored"), (node_id, "update_stored")),
                    "commit",
                ),
            ]
        )
    else:
        plan.extend(
            [
                _CommandTemplate(
                    Opcode.STORE_FACTOR,
                    node_id,
                    f"{prefix}.factor",
                    ((node_id, "panel"),),
                    "factor_stored",
                ),
                _CommandTemplate(
                    Opcode.NODE_COMMIT,
                    node_id,
                    f"{prefix}.front",
                    ((node_id, "factor_stored"),),
                    "commit",
                ),
            ]
        )
    return plan


def _build_commands(
    nodes: Sequence[NodeCompileRecord],
    map_tables: Sequence[Sequence[MapTableEntry]],
    parent: Sequence[int],
    emit_solve_commands: bool = True,
) -> tuple[list[_CommandSpec], dict[int, list[int]], dict[int, dict[str, int]]]:
    """Emit the command stream.

    Node blocks are emitted in an interleaved topological order: a node becomes
    ready once every dependency reference of its next command is already
    emitted (cross-node references are the child completion tokens of
    NODE_BEGIN/ASSEMBLE).  At most INTERLEAVE_WIDTH nodes are open at once and
    take turns command-by-command, so two independent fronts alternate in the
    stream with O(1) command spacing while a wider forest is emitted pair by
    pair.  command_id, signal_token and completion slots are (re)assigned
    densely in emission order; every wait token therefore references a strictly
    earlier command, which the manifest and C++ validators require.  Solve
    commands stay in a trailing chain.
    """

    plans: dict[int, list[_CommandTemplate]] = {
        node.node_id: _node_command_plan(nodes, map_tables, node) for node in nodes
    }
    commands: list[_CommandSpec] = []
    node_command_ids: dict[int, list[int]] = {node.node_id: [] for node in nodes}
    node_tokens: dict[int, dict[str, int]] = {node.node_id: {} for node in nodes}
    emitted_refs: set[tuple[int, str]] = set()

    def emit(
        opcode: Opcode,
        node_id: int,
        descriptor_name: str | None,
        waits: Iterable[int] = (),
        token_name: str | None = None,
    ) -> int:
        wait_tuple = tuple(dict.fromkeys(int(token) for token in waits))
        command_id = len(commands)
        token = command_id
        commands.append(
            _CommandSpec(opcode, node_id, descriptor_name, wait_tuple, token)
        )
        node_command_ids[node_id].append(command_id)
        if token_name is not None:
            node_tokens[node_id][token_name] = token
        return token

    preference = sibling_friendly_order(list(parent))
    cursors = {node_id: 0 for node_id in plans}
    open_nodes: list[int] = []
    rotation = 0
    total_commands = sum(len(plan) for plan in plans.values())

    def next_ready(node_id: int) -> bool:
        cursor = cursors[node_id]
        if cursor >= len(plans[node_id]):
            return False
        return all(ref in emitted_refs for ref in plans[node_id][cursor].wait_refs)

    while len(commands) < total_commands:
        open_nodes = [
            node_id for node_id in open_nodes if cursors[node_id] < len(plans[node_id])
        ]
        for node_id in preference:
            if len(open_nodes) >= INTERLEAVE_WIDTH:
                break
            if node_id in open_nodes or cursors[node_id] >= len(plans[node_id]):
                continue
            if next_ready(node_id):
                open_nodes.append(node_id)
        ready = [node_id for node_id in open_nodes if next_ready(node_id)]
        if not ready:
            raise CommandCompileError("command templates contain a dependency cycle")
        chosen: int | None = None
        for step in range(len(open_nodes)):
            index = (rotation + step) % len(open_nodes)
            if open_nodes[index] in ready:
                chosen = open_nodes[index]
                rotation = (index + 1) % max(1, len(open_nodes))
                break
        assert chosen is not None
        template = plans[chosen][cursors[chosen]]
        waits = tuple(node_tokens[ref_node][ref_name] for ref_node, ref_name in template.wait_refs)
        emit(template.opcode, chosen, template.descriptor_name, waits, template.token_name)
        if template.token_name is not None:
            emitted_refs.add((chosen, template.token_name))
        cursors[chosen] += 1

    if not emit_solve_commands:
        return commands, node_command_ids, node_tokens

    previous: int | None = None
    for node in nodes:
        solve_waits = [node_tokens[node.node_id]["commit"]]
        if previous is not None:
            solve_waits.append(previous)
        previous = emit(
            Opcode.SOLVE_FORWARD,
            node.node_id,
            f"node.{node.node_id}.solve_forward",
            solve_waits,
            "solve_forward",
        )
    for node in reversed(nodes):
        assert previous is not None
        previous = emit(
            Opcode.SOLVE_BACKWARD,
            node.node_id,
            f"node.{node.node_id}.solve_backward",
            (previous, node_tokens[node.node_id]["commit"]),
            "solve_backward",
        )
    return commands, node_command_ids, node_tokens


def _resolve_descriptor(
    spec: _DescriptorSpec,
    *,
    descriptor_ids: dict[str, int],
    data_regions: dict[str, MemoryRegion],
    region_specs: dict[str, _RegionSpec],
    payload_base: int,
    payload_offset: int | None,
) -> Descriptor:
    fields = spec.fields
    payload_bytes = len(spec.payload)
    absolute_payload = payload_base + payload_offset if payload_offset is not None else 0

    def descriptor_id(name: str | None) -> int:
        return NONE if name is None else descriptor_ids[name]

    if spec.descriptor_type == DescriptorType.REGION_DESC:
        region_name = fields["region_name"]
        region = data_regions[region_name]
        metadata = region_specs[region_name]
        body = (
            *_split_u64(region.offset),
            *_split_u64(region.size),
            metadata.row_stride,
            metadata.rows,
            metadata.cols,
            int(metadata.data_format),
            int(DataLayout.ROW_MAJOR),
            0,
        )
    elif spec.descriptor_type == DescriptorType.FRONT_DESC:
        body = (
            descriptor_id(fields["front"]),
            fields["total_dim"],
            fields["pivot_dim"],
            fields["tile_size"],
            descriptor_id(fields["contribution"]),
            descriptor_id(fields["factor"]),
            descriptor_id(fields["p"]),
            descriptor_id(fields["workspace"]),
            fields["pivot_policy"],
            fields["pivot_threshold"],
        )
    elif spec.descriptor_type == DescriptorType.CONTRIBUTION_DESC:
        body = (
            descriptor_id(fields["source"]),
            descriptor_id(fields["target"]),
            fields["child_id"],
            fields["parent_id"],
            fields["row_count"],
            fields["col_count"],
            fields["source_stride"],
            fields["target_stride"],
            int(fields["format"]),
            0,
        )
    elif spec.descriptor_type == DescriptorType.FACTOR_DESC:
        body = (
            descriptor_id(fields["l"]),
            descriptor_id(fields["u"]),
            descriptor_id(fields["p"]),
            fields["total_dim"],
            fields["pivot_dim"],
            int(fields["format"]),
            descriptor_id(fields["workspace"]),
            0,
            0,
            0,
        )
    elif spec.descriptor_type == DescriptorType.KERNEL_DESC:
        body = (
            descriptor_id(fields["a"]),
            descriptor_id(fields["b"]),
            descriptor_id(fields["c"]),
            fields["m"],
            fields["n"],
            fields["k"],
            int(fields["backend"]),
            int(fields["format"]),
            descriptor_id(fields["scale"]),
            fields["tile_size"],
        )
    elif spec.descriptor_type == DescriptorType.SOLVE_DESC:
        body = (
            descriptor_id(fields["factor"]),
            descriptor_id(fields["rhs"]),
            descriptor_id(fields["solution"]),
            descriptor_id(fields["p"]),
            descriptor_id(fields["workspace"]),
            fields["rhs_count"],
            int(fields["direction"]),
            int(fields["format"]),
            0,
            0,
        )
    elif spec.descriptor_type == DescriptorType.SCALE_DESC:
        def exponent_word(value: int | None) -> int:
            return SCALE_AUTO if value is None else value & 0xFFFFFFFF

        body = (
            exponent_word(fields["exponent_a"]),
            exponent_word(fields["exponent_b"]),
            SCALE_AUTO,
            fields["rounding"],
            fields["saturation"],
            fields["overflow"],
            0,
            0,
            0,
            0,
        )
    elif spec.descriptor_type == DescriptorType.DEPENDENCY_DESC:
        body = (len(fields["tokens"]),) + (0,) * 9
    else:
        raise CommandCompileError(f"unsupported descriptor type {spec.descriptor_type}")
    return Descriptor(
        descriptor_type=spec.descriptor_type,
        payload_offset=absolute_payload,
        payload_bytes=payload_bytes,
        body_words=tuple(int(value) for value in body),
    )


def _build_manifest(
    *,
    config: CommandCompilerConfig,
    pivot: PivotPolicyConfig,
    gemm: GemmConfig,
    total_bytes: int,
    regions: dict[str, MemoryRegion],
    descriptor_ids: dict[str, int],
    command_specs: Sequence[_CommandSpec],
    completion_templates: Sequence[Completion],
    ordered_nodes: Sequence[NodeCompileRecord],
    node_command_ids: dict[int, list[int]],
    node_tokens: dict[int, dict[str, int]],
    parent: Sequence[int],
    permutation: Sequence[int],
    map_tables: Sequence[Sequence[MapTableEntry]],
) -> dict[str, Any]:
    return {
        "abi": {
            "name": "command_descriptor",
            "version": COMMAND_SCHEMA_VERSION,
            "endianness": "little",
            "address_unit": "byte_offset_from_memory_image_base",
            "command_record_bytes": COMMAND_RECORD_BYTES,
            "descriptor_record_bytes": DESCRIPTOR_RECORD_BYTES,
            "completion_record_bytes": COMPLETION_RECORD_BYTES,
            "pivot_policy_field": "FRONT_DESC.body_words[8..9]",
        },
        "compiler": {
            "operation_scope": "factorization_and_solve" if config.emit_solve_commands else "factorization_only",
            "device_format": "FP32",
            "kernel_backend": (
                "SYSTEMC_INT32_GEMM_MODEL"
                if gemm.int32_gemm_enabled
                else "SYSTEMC_FP32_DEVICE_MODEL"
            ),
            "global_bfp": False,
            **asdict(config),
        },
        "gemm": {
            "backend": gemm.backend,
            "device_front_format": "FP32",
            "storage_format": (
                "INT32" if gemm.int32_gemm_enabled else "FP32"
            ),
            "effective_bits": gemm.effective_bits,
            "accumulator_bits": gemm.accumulator_bits,
            "k_block_max": gemm.k_block_max,
            "scale_policy": (
                "power_of_two_maxabs_auto"
                if gemm.scale_desc.exponent_a is None
                and gemm.scale_desc.exponent_b is None
                else "power_of_two_fixed_or_auto_per_operand"
            ),
            "scale_scope": "one_GEMM_SCHUR_call_per_tile",
            "output_scale": "AUTO (derived from the actual eA+eB)",
            "rounding_mode": "nearest_ties_even",
            "saturation_policy": (
                "CLAMP_AND_REPORT"
                if int(gemm.scale_desc.saturation) == 2
                else "FAIL"
            ),
            "overflow_policy": "FAIL",
            "conversion_rate_per_cycle": gemm.conversion_rate,
            "exponent_range": [-256, 256],
            "propagates_to_front_or_tree": False,
            "notes": (
                "INT32 storage / 30 effective quantization bits / INT64 "
                "accumulator is the supported range; the default 16-term dot "
                "product is < 2^62. Panel/TRSM and panel-internal updates "
                "remain FP32."
            ),
        },
        "pivot": {
            "policy": pivot.policy,
            "scope": "batch_level_uniform",
            "threshold_ratio": pivot.threshold_ratio,
            "threshold_bits_fp32": encode_pivot_threshold(pivot.threshold_ratio),
            "threshold_scale": "front_pre_elimination_max_abs",
            "threshold_scale_definition": (
                "max |A_front| over the assembled front before elimination "
                "of that front; computed once per front, never as a per-column "
                "candidate search"
            ),
            "static_pivot_failure_policy": (
                "report a numeric failure of the fixed elimination order "
                "(STATIC_PIVOT_FAILURE / PIVOT_UNSTABLE / NON_FINITE_VALUE); "
                "never publish a READY token; never claim the matrix is "
                "singular; no runtime switch to another policy; host retry "
                "restarts from the original inputs or fully rebuilt state"
            ),
            "p_vector": {
                "region_suffix": "p_vector",
                "format": "INT32",
                "length": "pivot_dim",
                "initial_value": PVECTOR_INIT_VALUE,
                "direction": "gather_within_pivot_block",
                "semantics": (
                    "p[i] = source pivot-block row for pivot position i; "
                    "STATIC writes the explicit identity; DYNAMIC_PANEL writes "
                    "the selected interchange within the pivot block"
                ),
                "omitted": False,
                "none_reference_allowed": False,
            },
        },
        "required_device_capabilities": {
            "always": [
                "command_fetch_fixed_width",
                "descriptor_reader",
                "byte_addressable_memory",
                "token_scoreboard",
                "completion_writeback",
                "panel_lu",
                "trsm",
                "gemm_schur",
                "assemble_extend_add",
                *(["tree_solve"] if config.emit_solve_commands else []),
            ],
            "STATIC": {
                "panel_lu_fixed_order": True,
                "pivot_search": False,
                "row_interchange": False,
                "hpu_atu_dynamic_mapping": "bypassed",
            },
            "DYNAMIC_PANEL": {
                "panel_lu_fixed_order": False,
                "pivot_search": "pivot_block_only",
                "row_interchange": "within_pivot_block",
                "hpu_atu_dynamic_mapping": "required",
                "cross_front_pivot": False,
                "update_row_promotion": False,
                "delayed_pivot": False,
                "dynamic_front_growth": False,
            },
            "executor_policy_refusal": (
                "an executor that does not implement the declared policy must "
                "refuse the PANEL_LU command with POLICY_NOT_SUPPORTED and "
                "must not publish its READY token"
            ),
        },
        "total_bytes": total_bytes,
        "memory_image": {
            "file": "memory_image.bin",
            "size": total_bytes,
            "regions": {name: asdict(region) for name, region in regions.items()},
        },
        "command_batch": {
            "region": "command_buffer",
            "command_count": len(command_specs),
            "record_bytes": COMMAND_RECORD_BYTES,
            "token_count": len(command_specs),
            "max_wait_tokens": config.max_wait_tokens,
            "static_batch": True,
            "ordering": "interleaved_topological",
            "admission_dependencies": (
                "NODE_BEGIN waits on the completion tokens of its children; "
                "the device must not infer the elimination forest itself"
            ),
        },
        "descriptor_table": {
            "region": "descriptor_table",
            "payload_region": "descriptor_payload",
            "descriptor_count": len(descriptor_ids),
            "record_bytes": DESCRIPTOR_RECORD_BYTES,
        },
        "completion_queue": {
            "region": "completion_status",
            "slot_count": len(completion_templates),
            "record_bytes": COMPLETION_RECORD_BYTES,
            "initialization": "expected_OK_templates; executor overwrites each slot",
        },
        "symbolic": {
            "pattern_source": "union_of_pattern_B_and_pattern_B_transpose",
            "node_count": len(ordered_nodes),
            "parent": list(parent),
            "permutation": [int(value) for value in permutation],
            "factorization_order": sibling_friendly_order(list(parent)),
        },
        "nodes": {
            str(node.node_id): {
                "parent_id": node.parent_id,
                "range": asdict(node.node_range),
                "front_indices": list(node.front_indices),
                "total_dim": node.total_dim,
                "pivot_dim": node.pivot_dim,
                "update_dim": node.update_dim,
                "regions": {
                    suffix: f"node.{node.node_id}.{suffix}"
                    for suffix in (
                        "front",
                        "update",
                        "factor_l",
                        "factor_u",
                        "p_vector",
                        "solve_workspace",
                    )
                },
                "descriptors": {
                    "front": descriptor_ids[f"node.{node.node_id}.front"],
                    "factor": descriptor_ids[f"node.{node.node_id}.factor"],
                    "contributions": [
                        descriptor_ids[
                            f"node.{node.node_id}.contribution.{entry.child_id}"
                        ]
                        for entry in map_tables[node.node_id]
                    ],
                },
                "command_ids": node_command_ids[node.node_id],
                "tokens": node_tokens[node.node_id],
            }
            for node in ordered_nodes
        },
        "expected_completions": [
            {
                "slot": index,
                "command_id": completion.command_id,
                "node_id": completion.node_id,
                "status": "OK",
            }
            for index, completion in enumerate(completion_templates)
        ],
    }


def _copy_region(image: bytearray, region: MemoryRegion, data: bytes) -> None:
    if len(data) != region.size:
        raise CommandCompileError("memory region payload size mismatch")
    image[region.offset : region.offset + region.size] = data


def _split_u64(value: int) -> tuple[int, int]:
    return value & 0xFFFFFFFF, value >> 32


def _pack_u32(values: Iterable[int]) -> bytes:
    packed = tuple(int(value) for value in values)
    return struct.pack(f"<{len(packed)}I", *packed) if packed else b""
