from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.command_codec import (
    COMMAND_RECORD_BYTES,
    COMMAND_SCHEMA_VERSION,
    COMPLETION_RECORD_BYTES,
    DESCRIPTOR_RECORD_BYTES,
    NONE,
    PVECTOR_ELEMENT_BYTES,
    PVECTOR_INIT_VALUE,
    SOLVE_DESC_PAYLOAD_KIND,
    DataFormat,
    DescriptorType,
    KernelBackend,
    Opcode,
    PivotPolicy,
    ScaleDesc,
    StatusCode,
    decode_commands,
    decode_completions,
    decode_descriptors,
    decode_pivot_threshold,
    validate_command_batch,
)
from src.dataStruct import MemoryRegion
from src.memory.planner import validate_region_layout
from src.scheduler.task_queue import sibling_friendly_order


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestValidationResult:
    node_count: int
    command_count: int
    descriptor_count: int
    completion_count: int


def validate_manifest(manifest_path: str | Path) -> ManifestValidationResult:
    path = Path(manifest_path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        _validate_abi(manifest)
        total_bytes = _as_int(manifest["total_bytes"], "total_bytes")
        image_meta = manifest["memory_image"]
        image_path = path.parent / image_meta["file"]
        image = image_path.read_bytes()
        if len(image) != total_bytes or _as_int(image_meta["size"], "image.size") != total_bytes:
            raise ManifestValidationError("memory image size does not match manifest")

        alignment = _as_int(manifest["compiler"]["alignment"], "alignment")
        regions = {
            name: MemoryRegion(
                offset=_as_int(value["offset"], f"{name}.offset"),
                size=_as_int(value["size"], f"{name}.size"),
            )
            for name, value in image_meta["regions"].items()
        }
        validate_region_layout(regions, total_bytes, alignment)
        for required in (
            "command_buffer",
            "descriptor_table",
            "descriptor_payload",
            "completion_status",
            "permutation_data",
            "rhs_data",
            "solution_data",
        ):
            if required not in regions:
                raise ManifestValidationError(f"missing memory region: {required}")

        batch = manifest["command_batch"]
        table = manifest["descriptor_table"]
        completion_meta = manifest["completion_queue"]
        command_count = _as_int(batch["command_count"], "command_count")
        descriptor_count = _as_int(table["descriptor_count"], "descriptor_count")
        completion_count = _as_int(completion_meta["slot_count"], "slot_count")
        if command_count <= 0 or descriptor_count <= 0:
            raise ManifestValidationError("command and descriptor counts must be positive")
        if completion_count != command_count:
            raise ManifestValidationError("completion slot count must equal command count")
        _require_region_size(
            regions, "command_buffer", command_count * COMMAND_RECORD_BYTES
        )
        _require_region_size(
            regions, "descriptor_table", descriptor_count * DESCRIPTOR_RECORD_BYTES
        )
        _require_region_size(
            regions, "completion_status", completion_count * COMPLETION_RECORD_BYTES
        )

        commands = decode_commands(_slice(image, regions["command_buffer"]))
        descriptors = decode_descriptors(
            _slice(image, regions["descriptor_table"]),
            memory_image_bytes=total_bytes,
        )
        completions = decode_completions(_slice(image, regions["completion_status"]))
        if len(commands) != command_count or len(descriptors) != descriptor_count:
            raise ManifestValidationError("decoded record count mismatch")
        token_count = _as_int(batch["token_count"], "token_count")
        validate_command_batch(
            commands,
            descriptors,
            image,
            token_count=token_count,
            max_wait_tokens=_as_int(batch["max_wait_tokens"], "max_wait_tokens"),
        )
        _validate_command_semantics(commands, descriptors, image, token_count)
        _validate_admission_dependencies(manifest, commands, descriptors, image)
        _validate_completion_templates(commands, completions)
        node_count = _validate_symbolic_and_nodes(manifest, commands, regions)
        _validate_solve_contract(manifest, commands, descriptors, image)
        _validate_descriptor_references(descriptors)
        _validate_pivot_contract(manifest, descriptors, image)
        _validate_device_precision(manifest, descriptors)
        _validate_reference_files(path.parent, manifest)
        return ManifestValidationResult(
            node_count=node_count,
            command_count=command_count,
            descriptor_count=descriptor_count,
            completion_count=completion_count,
        )
    except ManifestValidationError:
        raise
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(str(exc)) from exc


def _validate_abi(manifest: dict) -> None:
    abi = manifest.get("abi", {})
    version = _as_int(abi.get("version", -1), "abi.version")
    if abi.get("name") != "command_descriptor":
        raise ManifestValidationError("manifest is not a Command/Descriptor artifact")
    if version != COMMAND_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"manifest schema version {version} is not Command/Descriptor "
            f"v{COMMAND_SCHEMA_VERSION} (v1 artifacts must not be reinterpreted)"
        )
    if abi.get("endianness") != "little":
        raise ManifestValidationError("only little-endian artifacts are supported")
    expected = {
        "command_record_bytes": COMMAND_RECORD_BYTES,
        "descriptor_record_bytes": DESCRIPTOR_RECORD_BYTES,
        "completion_record_bytes": COMPLETION_RECORD_BYTES,
    }
    for name, value in expected.items():
        if _as_int(abi.get(name, -1), f"abi.{name}") != value:
            raise ManifestValidationError(f"manifest {name} mismatch")
    if abi.get("pivot_policy_field") != "FRONT_DESC.body_words[8..9]":
        raise ManifestValidationError("manifest pivot_policy_field is inconsistent")
    scope = manifest.get("compiler", {}).get("operation_scope", "factorization_and_solve")
    marker = abi.get("solve_desc_payload")
    if scope == "factorization_and_solve":
        if marker != SOLVE_DESC_PAYLOAD_KIND:
            raise ManifestValidationError(
                "solve-scope artifacts must declare "
                f'abi.solve_desc_payload = "{SOLVE_DESC_PAYLOAD_KIND}"'
            )
    elif marker is not None:
        raise ManifestValidationError(
            "factorization-only artifacts must not declare abi.solve_desc_payload"
        )


def _validate_completion_templates(commands, completions) -> None:
    for slot, (command, completion) in enumerate(zip(commands, completions)):
        if completion.command_id != command.command_id or completion.node_id != command.node_id:
            raise ManifestValidationError(
                f"completion template {slot} does not match its command"
            )
        if completion.status_code != StatusCode.OK:
            raise ManifestValidationError("completion template status must be OK")
        if any(
            (
                completion.pivot_count,
                completion.start_cycle,
                completion.finish_cycle,
                completion.read_bytes,
                completion.write_bytes,
                completion.stall_cycles,
                completion.overflow_count,
                completion.retry_count,
            )
        ):
            raise ManifestValidationError("completion template counters must start at zero")


def _validate_command_semantics(commands, descriptors, image: bytes, token_count: int) -> None:
    expected_descriptor_types = {
        Opcode.NODE_BEGIN: {DescriptorType.FRONT_DESC},
        Opcode.LOAD_FRONT: {DescriptorType.FRONT_DESC},
        Opcode.ASSEMBLE_EXTEND_ADD: {DescriptorType.CONTRIBUTION_DESC},
        Opcode.PANEL_LU: {DescriptorType.FRONT_DESC},
        Opcode.TRSM_LEFT: {DescriptorType.KERNEL_DESC},
        Opcode.TRSM_RIGHT: {DescriptorType.KERNEL_DESC},
        Opcode.GEMM_SCHUR: {DescriptorType.KERNEL_DESC},
        Opcode.STORE_FACTOR: {DescriptorType.FACTOR_DESC},
        Opcode.STORE_UPDATE: {DescriptorType.REGION_DESC},
        Opcode.SOLVE_FORWARD: {DescriptorType.SOLVE_DESC},
        Opcode.SOLVE_BACKWARD: {DescriptorType.SOLVE_DESC},
        Opcode.NODE_COMMIT: {DescriptorType.FRONT_DESC},
    }
    signals = {command.signal_token for command in commands}
    if signals != set(range(token_count)):
        raise ManifestValidationError("T03 requires one dense signal token per command")
    for command in commands:
        allowed = expected_descriptor_types.get(Opcode(command.opcode))
        if allowed is not None:
            if command.descriptor_id == 0xFFFFFFFF:
                raise ManifestValidationError(
                    f"command {command.command_id} is missing its descriptor"
                )
            actual = DescriptorType(descriptors[command.descriptor_id].descriptor_type)
            if actual not in allowed:
                raise ManifestValidationError(
                    f"command {command.command_id} uses the wrong descriptor type"
                )
        if command.wait_list_id != 0xFFFFFFFF:
            waits = descriptors[command.wait_list_id].dependency_tokens(
                image, token_count=token_count
            )
            if any(token >= command.command_id for token in waits):
                raise ManifestValidationError(
                    f"command {command.command_id} waits on a non-preceding token"
                )


def _validate_admission_dependencies(manifest, commands, descriptors, image: bytes) -> None:
    """NODE_BEGIN must wait on the completion token of every child.

    The device only sees dependency tokens and must never infer the
    elimination forest itself; this check keeps the admission contract
    compiler-visible instead of implicit.
    """
    parent = [_as_int(value, "parent") for value in manifest["symbolic"]["parent"]]
    node_count = len(parent)
    begin_command: dict[int, int] = {}
    commit_token: dict[int, int] = {}
    for command in commands:
        if command.opcode == Opcode.NODE_BEGIN:
            begin_command[command.node_id] = command.command_id
        elif command.opcode == Opcode.NODE_COMMIT:
            commit_token[command.node_id] = command.signal_token
    for node_id in range(node_count):
        if node_id not in begin_command or node_id not in commit_token:
            raise ManifestValidationError(
                f"node {node_id} is missing NODE_BEGIN or NODE_COMMIT"
            )
    for node_id in range(node_count):
        children = [child for child, p in enumerate(parent) if p == node_id]
        if not children:
            continue
        begin = commands[begin_command[node_id]]
        if begin.wait_list_id == 0xFFFFFFFF:
            raise ManifestValidationError(
                f"node {node_id} NODE_BEGIN is missing its admission dependency"
            )
        waits = set(
            descriptors[begin.wait_list_id].dependency_tokens(
                image, token_count=len(commands)
            )
        )
        for child in children:
            if commit_token[child] not in waits:
                raise ManifestValidationError(
                    f"node {node_id} NODE_BEGIN must wait for child {child} to complete"
                )


def _validate_symbolic_and_nodes(manifest, commands, regions) -> int:
    symbolic = manifest["symbolic"]
    if symbolic.get("pattern_source") != "union_of_pattern_B_and_pattern_B_transpose":
        raise ManifestValidationError("unsupported symbolic pattern source")
    node_count = _as_int(symbolic["node_count"], "node_count")
    nodes = manifest["nodes"]
    if node_count <= 0 or set(nodes) != {str(index) for index in range(node_count)}:
        raise ManifestValidationError("node IDs must form a dense unique 0..N-1 range")
    parent = [_as_int(value, "parent") for value in symbolic["parent"]]
    if len(parent) != node_count:
        raise ManifestValidationError("parent array length mismatch")
    try:
        expected_order = sibling_friendly_order(parent)
    except ValueError as exc:
        raise ManifestValidationError(f"invalid elimination forest: {exc}") from exc
    if symbolic["factorization_order"] != expected_order:
        raise ManifestValidationError("factorization order is inconsistent with forest")

    seen_commands: set[int] = set()
    max_front = _as_int(manifest["compiler"]["max_front_size"], "max_front_size")
    # Older manifests have no target scratchpad size; then only max_front applies.
    buffer_bytes = (
        _as_int(manifest["compiler"]["device_buffer_bytes"], "device_buffer_bytes")
        if "device_buffer_bytes" in manifest["compiler"]
        else None
    )
    scope = manifest["compiler"].get("operation_scope", "factorization_and_solve")
    if scope not in ("factorization_only", "factorization_and_solve"):
        raise ManifestValidationError("unknown compiler operation_scope")
    solve_opcodes = {Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD}
    for node_id in range(node_count):
        node = nodes[str(node_id)]
        if _as_int(node["parent_id"], "parent_id") != parent[node_id]:
            raise ManifestValidationError(f"node {node_id} parent mismatch")
        total = _as_int(node["total_dim"], "total_dim")
        pivot = _as_int(node["pivot_dim"], "pivot_dim")
        update = _as_int(node["update_dim"], "update_dim")
        if pivot <= 0 or total != pivot + update or total > max_front:
            raise ManifestValidationError(f"node {node_id} dimensions are invalid")
        if buffer_bytes is not None and total * total * 4 > buffer_bytes:
            raise ManifestValidationError(
                f"node {node_id} FP32 front exceeds device_buffer_bytes"
            )
        for region_name in node["regions"].values():
            if region_name not in regions:
                raise ManifestValidationError(
                    f"node {node_id} references unknown region {region_name}"
                )
        command_ids = [_as_int(value, "command_id") for value in node["command_ids"]]
        if seen_commands.intersection(command_ids):
            raise ManifestValidationError("a command is owned by multiple nodes")
        seen_commands.update(command_ids)
        if any(command_id >= len(commands) for command_id in command_ids):
            raise ManifestValidationError("node references an unknown command")
        if any(commands[command_id].node_id != node_id for command_id in command_ids):
            raise ManifestValidationError("command node ID disagrees with manifest")
        opcodes = {commands[command_id].opcode for command_id in command_ids}
        for required in (
            Opcode.NODE_BEGIN,
            Opcode.LOAD_FRONT,
            Opcode.PANEL_LU,
            Opcode.STORE_FACTOR,
            Opcode.NODE_COMMIT,
            *(() if scope == "factorization_only" else solve_opcodes),
        ):
            if required not in opcodes:
                raise ManifestValidationError(
                    f"node {node_id} is missing command {required.name}"
                )
        if scope == "factorization_only" and opcodes.intersection(solve_opcodes):
            raise ManifestValidationError("factorization-only artifact contains solve commands")
    if seen_commands != set(range(len(commands))):
        raise ManifestValidationError("manifest does not assign every command to a node")
    return node_count


def _validate_solve_contract(manifest, commands, descriptors, image: bytes) -> None:
    """T12 solve contract: payload tables, topological serial chain, capacity.

    The device only sees the descriptor stream; these invariants must hold in
    the artifact itself, not by host convention:

    * both SOLVE_DESCs of a node carry that node's ``front_indices`` payload;
    * the forward chain follows ``symbolic.factorization_order`` and the
      backward chain its reverse, each serial and waiting on its own commit;
    * every node's solve scratch fits the declared ``device_buffer_bytes``.
    """

    compiler = manifest.get("compiler", {})
    scope = compiler.get("operation_scope", "factorization_and_solve")
    solve_opcodes = {Opcode.SOLVE_FORWARD, Opcode.SOLVE_BACKWARD}
    if scope == "factorization_only":
        if any(command.opcode in solve_opcodes for command in commands):
            raise ManifestValidationError(
                "factorization-only artifact contains solve commands"
            )
        return

    solve_meta = manifest.get("solve")
    if not isinstance(solve_meta, dict):
        raise ManifestValidationError(
            "solve-scope manifests must describe the solve block"
        )
    for key, value in (
        ("chain", "serial"),
        ("forward_order", "factorization_order"),
        ("backward_order", "reverse_factorization_order"),
        ("solution_semantics", "zero-initialized accumulator; final = x_hat"),
        ("rhs_read_only", True),
        ("workspace_rows", ["forward_z", "backward_scratch"]),
        ("p_vector_semantics", "gather_within_pivot_block"),
    ):
        if solve_meta.get(key) != value:
            raise ManifestValidationError(f"solve.{key} is not the declared contract")
    depth = _as_int(solve_meta.get("row_buffer_depth", -1), "solve.row_buffer_depth")
    if depth <= 0:
        raise ManifestValidationError("solve.row_buffer_depth must be positive")

    symbolic = manifest["symbolic"]
    order = [
        _as_int(value, "factorization_order")
        for value in symbolic["factorization_order"]
    ]
    nodes = manifest["nodes"]
    forwards: list[tuple[int, int]] = []
    backwards: list[tuple[int, int]] = []
    for command in commands:
        if command.opcode == Opcode.SOLVE_FORWARD:
            forwards.append((command.command_id, command.node_id))
        elif command.opcode == Opcode.SOLVE_BACKWARD:
            backwards.append((command.command_id, command.node_id))
    if len(forwards) != len(order) or len(backwards) != len(order):
        raise ManifestValidationError(
            "every node must own exactly one SOLVE_FORWARD and one SOLVE_BACKWARD"
        )
    if [node for _, node in sorted(forwards)] != order:
        raise ManifestValidationError(
            "SOLVE_FORWARD chain is not in factorization (topological) order"
        )
    if [node for _, node in sorted(backwards)] != list(reversed(order)):
        raise ManifestValidationError(
            "SOLVE_BACKWARD chain is not in reverse factorization order"
        )

    commit_token: dict[int, int] = {}
    for command in commands:
        if command.opcode == Opcode.NODE_COMMIT:
            commit_token[command.node_id] = command.signal_token

    def waits_of(command) -> set[int]:
        if command.wait_list_id == NONE or command.wait_list_id >= len(descriptors):
            return set()
        return set(
            descriptors[command.wait_list_id].dependency_tokens(
                image, token_count=len(commands)
            )
        )

    # Revised 2026-10-11 (T12 design doc §2): there is no global solve gate;
    # each solve command waits on its own commit plus the previous solve
    # command, so wait lists stay bounded independent of the node count.
    previous: int | None = None
    for command_id, node in sorted(forwards):
        waits = waits_of(commands[command_id])
        if commit_token.get(node) not in waits:
            raise ManifestValidationError(
                f"SOLVE_FORWARD of node {node} must wait for its own commit"
            )
        if previous is not None and previous not in waits:
            raise ManifestValidationError("SOLVE_FORWARD chain is not serial")
        previous = command_id
    assert previous is not None
    for command_id, node in sorted(backwards):
        waits = waits_of(commands[command_id])
        if commit_token.get(node) not in waits:
            raise ManifestValidationError(
                f"SOLVE_BACKWARD of node {node} must wait for its own commit"
            )
        if previous not in waits:
            raise ManifestValidationError("SOLVE_BACKWARD chain is not serial")
        previous = command_id

    buffer_bytes = (
        _as_int(compiler["device_buffer_bytes"], "device_buffer_bytes")
        if "device_buffer_bytes" in compiler
        else None
    )
    for command_id, node in sorted(forwards + backwards):
        command = commands[command_id]
        if command.descriptor_id == NONE:
            raise ManifestValidationError("solve command is missing its descriptor")
        descriptor = descriptors[command.descriptor_id]
        if DescriptorType(descriptor.descriptor_type) != DescriptorType.SOLVE_DESC:
            raise ManifestValidationError("solve command does not use a SOLVE_DESC")
        factor_id = descriptor.body_words[0]
        if factor_id == NONE or factor_id >= len(descriptors):
            raise ManifestValidationError("SOLVE_DESC factor reference is unknown")
        factor = descriptors[factor_id]
        if DescriptorType(factor.descriptor_type) != DescriptorType.FACTOR_DESC:
            raise ManifestValidationError("SOLVE_DESC factor reference is not a FACTOR_DESC")
        total = _as_int(factor.body_words[3], "factor.total_dim")
        pivot = _as_int(factor.body_words[4], "factor.pivot_dim")
        if descriptor.payload_bytes != 4 * total:
            raise ManifestValidationError(
                "SOLVE_DESC payload_bytes must equal 4 * FACTOR_DESC.total_dim"
            )
        node_meta = nodes[str(node)]
        expected = [
            _as_int(value, "front_indices") for value in node_meta["front_indices"]
        ]
        variables = [int(value) for value in descriptor.solve_variables(image)]
        if variables != expected:
            raise ManifestValidationError(
                f"SOLVE_DESC payload of node {node} does not match front_indices"
            )
        row_range = node_meta["range"]
        start = _as_int(row_range["start"], "node range start")
        end = _as_int(row_range["end"], "node range end")
        if end - start != pivot or variables[:pivot] != list(range(start, end)):
            raise ManifestValidationError(
                f"SOLVE_DESC payload of node {node} is not the contiguous node range"
            )
        if buffer_bytes is not None:
            required = 4 * (depth * total + 2 * total + pivot)
            if required > buffer_bytes:
                raise ManifestValidationError(
                    f"node {node} solve scratch exceeds device_buffer_bytes"
                )


def _validate_descriptor_references(descriptors) -> None:
    count = len(descriptors)

    def require(index: int, allowed: set[DescriptorType], label: str) -> None:
        if index == NONE:
            return
        if index >= count or DescriptorType(descriptors[index].descriptor_type) not in allowed:
            raise ManifestValidationError(f"{label} has invalid descriptor reference")

    def require_mandatory(index: int, allowed: set[DescriptorType], label: str) -> None:
        if index == NONE:
            raise ManifestValidationError(f"{label} must not be NONE")
        require(index, allowed, label)

    for descriptor in descriptors:
        kind = DescriptorType(descriptor.descriptor_type)
        body = descriptor.body_words
        if kind == DescriptorType.FRONT_DESC:
            require(body[0], {DescriptorType.REGION_DESC}, "front region")
            require(body[4], {DescriptorType.CONTRIBUTION_DESC}, "front contribution")
            require_mandatory(body[5], {DescriptorType.FACTOR_DESC}, "front factor")
            require_mandatory(body[6], {DescriptorType.REGION_DESC}, "front P-vector")
            require_mandatory(body[7], {DescriptorType.REGION_DESC}, "front workspace")
        elif kind == DescriptorType.CONTRIBUTION_DESC:
            require(body[0], {DescriptorType.REGION_DESC}, "contribution source")
            require(body[1], {DescriptorType.REGION_DESC}, "contribution target")
        elif kind == DescriptorType.FACTOR_DESC:
            require(body[0], {DescriptorType.REGION_DESC}, "factor L region")
            require(body[1], {DescriptorType.REGION_DESC}, "factor U region")
            require_mandatory(body[2], {DescriptorType.REGION_DESC}, "factor P-vector")
            require(body[6], {DescriptorType.REGION_DESC}, "factor workspace")
        elif kind == DescriptorType.KERNEL_DESC:
            for index in (0, 1, 2):
                require(body[index], {DescriptorType.FRONT_DESC}, "kernel operand")
            try:
                backend = KernelBackend(body[6])
            except ValueError as error:
                raise ManifestValidationError(
                    "kernel backend is unknown"
                ) from error
            if backend == KernelBackend.SYSTEMC_INT32_GEMM_MODEL:
                # The local INT32 GEMM backend must reference a real
                # SCALE_DESC; a missing policy must never be silently
                # defaulted at execution time.
                require_mandatory(
                    body[8], {DescriptorType.SCALE_DESC}, "INT32 kernel scale"
                )
                _check_scale_desc(descriptors, body[8])
            else:
                if body[8] != NONE:
                    raise ManifestValidationError(
                        "FP32 kernel must not reference a SCALE_DESC "
                        "(scale_desc_id must be NONE)"
                    )
        elif kind == DescriptorType.SOLVE_DESC:
            require_mandatory(body[0], {DescriptorType.FACTOR_DESC}, "solve factor")
            require(body[1], {DescriptorType.REGION_DESC}, "solve RHS region")
            require(body[2], {DescriptorType.REGION_DESC}, "solve solution region")
            require_mandatory(body[3], {DescriptorType.REGION_DESC}, "solve P-vector")
            require(body[4], {DescriptorType.REGION_DESC}, "solve workspace")


def _check_scale_desc(descriptors, index: int) -> None:
    """Validate a referenced SCALE_DESC body (T10 frozen word semantics)."""

    if index >= len(descriptors):
        raise ManifestValidationError("kernel scale reference is unknown")
    descriptor = descriptors[index]
    if DescriptorType(descriptor.descriptor_type) != DescriptorType.SCALE_DESC:
        raise ManifestValidationError("kernel scale reference is not a SCALE_DESC")
    body = descriptor.body_words
    if any(word != 0 for word in body[6:10]):
        raise ManifestValidationError("SCALE_DESC reserved words 6..9 must be zero")
    try:
        ScaleDesc.from_body_words(body)
    except ValueError as error:
        raise ManifestValidationError(f"SCALE_DESC policy rejected: {error}") from error


def _check_p_vector_reference(
    descriptors, image: bytes, index: int, pivot_dim: int, label: str
) -> int:
    """Follow a P-vector descriptor reference down to the actual data region."""

    if index == NONE:
        raise ManifestValidationError(f"{label} P-vector reference must not be NONE")
    if index >= len(descriptors):
        raise ManifestValidationError(f"{label} P-vector reference is unknown")
    region = descriptors[index]
    if DescriptorType(region.descriptor_type) != DescriptorType.REGION_DESC:
        raise ManifestValidationError(f"{label} P-vector reference is not a REGION_DESC")
    body = region.body_words
    base_addr = body[0] | (body[1] << 32)
    byte_size = body[2] | (body[3] << 32)
    rows, cols = body[5], body[6]
    if body[7] != int(DataFormat.INT32):
        raise ManifestValidationError(f"{label} P-vector region must use INT32")
    if pivot_dim <= 0:
        raise ManifestValidationError(f"{label} pivot_dim must be positive")
    if rows * cols != pivot_dim or byte_size != pivot_dim * PVECTOR_ELEMENT_BYTES:
        raise ManifestValidationError(
            f"{label} P-vector region capacity must equal pivot_dim {pivot_dim}"
        )
    end = base_addr + byte_size
    if end > len(image):
        raise ManifestValidationError(
            f"{label} P-vector region exceeds the memory image"
        )
    words = struct.unpack(f"<{pivot_dim}i", image[base_addr:end])
    if any(word != PVECTOR_INIT_VALUE for word in words):
        raise ManifestValidationError(
            f"{label} P-vector must be initialized to the INT32 sentinel "
            f"{PVECTOR_INIT_VALUE}; a zero array is never identity"
        )
    return index


def _validate_pivot_contract(manifest, descriptors, image: bytes) -> None:
    pivot = manifest.get("pivot")
    if not isinstance(pivot, dict):
        raise ManifestValidationError("manifest is missing the pivot policy block")
    policy_name = pivot.get("policy")
    if not isinstance(policy_name, str):
        raise ManifestValidationError("manifest pivot policy must be a name")
    try:
        policy = PivotPolicy[policy_name]
    except KeyError as exc:
        raise ManifestValidationError(f"unknown pivot policy {policy_name!r}") from exc
    threshold_bits = _as_int(pivot.get("threshold_bits_fp32", -1), "threshold_bits_fp32")
    if pivot.get("scope") != "batch_level_uniform":
        raise ManifestValidationError("pivot policy scope must be batch_level_uniform")
    if pivot.get("threshold_scale") != "front_pre_elimination_max_abs":
        raise ManifestValidationError("pivot threshold scale must be explicit")
    if not pivot.get("static_pivot_failure_policy"):
        raise ManifestValidationError("manifest must declare the static failure policy")
    effective_threshold = decode_pivot_threshold(threshold_bits)
    if float(pivot.get("threshold_ratio", -1.0)) != effective_threshold:
        raise ManifestValidationError(
            "manifest threshold_ratio does not equal the binary32 threshold bits"
        )
    p_vector_meta = pivot.get("p_vector", {})
    if p_vector_meta.get("omitted") is not False:
        raise ManifestValidationError("v2 artifacts must keep explicit P-vector regions")
    if p_vector_meta.get("none_reference_allowed") is not False:
        raise ManifestValidationError("NONE P-vector references are not allowed in v2")

    front_count = 0
    for descriptor in descriptors:
        kind = DescriptorType(descriptor.descriptor_type)
        body = descriptor.body_words
        if kind == DescriptorType.FRONT_DESC:
            front_count += 1
            if body[8] != int(policy):
                raise ManifestValidationError(
                    "FRONT_DESC pivot policy disagrees with the manifest policy"
                )
            if body[9] != threshold_bits:
                raise ManifestValidationError(
                    "FRONT_DESC pivot threshold disagrees with the manifest threshold"
                )
            front_p = _check_p_vector_reference(
                descriptors, image, body[6], body[2], "front"
            )
            factor = descriptors[body[5]]
            if factor.body_words[2] != front_p:
                raise ManifestValidationError(
                    "FRONT_DESC and FACTOR_DESC must reference one P-vector region"
                )
        elif kind == DescriptorType.FACTOR_DESC:
            _check_p_vector_reference(
                descriptors, image, body[2], body[4], "factor"
            )
        elif kind == DescriptorType.SOLVE_DESC:
            factor = descriptors[body[0]]
            solve_p = _check_p_vector_reference(
                descriptors, image, body[3], factor.body_words[4], "solve"
            )
            if factor.body_words[2] != solve_p:
                raise ManifestValidationError(
                    "SOLVE_DESC and FACTOR_DESC must reference one P-vector region"
                )
    if front_count == 0:
        raise ManifestValidationError("artifact contains no FRONT_DESC")

    capabilities = manifest.get("required_device_capabilities", {})
    if not isinstance(capabilities, dict) or not capabilities.get("always"):
        raise ManifestValidationError("manifest must declare required device capabilities")
    policy_caps = capabilities.get(policy.name)
    if not isinstance(policy_caps, dict):
        raise ManifestValidationError("manifest must declare policy device capabilities")
    if policy == PivotPolicy.STATIC and policy_caps.get("hpu_atu_dynamic_mapping") != "bypassed":
        raise ManifestValidationError("STATIC must bypass the HPU/ATU dynamic mapping")
    if (
        policy == PivotPolicy.DYNAMIC_PANEL
        and policy_caps.get("hpu_atu_dynamic_mapping") != "required"
    ):
        raise ManifestValidationError("DYNAMIC_PANEL requires the HPU/ATU dynamic mapping")

    preprocess = manifest.get("preprocess")
    if not isinstance(preprocess, dict):
        raise ManifestValidationError("manifest is missing the preprocess block")
    matching = preprocess.get("matching", {})
    if matching.get("method") not in {"none", "weighted_matching"}:
        raise ManifestValidationError("unknown matching method in preprocess block")
    transform = preprocess.get("transform", {})
    row_match = [
        _as_int(value, "row_perm_match") for value in transform.get("row_perm_match", [])
    ]
    ordering = [
        _as_int(value, "ordering_perm") for value in transform.get("ordering_perm", [])
    ]
    rhs_map = [_as_int(value, "rhs_row_map") for value in transform.get("rhs_row_map", [])]
    dimension = len(ordering)
    if dimension == 0 or len(row_match) != dimension or len(rhs_map) != dimension:
        raise ManifestValidationError("preprocess permutation arrays have wrong length")
    for name, values in (
        ("row_perm_match", row_match),
        ("ordering_perm", ordering),
        ("rhs_row_map", rhs_map),
    ):
        if sorted(values) != list(range(dimension)):
            raise ManifestValidationError(f"{name} is not a 0..N-1 permutation")
    if rhs_map != [row_match[ordering[index]] for index in range(dimension)]:
        raise ManifestValidationError("rhs_row_map is inconsistent with the two permutations")
    if matching.get("method") == "weighted_matching":
        if matching.get("full_matching") is not True:
            raise ManifestValidationError("weighted_matching requires a full matching")
        matched_columns = matching.get("matched_column_of_row", [])
        if sorted(_as_int(value, "match") for value in matched_columns) != list(
            range(dimension)
        ):
            raise ManifestValidationError("matching columns are not a bijection")


def _validate_device_precision(manifest, descriptors) -> None:
    compiler = manifest["compiler"]
    if compiler.get("device_format") != "FP32" or compiler.get("global_bfp") is not False:
        raise ManifestValidationError("Command v2 main path must be FP32 without global BFP")
    for descriptor in descriptors:
        if descriptor.descriptor_type == DescriptorType.REGION_DESC:
            data_format = DataFormat(descriptor.body_words[7])
            if data_format not in {DataFormat.FP32, DataFormat.INT32}:
                raise ManifestValidationError(
                    "device image contains a non-current data format"
                )


def _validate_reference_files(out_dir: Path, manifest: dict) -> None:
    verification = manifest.get("verification")
    if verification is None:
        return
    if verification.get("device_memory_contains_fp64_reference") is not False:
        raise ManifestValidationError("FP64 reference must not be device input")
    if verification.get("residual_target") != "original_A_x_equals_b":
        raise ManifestValidationError("residuals must target the original system")
    matrix_dim = _as_int(manifest["matrix"]["n"], "matrix.n")
    nodes = manifest["nodes"]
    front_bytes = 8 * sum(
        _as_int(nodes[name]["total_dim"], "total_dim") ** 2 for name in nodes
    )
    expected_sizes = {
        verification["rhs_reference_file"]: matrix_dim * 8,
        verification["original_matrix_reference_file"]: matrix_dim * matrix_dim * 8,
        verification["original_rhs_reference_file"]: matrix_dim * 8,
        verification["row_scale_exponent_file"]: matrix_dim * 2,
        verification["reference_front_file"]: front_bytes,
    }
    solution_file = verification.get("solution_reference_file")
    if solution_file is None:
        if verification.get("solution_reference_space") is not None:
            raise ManifestValidationError(
                "missing golden solution must not declare a solution space"
            )
    else:
        if verification.get("solution_reference_space") != "original_variable_order":
            raise ManifestValidationError(
                "golden solution must be in original variable order"
            )
        expected_sizes[solution_file] = matrix_dim * 8

    host_reference = verification.get("host_reference")
    if not isinstance(host_reference, dict):
        raise ManifestValidationError("verification must declare the host reference block")
    performed = host_reference.get("performed")
    if performed is True:
        if host_reference.get("policy") != manifest["pivot"]["policy"]:
            raise ManifestValidationError(
                "host reference policy must match the artifact policy"
            )
        x_hat_file = host_reference.get("x_hat_reference_file")
        if not isinstance(x_hat_file, str):
            raise ManifestValidationError(
                "performed host reference must declare its x_hat file"
            )
        if host_reference.get("x_hat_reference_space") != "ordering_perm_variable_order":
            raise ManifestValidationError("x_hat must be in the ordering_perm space")
        expected_sizes[x_hat_file] = matrix_dim * 8
        for field in (
            "original_reference_residual_norm",
            "original_reference_backward_error",
        ):
            if not isinstance(host_reference.get(field), (int, float)):
                raise ManifestValidationError(
                    f"performed host reference must report {field}"
                )
    elif performed is False:
        if host_reference.get("x_hat_reference_file") is not None:
            raise ManifestValidationError(
                "host reference not performed must not declare an x_hat file"
            )
        for field in (
            "original_reference_residual_norm",
            "original_reference_backward_error",
        ):
            if host_reference.get(field) is not None:
                raise ManifestValidationError(
                    f"host reference not performed must not report {field}"
                )
    else:
        raise ManifestValidationError("host_reference.performed must be a boolean")

    for name, expected in expected_sizes.items():
        if (out_dir / name).stat().st_size != expected:
            raise ManifestValidationError(f"reference file {name} has wrong size")


def _require_region_size(regions, name: str, expected: int) -> None:
    if regions[name].size != expected:
        raise ManifestValidationError(f"{name} has wrong size")


def _slice(image: bytes, region: MemoryRegion) -> bytes:
    return image[region.offset : region.offset + region.size]


def _as_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestValidationError(f"{label}: expected integer")
    return value
