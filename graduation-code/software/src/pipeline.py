from __future__ import annotations

import struct
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import MatrixRankWarning

from src.config import PipelineConfig
from src.dataStruct import (
    ABI_VERSION,
    NODE_TASK_BYTE_SIZE,
    ROOT_PARENT_ID,
    GlobalMemoryPlan,
    NodeRange,
    NodeTask,
    SymbolicResult,
)
from src.equilibration import EquilibrationResult, equilibrate_system
from src.io import write_front_data, write_manifest, write_map_table, write_tasks
from src.matrix_io import load_matrix, load_vector
from src.memory.planner import plan_memory
from src.quant.bfp_quant import (
    QuantizationStats,
    flatten_quantized_source,
    quant_limit,
    quantize_local_contribution,
)
from src.scheduler.map_gen import generate_map_tables
from src.scheduler.task_queue import sibling_friendly_order
from src.symbolic.etree import children_from_parent
from src.symbolic.fill import require_structurally_symmetric, symbolic_fill_pattern
from src.symbolic.ordering import apply_permutation, compute_ordering
from src.symbolic.supernode import (
    build_front_indices_from_filled,
    build_supernode_parent,
    build_supernodes_from_filled,
)
from src.verify.manifest import validate_manifest
from src.verify.metrics import residual_norm


@dataclass(frozen=True)
class PipelineOutputs:
    out_dir: Path
    tasks_path: Path
    map_table_path: Path
    front_q_path: Path
    front_e_path: Path
    memory_image_path: Path
    reference_front_path: Path
    rhs_reference_path: Path
    original_matrix_reference_path: Path
    original_rhs_reference_path: Path
    row_scale_exponents_path: Path
    solution_reference_path: Path
    manifest_path: Path
    residual_norm: float
    original_residual_norm: float
    node_count: int
    task_count: int


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    original_matrix = load_matrix(
        path=config.matrix.path,
        n=config.matrix.n,
        density=config.matrix.density,
        seed=config.matrix.seed,
    )
    require_structurally_symmetric(original_matrix)

    rhs_original, solution_original, rhs_source_kind = _prepare_rhs(
        original_matrix, config
    )
    equilibration = equilibrate_system(
        original_matrix,
        rhs_original,
        config.equilibration,
    )
    matrix = equilibration.matrix
    rhs_scaled = equilibration.rhs

    symbolic = run_symbolic_analysis(matrix, config)
    permuted = apply_permutation(
        matrix, np.asarray(symbolic.permutation, dtype=np.int32)
    )
    matrix_dim = int(permuted.shape[0])
    permutation = np.asarray(symbolic.permutation, dtype=np.int32)
    rhs_permuted = rhs_scaled[permutation]
    solution_permuted = solution_original[permutation]

    node_range_tuples = [node_range.as_tuple() for node_range in symbolic.node_ranges]
    map_tables = generate_map_tables(
        node_range_tuples,
        symbolic.parent,
        symbolic.front_indices,
    )

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = out_dir / "tasks.bin"
    map_path = out_dir / "map_table.bin"
    front_q_path = out_dir / "front_q.bin"
    front_e_path = out_dir / "front_e.bin"
    rhs_q_path = out_dir / "rhs_q.bin"
    rhs_e_path = out_dir / "rhs_e.bin"
    memory_image_path = out_dir / "memory_image.bin"
    reference_front_path = out_dir / "reference_front_f64.bin"
    rhs_reference_path = out_dir / "rhs_f64.bin"
    original_matrix_reference_path = out_dir / "original_matrix_f64.bin"
    original_rhs_reference_path = out_dir / "original_rhs_f64.bin"
    row_scale_exponents_path = out_dir / "row_scale_e.bin"
    solution_reference_path = out_dir / "x_reference_f64.bin"
    manifest_path = out_dir / "manifest.json"

    front_q_path.write_bytes(b"")
    front_e_path.write_bytes(b"")
    rhs_q_path.write_bytes(b"")
    rhs_e_path.write_bytes(b"")
    reference_front_path.write_bytes(b"")

    map_offsets = write_map_table(str(map_path), map_tables)
    map_sizes = _sizes_from_offsets(map_offsets, map_path.stat().st_size)

    q_sizes: List[int] = []
    e_sizes: List[int] = []
    q_offsets: List[int] = []
    e_offsets: List[int] = []
    source_shapes: List[tuple[int, int]] = []
    source_exponents: List[int] = []
    reference_offsets: List[int] = []
    quant_stats: List[QuantizationStats] = []
    clip_total = 0
    sat_total = 0
    q_cursor = 0
    e_cursor = 0
    reference_cursor = 0

    for node_range, front_indices in zip(symbolic.node_ranges, symbolic.front_indices):
        local_contribution = extract_local_contribution(
            permuted, front_indices, node_range
        )
        reference_offsets.append(reference_cursor)
        reference_bytes = local_contribution.astype("<f8", copy=False).tobytes(order="C")
        with reference_front_path.open("ab") as reference_file:
            reference_file.write(reference_bytes)
        reference_cursor += len(reference_bytes)

        source = quantize_local_contribution(local_contribution, config.quant)
        q_list, e_list = flatten_quantized_source(source)

        q_offsets.append(q_cursor)
        e_offsets.append(e_cursor)
        q_bytes, e_bytes = write_front_data(str(front_q_path), str(front_e_path), q_list, e_list)
        q_sizes.append(q_bytes)
        e_sizes.append(e_bytes)
        q_cursor += q_bytes
        e_cursor += e_bytes

        source_shapes.append(source.shape)
        source_exponents.append(source.exponent)
        quant_stats.append(source.stats)
        clip_total += source.stats.clip_count
        sat_total += source.stats.sat_count

    rhs_source = quantize_local_contribution(rhs_permuted.reshape(-1, 1), config.quant)
    rhs_q_values, rhs_e_values = flatten_quantized_source(rhs_source)
    write_front_data(
        str(rhs_q_path),
        str(rhs_e_path),
        rhs_q_values,
        rhs_e_values,
    )
    rhs_reference_path.write_bytes(rhs_permuted.astype("<f8").tobytes(order="C"))
    solution_reference_path.write_bytes(
        solution_permuted.astype("<f8").tobytes(order="C")
    )
    original_matrix_reference_path.write_bytes(
        original_matrix.toarray().astype("<f8", copy=False).tobytes(order="C")
    )
    original_rhs_reference_path.write_bytes(
        rhs_original.astype("<f8", copy=False).tobytes(order="C")
    )
    row_scale_exponents_path.write_bytes(
        equilibration.row_scale_exponents.astype("<i2", copy=False).tobytes(
            order="C"
        )
    )

    (
        update_q_sizes,
        update_e_sizes,
        l_factor_sizes,
        u_factor_sizes,
        p_vector_sizes,
        node_meta_sizes,
        solve_workspace_sizes,
    ) = estimate_work_region_sizes(symbolic.node_ranges, symbolic.front_indices)

    global_plan, mem_plans, total_bytes = plan_memory(
        len(symbolic.node_ranges),
        matrix_dim,
        q_sizes,
        e_sizes,
        map_sizes,
        update_q_sizes=update_q_sizes,
        update_e_sizes=update_e_sizes,
        l_factor_sizes=l_factor_sizes,
        u_factor_sizes=u_factor_sizes,
        p_vector_sizes=p_vector_sizes,
        node_meta_sizes=node_meta_sizes,
        solve_workspace_sizes=solve_workspace_sizes,
        task_record_size=NODE_TASK_BYTE_SIZE,
        align=config.memory.alignment,
    )

    tasks = build_node_tasks(
        symbolic.parent,
        symbolic.node_ranges,
        symbolic.front_indices,
        mem_plans,
        map_sizes,
    )
    task_order = sibling_friendly_order(symbolic.parent)
    ordered_tasks = [tasks[node_id] for node_id in task_order]
    write_tasks(str(tasks_path), ordered_tasks)

    _write_memory_image(
        memory_image_path=memory_image_path,
        total_bytes=total_bytes,
        global_plan=global_plan,
        mem_plans=mem_plans,
        ordered_tasks=ordered_tasks,
        permutation=symbolic.permutation,
        front_q_data=front_q_path.read_bytes(),
        front_e_data=front_e_path.read_bytes(),
        map_data=map_path.read_bytes(),
        q_offsets=q_offsets,
        e_offsets=e_offsets,
        map_offsets=map_offsets,
        q_sizes=q_sizes,
        e_sizes=e_sizes,
        map_sizes=map_sizes,
        rhs_q_data=rhs_q_path.read_bytes(),
        rhs_e_data=rhs_e_path.read_bytes(),
    )

    residual = residual_norm(permuted, solution_permuted, rhs_permuted)
    original_residual = residual_norm(
        original_matrix, solution_original, rhs_original
    )
    manifest = build_manifest(
        config=config,
        symbolic=symbolic,
        mem_plans=mem_plans,
        global_plan=global_plan,
        total_bytes=total_bytes,
        q_offsets=q_offsets,
        e_offsets=e_offsets,
        map_offsets=map_offsets,
        reference_offsets=reference_offsets,
        source_shapes=source_shapes,
        source_exponents=source_exponents,
        quant_stats=quant_stats,
        clip_total=clip_total,
        sat_total=sat_total,
        task_order=task_order,
        residual=residual,
        original_residual=original_residual,
        actual_matrix_dim=matrix_dim,
        rhs_source=rhs_source,
        rhs_source_kind=rhs_source_kind,
        equilibration=equilibration,
        output_sizes={
            "tasks.bin": tasks_path.stat().st_size,
            "map_table.bin": map_path.stat().st_size,
            "front_q.bin": front_q_path.stat().st_size,
            "front_e.bin": front_e_path.stat().st_size,
            "rhs_q.bin": rhs_q_path.stat().st_size,
            "rhs_e.bin": rhs_e_path.stat().st_size,
            "memory_image.bin": memory_image_path.stat().st_size,
            "reference_front_f64.bin": reference_front_path.stat().st_size,
            "rhs_f64.bin": rhs_reference_path.stat().st_size,
            "original_matrix_f64.bin":
                original_matrix_reference_path.stat().st_size,
            "original_rhs_f64.bin": original_rhs_reference_path.stat().st_size,
            "row_scale_e.bin": row_scale_exponents_path.stat().st_size,
            "x_reference_f64.bin": solution_reference_path.stat().st_size,
        },
    )
    write_manifest(str(manifest_path), manifest)
    validate_manifest(manifest_path)

    return PipelineOutputs(
        out_dir=out_dir,
        tasks_path=tasks_path,
        map_table_path=map_path,
        front_q_path=front_q_path,
        front_e_path=front_e_path,
        memory_image_path=memory_image_path,
        reference_front_path=reference_front_path,
        rhs_reference_path=rhs_reference_path,
        original_matrix_reference_path=original_matrix_reference_path,
        original_rhs_reference_path=original_rhs_reference_path,
        row_scale_exponents_path=row_scale_exponents_path,
        solution_reference_path=solution_reference_path,
        manifest_path=manifest_path,
        residual_norm=residual,
        original_residual_norm=original_residual,
        node_count=len(symbolic.node_ranges),
        task_count=len(ordered_tasks),
    )


def run_symbolic_analysis(matrix, config: PipelineConfig) -> SymbolicResult:
    permutation = compute_ordering(matrix, config.ordering.method)
    permuted = apply_permutation(matrix, permutation)
    filled = symbolic_fill_pattern(permuted)
    supernodes = build_supernodes_from_filled(
        filled.parent,
        filled.columns,
        max_size=config.ordering.max_supernode_size,
    )
    node_ranges = build_node_ranges(supernodes)
    node_parent = build_supernode_parent(filled.parent, supernodes)
    front_indices = build_front_indices_from_filled(supernodes, filled.columns)
    return SymbolicResult(
        permutation=permutation.astype(np.int32).tolist(),
        parent=node_parent,
        supernodes=supernodes,
        node_ranges=node_ranges,
        front_indices=front_indices,
    )


def build_node_ranges(supernodes: List[List[int]]) -> List[NodeRange]:
    ranges: List[NodeRange] = []
    start = 0
    for supernode in supernodes:
        end = start + len(supernode)
        ranges.append(NodeRange(start=start, end=end))
        start = end
    return ranges


def extract_local_contribution(
    matrix,
    front_indices: List[int],
    node_range: NodeRange,
) -> np.ndarray:
    """Return the unique original-matrix contribution owned by one node.

    The pivot variables are first in ``front_indices``.  Original entries
    touching a pivot row or column are owned by this node; the update/update
    block starts at zero and is populated only by child Schur contributions.
    """

    pivot_dim = node_range.size
    if front_indices[:pivot_dim] != list(range(node_range.start, node_range.end)):
        raise ValueError("front pivot prefix does not match node range")
    local = matrix[front_indices][:, front_indices].toarray().astype(np.float64)
    if pivot_dim < len(front_indices):
        local[pivot_dim:, pivot_dim:] = 0.0
    return local


def estimate_work_region_sizes(
    node_ranges: List[NodeRange],
    front_indices: List[List[int]],
) -> tuple[
    List[int],
    List[int],
    List[int],
    List[int],
    List[int],
    List[int],
    List[int],
]:
    update_q_sizes: List[int] = []
    update_e_sizes: List[int] = []
    l_factor_sizes: List[int] = []
    u_factor_sizes: List[int] = []
    p_vector_sizes: List[int] = []
    node_meta_sizes: List[int] = []
    solve_workspace_sizes: List[int] = []

    for node_range, front in zip(node_ranges, front_indices):
        pivot_dim = node_range.size
        total_dim = len(front)
        update_dim = max(total_dim - pivot_dim, 0)

        update_q_sizes.append(update_dim * update_dim * 4)
        update_e_sizes.append(2 if update_dim else 0)
        l_factor_sizes.append(total_dim * pivot_dim * 4)
        u_factor_sizes.append(pivot_dim * total_dim * 4)
        p_vector_sizes.append(pivot_dim * 2)
        node_meta_sizes.append(64)
        solve_workspace_sizes.append(pivot_dim * 16)

    return (
        update_q_sizes,
        update_e_sizes,
        l_factor_sizes,
        u_factor_sizes,
        p_vector_sizes,
        node_meta_sizes,
        solve_workspace_sizes,
    )


def build_node_tasks(
    parent: List[int],
    node_ranges: List[NodeRange],
    front_indices: List[List[int]],
    mem_plans,
    map_sizes: List[int],
) -> List[NodeTask]:
    children = children_from_parent(np.asarray(parent, dtype=np.int32))
    tasks: List[NodeTask] = []

    for node_id, node_range in enumerate(node_ranges):
        parent_id = parent[node_id]
        pivot_dim = node_range.size
        total_dim = len(front_indices[node_id])
        flags = 0
        if len(children[node_id]) == 0:
            flags |= 1
        if parent_id < 0:
            flags |= 2

        plan = mem_plans[node_id]
        tile_count = _ceil_div(pivot_dim, 16)
        tail_dim = pivot_dim % 16 or min(pivot_dim, 16)
        tasks.append(
            NodeTask(
                node_id=node_id,
                flags=flags,
                parent_id=parent_id if parent_id >= 0 else ROOT_PARENT_ID,
                children_count=len(children[node_id]),
                total_dim=total_dim,
                pivot_dim=pivot_dim,
                tile_count=tile_count,
                tail_dim=tail_dim,
                map_table_bytes=map_sizes[node_id],
                reserved=0,
                front_q_addr=plan.front_q.offset,
                front_e_addr=plan.front_e.offset,
                update_q_addr=plan.update_q.offset,
                update_e_addr=plan.update_e.offset,
                map_table_addr=plan.map_table.offset,
                l_factor_addr=plan.l_factor.offset,
                u_factor_addr=plan.u_factor.offset,
                p_vector_addr=plan.p_vector.offset,
                node_meta_addr=plan.node_meta.offset,
                solve_workspace_addr=plan.solve_workspace.offset,
                reserved_addr0=0,
                reserved_addr1=0,
            )
        )

    return tasks


def build_manifest(
    *,
    config: PipelineConfig,
    symbolic: SymbolicResult,
    mem_plans,
    global_plan: GlobalMemoryPlan,
    total_bytes: int,
    q_offsets: List[int],
    e_offsets: List[int],
    map_offsets: List[int],
    reference_offsets: List[int],
    source_shapes: List[tuple[int, int]],
    source_exponents: List[int],
    quant_stats: List[QuantizationStats],
    clip_total: int,
    sat_total: int,
    task_order: List[int],
    residual: float,
    original_residual: float,
    actual_matrix_dim: int,
    rhs_source,
    rhs_source_kind: str,
    equilibration: EquilibrationResult,
    output_sizes: dict[str, int],
) -> dict:
    return {
        "abi": {
            "version": ABI_VERSION,
            "node_task_byte_size": NODE_TASK_BYTE_SIZE,
            "endianness": "little",
            "map_table_format": "u32_count_then_u32_child_row_count_col_count_arrays",
        },
        "config": {
            "ordering": asdict(config.ordering),
            "quant": asdict(config.quant),
            "equilibration": asdict(config.equilibration),
            "memory": asdict(config.memory),
        },
        "matrix": {
            "path": config.matrix.path,
            "n": actual_matrix_dim,
            "density": config.matrix.density,
            "seed": config.matrix.seed,
            "structurally_symmetric": True,
        },
        "total_bytes": total_bytes,
        "memory_image": {
            "file": "memory_image.bin",
            "size": total_bytes,
            "global_regions": {
                name: asdict(getattr(global_plan, name))
                for name in (
                    "task_queue",
                    "permutation",
                    "rhs_q",
                    "rhs_e",
                    "solution_q",
                    "solution_e",
                )
            },
        },
        "output_sizes": output_sizes,
        "verification": {
            "reference_residual_norm": residual,
            "original_reference_residual_norm": original_residual,
            "reference_front_file": "reference_front_f64.bin",
            "rhs_reference_file": "rhs_f64.bin",
            "original_matrix_reference_file": "original_matrix_f64.bin",
            "original_rhs_reference_file": "original_rhs_f64.bin",
            "solution_reference_file": "x_reference_f64.bin",
        },
        "equilibration": {
            "mode": config.equilibration.mode,
            "equation": "D_r * A * x = D_r * b",
            "solution_requires_unscale": False,
            "row_scale_exponent_file": "row_scale_e.bin",
            "row_scale_exponent_dtype": "int16",
            "row_scale_exponent_count": actual_matrix_dim,
            "row_max_before": {
                "min_nonzero": equilibration.row_max_before_min,
                "max": equilibration.row_max_before_max,
            },
            "row_max_after": {
                "min_nonzero": equilibration.row_max_after_min,
                "max": equilibration.row_max_after_max,
            },
        },
        "symbolic": {
            "node_count": len(symbolic.node_ranges),
            "supernode_count": len(symbolic.supernodes),
            "permutation": symbolic.permutation,
            "parent": symbolic.parent,
            "supernodes": symbolic.supernodes,
        },
        "quantization": {
            "format": "S_format_local_contribution",
            "software_role": (
                "quantize each node's local original-matrix contribution A_local "
                "and preload mantissa/exponent sources into DDR"
            ),
            "hardware_role": (
                "assemble local sources with child updates, choose node-scale, "
                "execute integer LU/TRSM/GEMM, and generate child updates"
            ),
            "mantissa_dtype": "int32",
            "exponent_dtype": "int16",
            "q_limit": quant_limit(config.quant.effective_bits),
            "clip_count": clip_total,
            "sat_count": sat_total,
        },
        "solve": {
            "rhs_source": rhs_source_kind,
            "rhs_shape": [actual_matrix_dim],
            "rhs_format": "S_format",
            "rhs_exponent": int(rhs_source.exponent),
            "rhs_stats": asdict(rhs_source.stats),
            "solution_storage": "int64_mantissa_plus_per_node_int16_exponent",
        },
        "nodes": {
            str(node_id): {
                "range": asdict(symbolic.node_ranges[node_id]),
                "front_indices": symbolic.front_indices[node_id],
                "local_source": {
                    "format": "S_format",
                    "owner": "software",
                    "meaning": "quantized A_local source prepared for hardware DDR input",
                    "shape": list(source_shapes[node_id]),
                    "exponent": source_exponents[node_id],
                    "stats": asdict(quant_stats[node_id]),
                },
                "front_q": asdict(mem_plans[node_id].front_q),
                "front_e": asdict(mem_plans[node_id].front_e),
                "update_q": asdict(mem_plans[node_id].update_q),
                "update_e": asdict(mem_plans[node_id].update_e),
                "l_factor": asdict(mem_plans[node_id].l_factor),
                "u_factor": asdict(mem_plans[node_id].u_factor),
                "p_vector": asdict(mem_plans[node_id].p_vector),
                "node_meta": asdict(mem_plans[node_id].node_meta),
                "solve_workspace": asdict(mem_plans[node_id].solve_workspace),
                "hardware_owned_regions": [
                    "update_q",
                    "update_e",
                    "l_factor",
                    "u_factor",
                    "p_vector",
                    "node_meta",
                    "solve_workspace",
                ],
                "map_table": asdict(mem_plans[node_id].map_table),
                "front_q_file_offset": q_offsets[node_id],
                "front_e_file_offset": e_offsets[node_id],
                "map_table_file_offset": map_offsets[node_id],
                "reference_front_file_offset": reference_offsets[node_id],
            }
            for node_id in range(len(symbolic.node_ranges))
        },
        "task_order": task_order,
    }


def _prepare_rhs(
    matrix,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, str]:
    dimension = int(matrix.shape[0])
    if config.solve.rhs_path:
        rhs = load_vector(config.solve.rhs_path, dimension)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MatrixRankWarning)
            solution = np.asarray(spla.spsolve(matrix.tocsc(), rhs), dtype=np.float64)
        if not np.all(np.isfinite(solution)):
            raise ValueError("reference solve produced non-finite values for supplied RHS")
        return rhs, solution, "file"

    rng = np.random.default_rng(config.solve.seed)
    solution = rng.uniform(-1.0, 1.0, dimension).astype(np.float64)
    rhs = np.asarray(matrix @ solution, dtype=np.float64).reshape(-1)
    return rhs, solution, "generated_from_x_true"


def _write_memory_image(
    *,
    memory_image_path: Path,
    total_bytes: int,
    global_plan: GlobalMemoryPlan,
    mem_plans,
    ordered_tasks: List[NodeTask],
    permutation: List[int],
    front_q_data: bytes,
    front_e_data: bytes,
    map_data: bytes,
    q_offsets: List[int],
    e_offsets: List[int],
    map_offsets: List[int],
    q_sizes: List[int],
    e_sizes: List[int],
    map_sizes: List[int],
    rhs_q_data: bytes,
    rhs_e_data: bytes,
) -> None:
    image = bytearray(total_bytes)

    def copy_region(offset: int, size: int, data: bytes, label: str) -> None:
        if len(data) != size:
            raise ValueError(
                f"{label} payload size {len(data)} does not match region size {size}"
            )
        if offset < 0 or offset + size > len(image):
            raise ValueError(f"{label} region is outside memory image")
        image[offset : offset + size] = data

    task_bytes = b"".join(task.to_bytes() for task in ordered_tasks)
    copy_region(
        global_plan.task_queue.offset,
        global_plan.task_queue.size,
        task_bytes,
        "task_queue",
    )
    permutation_bytes = (
        struct.pack("<" + "I" * len(permutation), *permutation)
        if permutation
        else b""
    )
    copy_region(
        global_plan.permutation.offset,
        global_plan.permutation.size,
        permutation_bytes,
        "permutation",
    )
    copy_region(
        global_plan.rhs_q.offset,
        global_plan.rhs_q.size,
        rhs_q_data,
        "rhs_q",
    )
    copy_region(
        global_plan.rhs_e.offset,
        global_plan.rhs_e.size,
        rhs_e_data,
        "rhs_e",
    )

    for node_id, plan in mem_plans.items():
        copy_region(
            plan.front_q.offset,
            plan.front_q.size,
            front_q_data[q_offsets[node_id] : q_offsets[node_id] + q_sizes[node_id]],
            f"node {node_id} front_q",
        )
        copy_region(
            plan.front_e.offset,
            plan.front_e.size,
            front_e_data[e_offsets[node_id] : e_offsets[node_id] + e_sizes[node_id]],
            f"node {node_id} front_e",
        )
        copy_region(
            plan.map_table.offset,
            plan.map_table.size,
            map_data[
                map_offsets[node_id] : map_offsets[node_id] + map_sizes[node_id]
            ],
            f"node {node_id} map_table",
        )

    memory_image_path.write_bytes(image)


def _sizes_from_offsets(offsets: List[int], total_size: int) -> List[int]:
    sizes: List[int] = []
    for idx, offset in enumerate(offsets):
        next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else total_size
        sizes.append(next_offset - offset)
    return sizes


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
