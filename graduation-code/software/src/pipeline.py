from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import scipy.sparse.linalg as spla

from src.config import PipelineConfig
from src.dataStruct import (
    ABI_VERSION,
    NODE_TASK_BYTE_SIZE,
    ROOT_PARENT_ID,
    NodeRange,
    NodeTask,
    SymbolicResult,
)
from src.io import write_front_data, write_manifest, write_map_table, write_tasks
from src.matrix_io import load_matrix
from src.memory.planner import plan_memory
from src.quant.bfp_quant import (
    QuantizationStats,
    flatten_quantized_source,
    quant_limit,
    quantize_local_contribution,
)
from src.scheduler.map_gen import generate_map_tables
from src.scheduler.task_queue import sibling_friendly_order
from src.symbolic.etree import children_from_parent, elimination_tree
from src.symbolic.ordering import apply_permutation, compute_ordering
from src.symbolic.supernode import build_front_indices, build_supernode_parent, build_supernodes
from src.verify.manifest import validate_manifest
from src.verify.metrics import residual_norm


@dataclass(frozen=True)
class PipelineOutputs:
    out_dir: Path
    tasks_path: Path
    map_table_path: Path
    front_q_path: Path
    front_e_path: Path
    manifest_path: Path
    residual_norm: float
    node_count: int
    task_count: int


def run_pipeline(config: PipelineConfig) -> PipelineOutputs:
    matrix = load_matrix(
        path=config.matrix.path,
        n=config.matrix.n,
        density=config.matrix.density,
        seed=config.matrix.seed,
    )

    symbolic = run_symbolic_analysis(matrix, config)
    permuted = apply_permutation(matrix, np.asarray(symbolic.permutation, dtype=np.int32))
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
    manifest_path = out_dir / "manifest.json"

    front_q_path.write_bytes(b"")
    front_e_path.write_bytes(b"")

    map_offsets = write_map_table(str(map_path), map_tables)
    map_sizes = _sizes_from_offsets(map_offsets, map_path.stat().st_size)

    q_sizes: List[int] = []
    e_sizes: List[int] = []
    q_offsets: List[int] = []
    e_offsets: List[int] = []
    source_shapes: List[tuple[int, int]] = []
    source_exponents: List[int] = []
    quant_stats: List[QuantizationStats] = []
    clip_total = 0
    sat_total = 0
    q_cursor = 0
    e_cursor = 0

    for front_indices in symbolic.front_indices:
        local_contribution = extract_local_contribution(permuted, front_indices)
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

    update_q_sizes, update_e_sizes, l_factor_sizes, u_factor_sizes, task_desc_sizes = (
        estimate_work_region_sizes(symbolic.node_ranges, symbolic.front_indices)
    )

    mem_plans, total_bytes = plan_memory(
        len(symbolic.node_ranges),
        q_sizes,
        e_sizes,
        map_sizes,
        update_q_sizes=update_q_sizes,
        update_e_sizes=update_e_sizes,
        l_factor_sizes=l_factor_sizes,
        u_factor_sizes=u_factor_sizes,
        task_desc_sizes=task_desc_sizes,
        align=config.memory.alignment,
    )

    tasks = build_node_tasks(
        symbolic.parent,
        symbolic.node_ranges,
        symbolic.front_indices,
        mem_plans,
    )
    task_order = sibling_friendly_order(symbolic.parent)
    ordered_tasks = [tasks[node_id] for node_id in task_order]
    write_tasks(str(tasks_path), ordered_tasks)

    residual = _reference_residual(permuted)
    manifest = build_manifest(
        config=config,
        symbolic=symbolic,
        mem_plans=mem_plans,
        total_bytes=total_bytes,
        q_offsets=q_offsets,
        e_offsets=e_offsets,
        map_offsets=map_offsets,
        source_shapes=source_shapes,
        source_exponents=source_exponents,
        quant_stats=quant_stats,
        clip_total=clip_total,
        sat_total=sat_total,
        task_order=task_order,
        residual=residual,
        output_sizes={
            "tasks.bin": tasks_path.stat().st_size,
            "map_table.bin": map_path.stat().st_size,
            "front_q.bin": front_q_path.stat().st_size,
            "front_e.bin": front_e_path.stat().st_size,
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
        manifest_path=manifest_path,
        residual_norm=residual,
        node_count=len(symbolic.node_ranges),
        task_count=len(ordered_tasks),
    )


def run_symbolic_analysis(matrix, config: PipelineConfig) -> SymbolicResult:
    permutation = compute_ordering(matrix, config.ordering.method)
    permuted = apply_permutation(matrix, permutation)
    parent_arr = elimination_tree(permuted)
    supernodes = build_supernodes(
        parent_arr,
        permuted,
        max_size=config.ordering.max_supernode_size,
    )
    node_ranges = build_node_ranges(supernodes)
    node_parent = build_supernode_parent(parent_arr, supernodes)
    front_indices = build_front_indices(permuted, supernodes)
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


def extract_local_contribution(matrix, front_indices: List[int]) -> np.ndarray:
    """Return the node-local A contribution over this front's variables."""
    return matrix[front_indices][:, front_indices].toarray().astype(np.float32)


def estimate_work_region_sizes(
    node_ranges: List[NodeRange],
    front_indices: List[List[int]],
) -> tuple[List[int], List[int], List[int], List[int], List[int]]:
    update_q_sizes: List[int] = []
    update_e_sizes: List[int] = []
    l_factor_sizes: List[int] = []
    u_factor_sizes: List[int] = []
    task_desc_sizes: List[int] = []

    for node_range, front in zip(node_ranges, front_indices):
        pivot_dim = node_range.size
        total_dim = len(front)
        update_dim = max(total_dim - pivot_dim, 0)

        update_q_sizes.append(update_dim * update_dim * 4)
        update_e_sizes.append(2 if update_dim else 0)
        l_factor_sizes.append(total_dim * pivot_dim * 4)
        u_factor_sizes.append(pivot_dim * total_dim * 4)
        task_desc_sizes.append(NODE_TASK_BYTE_SIZE)

    return update_q_sizes, update_e_sizes, l_factor_sizes, u_factor_sizes, task_desc_sizes


def build_node_tasks(
    parent: List[int],
    node_ranges: List[NodeRange],
    front_indices: List[List[int]],
    mem_plans,
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
        nums_sub_matrix = _ceil_div(pivot_dim, 16)
        last_sub_matrix_size = pivot_dim % 16 or min(pivot_dim, 16)
        tasks.append(
            NodeTask(
                node_id=node_id,
                flags=flags,
                parent_id=parent_id if parent_id >= 0 else ROOT_PARENT_ID,
                children_count=len(children[node_id]),
                total_dim=total_dim,
                pivot_dim=pivot_dim,
                nums_sub_matrix=nums_sub_matrix,
                last_sub_matrix_size=last_sub_matrix_size,
                data_addr=plan.front_q.offset,
                parent_address=mem_plans[parent_id].front_q.offset if parent_id >= 0 else 0,
                map_table_addr=plan.map_table.offset,
                l_factor_addr=plan.l_factor.offset,
                u_factor_addr=plan.u_factor.offset,
                p_vector_addr=0,
                reversed=0,
            )
        )

    return tasks


def build_manifest(
    *,
    config: PipelineConfig,
    symbolic: SymbolicResult,
    mem_plans,
    total_bytes: int,
    q_offsets: List[int],
    e_offsets: List[int],
    map_offsets: List[int],
    source_shapes: List[tuple[int, int]],
    source_exponents: List[int],
    quant_stats: List[QuantizationStats],
    clip_total: int,
    sat_total: int,
    task_order: List[int],
    residual: float,
    output_sizes: dict[str, int],
) -> dict:
    return {
        "abi": {
            "version": ABI_VERSION,
            "node_task_byte_size": NODE_TASK_BYTE_SIZE,
        },
        "config": {
            "ordering": asdict(config.ordering),
            "quant": asdict(config.quant),
            "memory": asdict(config.memory),
        },
        "matrix": {
            "path": config.matrix.path,
            "n": config.matrix.n,
            "density": config.matrix.density,
            "seed": config.matrix.seed,
        },
        "total_bytes": total_bytes,
        "output_sizes": output_sizes,
        "verification": {
            "reference_residual_norm": residual,
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
                "hardware_owned_regions": [
                    "update_q",
                    "update_e",
                    "l_factor",
                    "u_factor",
                ],
                "map_table": asdict(mem_plans[node_id].map_table),
                "task_desc": asdict(mem_plans[node_id].task_desc),
                "front_q_file_offset": q_offsets[node_id],
                "front_e_file_offset": e_offsets[node_id],
                "map_table_file_offset": map_offsets[node_id],
            }
            for node_id in range(len(symbolic.node_ranges))
        },
        "task_order": task_order,
    }


def _sizes_from_offsets(offsets: List[int], total_size: int) -> List[int]:
    sizes: List[int] = []
    for idx, offset in enumerate(offsets):
        next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else total_size
        sizes.append(next_offset - offset)
    return sizes


def _reference_residual(matrix) -> float:
    b = np.ones(matrix.shape[0], dtype=np.float32)
    x = spla.spsolve(matrix.tocsr(), b)
    return residual_norm(matrix, x, b)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
