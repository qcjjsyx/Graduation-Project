from __future__ import annotations

from typing import Dict, List, Tuple

from src.dataStruct import MemoryRegion, NodeMemoryPlan


def _align(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


def plan_memory(
    num_nodes: int,
    front_q_sizes: List[int],
    front_e_sizes: List[int],
    map_table_sizes: List[int],
    update_q_sizes: List[int] | None = None,
    update_e_sizes: List[int] | None = None,
    l_factor_sizes: List[int] | None = None,
    u_factor_sizes: List[int] | None = None,
    task_desc_sizes: List[int] | None = None,
    align: int = 64,
) -> Tuple[Dict[int, NodeMemoryPlan], int]:
    """Plan DDR offsets for each node (offset-based simulation)."""
    update_q_sizes = _default_sizes(update_q_sizes, num_nodes)
    update_e_sizes = _default_sizes(update_e_sizes, num_nodes)
    l_factor_sizes = _default_sizes(l_factor_sizes, num_nodes)
    u_factor_sizes = _default_sizes(u_factor_sizes, num_nodes)
    task_desc_sizes = _default_sizes(task_desc_sizes, num_nodes)

    size_lists = (
        front_q_sizes,
        front_e_sizes,
        map_table_sizes,
        update_q_sizes,
        update_e_sizes,
        l_factor_sizes,
        u_factor_sizes,
        task_desc_sizes,
    )
    if any(len(sizes) != num_nodes for sizes in size_lists):
        raise ValueError("size lists must match num_nodes")

    plans: Dict[int, NodeMemoryPlan] = {}
    offset = 0
    for node_id in range(num_nodes):
        front_q, offset = _place_region(offset, front_q_sizes[node_id], align)
        front_e, offset = _place_region(offset, front_e_sizes[node_id], align)
        update_q, offset = _place_region(offset, update_q_sizes[node_id], align)
        update_e, offset = _place_region(offset, update_e_sizes[node_id], align)
        l_factor, offset = _place_region(offset, l_factor_sizes[node_id], align)
        u_factor, offset = _place_region(offset, u_factor_sizes[node_id], align)
        map_table, offset = _place_region(offset, map_table_sizes[node_id], align)
        task_desc, offset = _place_region(offset, task_desc_sizes[node_id], align)

        plans[node_id] = NodeMemoryPlan(
            front_q=front_q,
            front_e=front_e,
            update_q=update_q,
            update_e=update_e,
            l_factor=l_factor,
            u_factor=u_factor,
            map_table=map_table,
            task_desc=task_desc,
        )

    return plans, offset


def _default_sizes(sizes: List[int] | None, num_nodes: int) -> List[int]:
    return [0] * num_nodes if sizes is None else sizes


def _place_region(offset: int, size: int, align: int) -> Tuple[MemoryRegion, int]:
    if size < 0:
        raise ValueError(f"memory region size must be non-negative, got {size}")
    aligned = _align(offset, align)
    return MemoryRegion(aligned, size), aligned + size
