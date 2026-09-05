from __future__ import annotations

from typing import Dict, List, Tuple

from src.dataStruct import GlobalMemoryPlan, MemoryRegion, NodeMemoryPlan


def _align(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


def plan_memory(
    num_nodes: int,
    matrix_dim: int,
    front_q_sizes: List[int],
    front_e_sizes: List[int],
    map_table_sizes: List[int],
    update_q_sizes: List[int] | None = None,
    update_e_sizes: List[int] | None = None,
    l_factor_sizes: List[int] | None = None,
    u_factor_sizes: List[int] | None = None,
    p_vector_sizes: List[int] | None = None,
    node_meta_sizes: List[int] | None = None,
    solve_workspace_sizes: List[int] | None = None,
    task_record_size: int = 128,
    align: int = 64,
) -> Tuple[GlobalMemoryPlan, Dict[int, NodeMemoryPlan], int]:
    """Plan DDR offsets for each node (offset-based simulation)."""
    if num_nodes < 0 or matrix_dim < 0:
        raise ValueError("num_nodes and matrix_dim must be non-negative")
    update_q_sizes = _default_sizes(update_q_sizes, num_nodes)
    update_e_sizes = _default_sizes(update_e_sizes, num_nodes)
    l_factor_sizes = _default_sizes(l_factor_sizes, num_nodes)
    u_factor_sizes = _default_sizes(u_factor_sizes, num_nodes)
    p_vector_sizes = _default_sizes(p_vector_sizes, num_nodes)
    node_meta_sizes = _default_sizes(node_meta_sizes, num_nodes)
    solve_workspace_sizes = _default_sizes(solve_workspace_sizes, num_nodes)

    size_lists = (
        front_q_sizes,
        front_e_sizes,
        map_table_sizes,
        update_q_sizes,
        update_e_sizes,
        l_factor_sizes,
        u_factor_sizes,
        p_vector_sizes,
        node_meta_sizes,
        solve_workspace_sizes,
    )
    if any(len(sizes) != num_nodes for sizes in size_lists):
        raise ValueError("size lists must match num_nodes")

    offset = 0
    task_queue, offset = _place_region(offset, num_nodes * task_record_size, align)
    permutation, offset = _place_region(offset, matrix_dim * 4, align)
    rhs_q, offset = _place_region(offset, matrix_dim * 4, align)
    rhs_e, offset = _place_region(offset, 2, align)
    solution_q, offset = _place_region(offset, matrix_dim * 8, align)
    solution_e, offset = _place_region(offset, num_nodes * 2, align)
    global_plan = GlobalMemoryPlan(
        task_queue=task_queue,
        permutation=permutation,
        rhs_q=rhs_q,
        rhs_e=rhs_e,
        solution_q=solution_q,
        solution_e=solution_e,
    )

    plans: Dict[int, NodeMemoryPlan] = {}
    for node_id in range(num_nodes):
        front_q, offset = _place_region(offset, front_q_sizes[node_id], align)
        front_e, offset = _place_region(offset, front_e_sizes[node_id], align)
        update_q, offset = _place_region(offset, update_q_sizes[node_id], align)
        update_e, offset = _place_region(offset, update_e_sizes[node_id], align)
        l_factor, offset = _place_region(offset, l_factor_sizes[node_id], align)
        u_factor, offset = _place_region(offset, u_factor_sizes[node_id], align)
        map_table, offset = _place_region(offset, map_table_sizes[node_id], align)
        p_vector, offset = _place_region(offset, p_vector_sizes[node_id], align)
        node_meta, offset = _place_region(offset, node_meta_sizes[node_id], align)
        solve_workspace, offset = _place_region(
            offset, solve_workspace_sizes[node_id], align
        )

        plans[node_id] = NodeMemoryPlan(
            front_q=front_q,
            front_e=front_e,
            update_q=update_q,
            update_e=update_e,
            l_factor=l_factor,
            u_factor=u_factor,
            map_table=map_table,
            p_vector=p_vector,
            node_meta=node_meta,
            solve_workspace=solve_workspace,
        )

    return global_plan, plans, offset


def _default_sizes(sizes: List[int] | None, num_nodes: int) -> List[int]:
    return [0] * num_nodes if sizes is None else sizes


def _place_region(offset: int, size: int, align: int) -> Tuple[MemoryRegion, int]:
    if size < 0:
        raise ValueError(f"memory region size must be non-negative, got {size}")
    aligned = _align(offset, align)
    return MemoryRegion(aligned, size), aligned + size
