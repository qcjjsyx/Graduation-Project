from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class MemoryRegion:
    offset: int
    size: int


@dataclass
class NodeMemoryPlan:
    front_q: MemoryRegion
    front_e: MemoryRegion
    update_q: MemoryRegion
    update_e: MemoryRegion
    l_factor: MemoryRegion
    u_factor: MemoryRegion
    map_table: MemoryRegion
    task_desc: MemoryRegion


def _align(offset: int, align: int) -> int:
    return (offset + align - 1) // align * align


def plan_memory(
    num_nodes: int,
    front_q_sizes: List[int],
    front_e_sizes: List[int],
    map_table_sizes: List[int],
    align: int = 64,
) -> Tuple[Dict[int, NodeMemoryPlan], int]:
    """Plan DDR offsets for each node (offset-based simulation)."""
    if not (len(front_q_sizes) == len(front_e_sizes) == len(map_table_sizes) == num_nodes):
        raise ValueError("size lists must match num_nodes")

    plans: Dict[int, NodeMemoryPlan] = {}
    offset = 0
    for node_id in range(num_nodes):
        offset = _align(offset, align)
        front_q = MemoryRegion(offset, front_q_sizes[node_id])
        offset += front_q.size

        offset = _align(offset, align)
        front_e = MemoryRegion(offset, front_e_sizes[node_id])
        offset += front_e.size

        # Placeholder sizes for update/L/U (zero in minimal prototype)
        update_q = MemoryRegion(_align(offset, align), 0)
        update_e = MemoryRegion(_align(offset, align), 0)
        l_factor = MemoryRegion(_align(offset, align), 0)
        u_factor = MemoryRegion(_align(offset, align), 0)

        offset = _align(offset, align)
        map_table = MemoryRegion(offset, map_table_sizes[node_id])
        offset += map_table.size

        offset = _align(offset, align)
        task_desc = MemoryRegion(offset, 0)

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