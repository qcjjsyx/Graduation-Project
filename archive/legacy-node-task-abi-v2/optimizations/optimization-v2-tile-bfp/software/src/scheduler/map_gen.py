from __future__ import annotations

from typing import List, Tuple

from src.dataStruct import MapTableEntry


def generate_map_tables(
    node_ranges: List[Tuple[int, int]],
    parent: List[int],
    front_indices: List[List[int]],
) -> List[List[MapTableEntry]]:
    """Generate child-update to parent-front map tables."""
    if not (len(node_ranges) == len(parent) == len(front_indices)):
        raise ValueError("node_ranges, parent, and front_indices lengths must match")

    map_tables: List[List[MapTableEntry]] = [[] for _ in range(len(node_ranges))]
    parent_pos_cache = [
        {global_col: local_idx for local_idx, global_col in enumerate(front)}
        for front in front_indices
    ]

    for child_id, parent_id in enumerate(parent):
        if parent_id < 0:
            continue

        child_pivot_dim = node_ranges[child_id][1] - node_ranges[child_id][0]
        child_update_vars = front_indices[child_id][child_pivot_dim:]
        parent_pos = parent_pos_cache[parent_id]

        row_map: List[int] = []
        col_map: List[int] = []
        for update_idx, global_col in enumerate(child_update_vars):
            if global_col not in parent_pos:
                raise ValueError(
                    f"child {child_id} update variable {global_col} is absent "
                    f"from parent {parent_id} front; symbolic front is incomplete"
                )
            row_map.append(update_idx)
            col_map.append(parent_pos[global_col])

        map_tables[parent_id].append(
            MapTableEntry(child_id=child_id, row_map=row_map, col_map=col_map)
        )

    return map_tables
