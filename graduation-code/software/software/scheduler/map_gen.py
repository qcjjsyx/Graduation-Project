from __future__ import annotations

from typing import Dict, List, Tuple

from software.io import MapTableEntry


def generate_map_tables(
    node_ranges: List[Tuple[int, int]],
    parent: List[int],
) -> List[List[MapTableEntry]]:
    """Generate simple identity map tables between child and parent.

    child update rows/cols are mapped to parent frontal by matching indices.
    """
    map_tables: List[List[MapTableEntry]] = [[] for _ in range(len(node_ranges))]
    for child_id, p in enumerate(parent):
        if p < 0:
            continue
        c_start, c_end = node_ranges[child_id]
        p_start, p_end = node_ranges[p]
        # Identity mapping over overlapping indices
        overlap_start = max(c_start, p_start)
        overlap_end = min(c_end, p_end)
        row_map: List[int] = []
        col_map: List[int] = []
        for idx in range(overlap_start, overlap_end):
            row_map.append(idx - c_start)
            col_map.append(idx - p_start)
        map_tables[p].append(MapTableEntry(child_id=child_id, row_map=row_map, col_map=col_map))
    return map_tables