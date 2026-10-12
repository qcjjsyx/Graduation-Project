from __future__ import annotations

from typing import List

import numpy as np


def build_supernodes_from_filled(
    parent: np.ndarray,
    filled_columns: List[List[int]],
    max_size: int = 256,
) -> List[List[int]]:
    """Build exact consecutive supernodes from a filled factor pattern.

    Consecutive columns share a supernode when the first column's off-diagonal
    structure is the next pivot followed by the next column's off-diagonal
    structure.  This is the standard structural condition needed by the dense
    frontal blocks; unlike ``build_supernodes`` it is based on fill, not only
    on the original graph.
    """

    n = len(parent)
    if len(filled_columns) != n:
        raise ValueError("filled column count must match parent length")
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")

    supernodes: List[List[int]] = []
    column = 0
    while column < n:
        current = [column]
        while (
            column + 1 < n
            and len(current) < max_size
            and int(parent[column]) == column + 1
            and filled_columns[column][1:] == filled_columns[column + 1]
        ):
            column += 1
            current.append(column)
        supernodes.append(current)
        column += 1
    return supernodes


def build_supernode_parent(column_parent: np.ndarray, supernodes: List[List[int]]) -> List[int]:
    column_to_node = build_column_to_supernode(supernodes, len(column_parent))
    parent: List[int] = []
    for node_id, columns in enumerate(supernodes):
        parent_node = -1
        for col in reversed(columns):
            p = int(column_parent[col])
            if p >= 0:
                candidate = int(column_to_node[p])
                if candidate != node_id:
                    parent_node = candidate
                    break
        parent.append(parent_node)
    return parent


def build_front_indices_from_filled(
    supernodes: List[List[int]],
    filled_columns: List[List[int]],
) -> List[List[int]]:
    fronts: List[List[int]] = []
    for columns in supernodes:
        if not columns:
            raise ValueError("supernode cannot be empty")
        pivot_set = set(columns)
        update: set[int] = set()
        for column in columns:
            update.update(filled_columns[column][1:])
        update.difference_update(pivot_set)
        fronts.append([*columns, *sorted(update)])
    return fronts


def build_column_to_supernode(supernodes: List[List[int]], n: int) -> np.ndarray:
    column_to_node = np.full(n, -1, dtype=np.int32)
    for node_id, columns in enumerate(supernodes):
        for col in columns:
            column_to_node[col] = node_id
    if np.any(column_to_node < 0):
        raise ValueError("supernodes do not cover every column")
    return column_to_node
