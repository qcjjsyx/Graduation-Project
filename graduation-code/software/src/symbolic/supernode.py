from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp


def build_supernodes(
    parent: np.ndarray,
    matrix: sp.spmatrix | None = None,
    max_size: int = 256,
) -> List[List[int]]:
    """Build relaxed supernodes from consecutive columns.

    Consecutive columns are merged when they form an elimination-tree chain and
    their lower-triangular sparsity patterns match after removing the next pivot
    row. The implementation is dependency-free and intentionally conservative.
    """
    n = len(parent)
    if n == 0:
        return []
    if matrix is None:
        return [[i] for i in range(n)]

    lower_patterns = _lower_column_patterns(matrix)
    supernodes: List[List[int]] = []
    col = 0
    while col < n:
        current = [col]
        while (
            col + 1 < n
            and len(current) < max_size
            and int(parent[col]) == col + 1
            and _can_merge(lower_patterns[col], lower_patterns[col + 1], col + 1)
        ):
            col += 1
            current.append(col)
        supernodes.append(current)
        col += 1
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


def build_front_indices(matrix: sp.spmatrix, supernodes: List[List[int]]) -> List[List[int]]:
    patterns = _closed_column_patterns(matrix)
    fronts: List[List[int]] = []
    for columns in supernodes:
        pivot_cols = list(columns)
        start = pivot_cols[0]
        front = set(pivot_cols)
        for col in pivot_cols:
            front.update(row for row in patterns[col] if row >= start)
        ordered = pivot_cols + sorted(idx for idx in front if idx not in pivot_cols)
        fronts.append(ordered)
    return fronts


def build_column_to_supernode(supernodes: List[List[int]], n: int) -> np.ndarray:
    column_to_node = np.full(n, -1, dtype=np.int32)
    for node_id, columns in enumerate(supernodes):
        for col in columns:
            column_to_node[col] = node_id
    if np.any(column_to_node < 0):
        raise ValueError("supernodes do not cover every column")
    return column_to_node


def _can_merge(left_pattern: set[int], right_pattern: set[int], next_col: int) -> bool:
    return (left_pattern - {next_col}) == right_pattern


def _lower_column_patterns(matrix: sp.spmatrix) -> List[set[int]]:
    csc = matrix.tocsc()
    patterns: List[set[int]] = []
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        rows = csc.indices[start:end]
        patterns.append(set(int(row) for row in rows if row > col))
    return patterns


def _closed_column_patterns(matrix: sp.spmatrix) -> List[set[int]]:
    csc = matrix.tocsc()
    patterns: List[set[int]] = []
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        rows = set(int(row) for row in csc.indices[start:end])
        rows.add(col)
        patterns.append(rows)
    return patterns
