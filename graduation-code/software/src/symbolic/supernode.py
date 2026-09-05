from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp


def build_supernodes(
    parent: np.ndarray,
    matrix: sp.spmatrix | None = None,
    max_size: int = 256,
) -> List[List[int]]:
    """Build supernodes from consecutive indistinguishable columns.

    The merge rule follows the closed-neighborhood criterion used for
    indistinguishable variables in AMD-style quotient graph descriptions:
    two columns can be merged when Adj(i) U {i} == Adj(j) U {j}. The pipeline
    keeps the additional engineering restriction that merged columns must be
    consecutive in the current ordering, so compiler node ranges remain compact.
    """
    n = len(parent)
    if n == 0:
        return []
    if max_size <= 0:
        raise ValueError(f"max_size must be positive, got {max_size}")
    if matrix is None:
        raise ValueError("matrix is required to build supernodes")
    if matrix.shape != (n, n):
        raise ValueError("matrix shape must match parent length")

    closed_adj = _closed_graph_adjacency(matrix)
    supernodes: List[List[int]] = []
    col = 0
    while col < n:
        signature = closed_adj[col]
        current = [col]
        while col + 1 < n and len(current) < max_size and closed_adj[col + 1] == signature:
            col += 1
            current.append(col)
        supernodes.append(current)
        col += 1
    return supernodes


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


def _closed_column_patterns(matrix: sp.spmatrix) -> List[set[int]]:
    if matrix is None:
        raise ValueError("matrix is required to build supernodes")
    csc = matrix.tocsc() # type: ignore
    patterns: List[set[int]] = []
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        rows = set(int(row) for row in csc.indices[start:end])
        rows.add(col)
        patterns.append(rows)
    return patterns


def _closed_graph_adjacency(matrix: sp.spmatrix) -> List[tuple[int, ...]]:
    pattern = (matrix != 0).astype(np.int8) # type: ignore
    graph = (pattern + pattern.T).astype(bool).tocsr()
    graph.setdiag(True)
    graph.eliminate_zeros()

    adjacency: List[tuple[int, ...]] = []
    for row in range(graph.shape[0]):
        start, end = graph.indptr[row], graph.indptr[row + 1]
        adjacency.append(tuple(int(col) for col in graph.indices[start:end]))
    return adjacency
