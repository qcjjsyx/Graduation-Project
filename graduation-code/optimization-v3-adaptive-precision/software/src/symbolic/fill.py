from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class FilledPattern:
    """Symbolic factor pattern for a structurally symmetric matrix.

    ``columns[k]`` contains ``k`` followed by the filled row indices below the
    diagonal in column ``k``.  The implementation uses an explicit elimination
    graph; this is intentionally simple and deterministic for the matrices used
    by the architecture model.
    """

    parent: np.ndarray
    columns: List[List[int]]
    fill_edge_count: int


def require_structurally_symmetric(matrix: sp.spmatrix) -> None:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    pattern = (matrix != 0).astype(np.int8).tocsr()
    difference = pattern != pattern.T
    if difference.nnz:
        raise ValueError(
            "ABI v2 currently requires a structurally symmetric sparsity pattern; "
            f"found {difference.nnz} asymmetric pattern entries"
        )


def symbolic_fill_pattern(matrix: sp.spmatrix) -> FilledPattern:
    """Build the filled lower-column pattern and elimination forest.

    Numeric values are deliberately ignored.  At elimination step ``k`` the
    higher-numbered neighbours form the filled column of L and are connected
    into a clique.  The first higher neighbour is the etree parent.
    """

    require_structurally_symmetric(matrix)
    pattern = ((matrix != 0) + (matrix.T != 0)).astype(bool).tocsr()
    n = int(pattern.shape[0])
    adjacency: List[set[int]] = []
    for row in range(n):
        start, end = pattern.indptr[row], pattern.indptr[row + 1]
        adjacency.append(
            {int(col) for col in pattern.indices[start:end] if int(col) != row}
        )

    parent = np.full(n, -1, dtype=np.int32)
    columns: List[List[int]] = []
    fill_edge_count = 0

    for pivot in range(n):
        higher = sorted(neighbour for neighbour in adjacency[pivot] if neighbour > pivot)
        columns.append([pivot, *higher])
        if higher:
            parent[pivot] = higher[0]

        for left_index, left in enumerate(higher):
            left_adj = adjacency[left]
            for right in higher[left_index + 1 :]:
                if right not in left_adj:
                    left_adj.add(right)
                    adjacency[right].add(left)
                    fill_edge_count += 1

        for neighbour in tuple(adjacency[pivot]):
            adjacency[neighbour].discard(pivot)
        adjacency[pivot].clear()

    return FilledPattern(
        parent=parent,
        columns=columns,
        fill_edge_count=fill_edge_count,
    )

