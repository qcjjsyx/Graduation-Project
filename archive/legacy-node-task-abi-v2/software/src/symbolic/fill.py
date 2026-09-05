from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import scipy.sparse as sp

from src.symbolic.pattern import symmetric_sparsity_pattern


@dataclass(frozen=True)
class FilledPattern:
    """Symbolic factor pattern for a symmetric sparsity envelope.

    ``columns[k]`` contains ``k`` followed by the filled row indices below the
    diagonal in column ``k``.  The implementation uses an explicit elimination
    graph; this is intentionally simple and deterministic for the matrices used
    by the architecture model.
    """

    parent: np.ndarray
    columns: List[List[int]]
    fill_edge_count: int


def symbolic_fill_pattern(matrix: sp.spmatrix) -> FilledPattern:
    """Build the filled pattern and forest from a symmetric envelope.

    The input may have an asymmetric nonzero structure.  Numeric values are
    discarded and ``pattern(A) union pattern(A.T)`` is formed first.  At
    elimination step ``k`` the higher-numbered neighbours form the filled
    column and are connected into a clique.  The first higher neighbour is the
    etree parent.
    """

    pattern = symmetric_sparsity_pattern(matrix)
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
