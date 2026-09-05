from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def compute_ordering(a: sp.spmatrix, method: str = "amd") -> np.ndarray:
    method = method.lower()
    if method == "amd":
        return approximate_minimum_degree(a)
    if method == "rcm":
        return reverse_cuthill_mckee_ordering(a)
    if method == "identity":
        return identity_ordering(a)
    raise ValueError(f"unknown ordering method: {method}")


def identity_ordering(a: sp.spmatrix) -> np.ndarray:
    _validate_square_sparse(a)
    return np.arange(a.shape[0], dtype=np.int32)


def reverse_cuthill_mckee_ordering(a: sp.spmatrix) -> np.ndarray:
    _validate_square_sparse(a)
    perm = reverse_cuthill_mckee(a, symmetric_mode=True)
    return np.asarray(perm, dtype=np.int32)


def approximate_minimum_degree(a: sp.spmatrix) -> np.ndarray:
    """Minimum-degree ordering for the symmetric sparsity graph.

    This is intentionally dependency-free. It is not a production-grade SuiteSparse
    AMD implementation, but it follows the same core idea: repeatedly eliminate the
    currently lowest-degree node and update the graph with fill edges.
    """
    _validate_square_sparse(a)
    adjacency = _symmetric_adjacency(a)
    remaining = set(range(a.shape[0]))
    order: list[int] = []

    while remaining:
        node = min(remaining, key=lambda idx: (len(adjacency[idx] & remaining), idx))
        neighbors = sorted(adjacency[node] & remaining)
        neighbors = [idx for idx in neighbors if idx != node]

        for i, left in enumerate(neighbors):
            fill_targets = neighbors[i + 1 :]
            adjacency[left].update(fill_targets)
            for right in fill_targets:
                adjacency[right].add(left)

        remaining.remove(node)
        for neighbor in neighbors:
            adjacency[neighbor].discard(node)
        adjacency[node].clear()
        order.append(node)

    return np.asarray(order, dtype=np.int32)


def apply_permutation(a: sp.spmatrix, perm: np.ndarray) -> sp.spmatrix:
    perm = np.asarray(perm, dtype=np.int32)
    if perm.ndim != 1 or len(perm) != a.shape[0]:
        raise ValueError("permutation length must match matrix dimension")
    return a[perm][:, perm] # type: ignore


def _validate_square_sparse(a: sp.spmatrix) -> None:
    if not sp.isspmatrix(a):
        raise TypeError("a must be a scipy sparse matrix")
    if a.shape[0] != a.shape[1]:
        raise ValueError("a must be square")


def _symmetric_adjacency(a: sp.spmatrix) -> list[set[int]]:
    pattern = (a != 0).astype(np.int8) # type: ignore
    pattern = (pattern + pattern.T).astype(bool).tocsr()
    pattern.setdiag(False)
    pattern.eliminate_zeros()

    adjacency: list[set[int]] = []
    for row in range(pattern.shape[0]):
        start, end = pattern.indptr[row], pattern.indptr[row + 1]
        adjacency.append(set(int(col) for col in pattern.indices[start:end]))
    return adjacency
