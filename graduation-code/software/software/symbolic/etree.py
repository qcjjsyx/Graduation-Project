from __future__ import annotations

from typing import List

import numpy as np
import scipy.sparse as sp


def elimination_tree(a: sp.spmatrix) -> np.ndarray:
    """Compute elimination tree for a symmetric sparse matrix.

    Simplified implementation; suitable for small demos.
    """
    if not sp.isspmatrix_csc(a):
        a = a.tocsc()
    n = a.shape[0]
    parent = np.full(n, -1, dtype=np.int32)
    ancestor = np.full(n, -1, dtype=np.int32)

    indptr = a.indptr
    indices = a.indices

    for k in range(n):
        ancestor[k] = -1
        for idx in range(indptr[k], indptr[k + 1]):
            i = indices[idx]
            if i >= k:
                continue
            while i != -1 and i < k:
                next_i = ancestor[i]
                ancestor[i] = k
                if next_i == -1:
                    parent[i] = k
                    i = -1
                else:
                    i = next_i
    return parent


def children_from_parent(parent: np.ndarray) -> List[List[int]]:
    n = len(parent)
    children: List[List[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    return children