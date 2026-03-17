from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


def reorder_rcm(a: sp.spmatrix) -> np.ndarray:
    """Reorder using reverse Cuthill-McKee (placeholder for AMD/METIS)."""
    if not sp.isspmatrix(a):
        raise TypeError("a must be a scipy sparse matrix")
    perm = reverse_cuthill_mckee(a, symmetric_mode=True)
    return np.asarray(perm, dtype=np.int32)


def apply_permutation(a: sp.spmatrix, perm: Sequence[int]) -> sp.spmatrix:
    perm = np.asarray(perm, dtype=np.int32)
    return a[perm][:, perm]