from __future__ import annotations

from typing import List

import numpy as np


def build_supernodes(parent: np.ndarray) -> List[List[int]]:
    """Build supernodes from an elimination tree.

    Simplified: each column is its own supernode.
    TODO: merge consecutive nodes with identical structure.
    """
    return [[i] for i in range(len(parent))]