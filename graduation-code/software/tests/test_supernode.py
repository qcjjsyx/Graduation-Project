import numpy as np
import scipy.sparse as sp

from src.symbolic.etree import elimination_tree
from src.symbolic.supernode import (
    build_front_indices,
    build_supernode_parent,
    build_supernodes,
)


def test_dense_block_merges_into_one_supernode():
    a = sp.csr_matrix(np.ones((4, 4), dtype=np.float64))
    parent = elimination_tree(a)
    supernodes = build_supernodes(parent, a)
    assert supernodes == [[0, 1, 2, 3]]
    assert build_supernode_parent(parent, supernodes) == [-1]
    assert build_front_indices(a, supernodes) == [[0, 1, 2, 3]]


def test_block_diagonal_supernodes_stay_separate_by_component():
    block = np.ones((3, 3), dtype=np.float64)
    a = sp.block_diag((block, block), format="csr")
    parent = elimination_tree(a)
    supernodes = build_supernodes(parent, a)
    assert supernodes == [[0, 1, 2], [3, 4, 5]]
    assert build_supernode_parent(parent, supernodes) == [-1, -1]
