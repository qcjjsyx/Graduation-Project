import numpy as np
import pytest
import scipy.sparse as sp

from src.symbolic.fill import symbolic_fill_pattern
from src.symbolic.supernode import (
    build_front_indices_from_filled,
    build_supernode_parent,
    build_supernodes_from_filled,
)


def _symbolic(matrix, max_size=256):
    filled = symbolic_fill_pattern(matrix)
    supernodes = build_supernodes_from_filled(
        filled.parent, filled.columns, max_size=max_size
    )
    return filled, supernodes


def test_dense_block_merges_into_one_supernode():
    a = sp.csr_matrix(np.ones((4, 4), dtype=np.float64))
    filled, supernodes = _symbolic(a)
    assert supernodes == [[0, 1, 2, 3]]
    assert build_supernode_parent(filled.parent, supernodes) == [-1]
    assert build_front_indices_from_filled(supernodes, filled.columns) == [[0, 1, 2, 3]]


def test_block_diagonal_supernodes_stay_separate_by_component():
    block = np.ones((3, 3), dtype=np.float64)
    a = sp.csr_matrix(sp.block_diag((block, block)))
    filled, supernodes = _symbolic(a)
    assert supernodes == [[0, 1, 2], [3, 4, 5]]
    assert build_supernode_parent(filled.parent, supernodes) == [-1, -1]


def test_supernode_merge_respects_max_size_for_large_dense_pattern():
    a = sp.csr_matrix(np.ones((300, 300), dtype=np.float64))
    filled, supernodes = _symbolic(a, max_size=256)
    assert [len(node) for node in supernodes] == [256, 44]
    assert build_supernode_parent(filled.parent, supernodes) == [1, -1]
    fronts = build_front_indices_from_filled(supernodes, filled.columns)
    assert fronts[0] == list(range(300))
    assert fronts[1] == list(range(256, 300))


def test_arrow_matrix_fill_creates_update_rows():
    # Column 0 couples to every later column; after eliminating 0 the
    # remaining columns form a clique, so node 0's front covers all variables.
    n = 5
    a = sp.lil_matrix((n, n))
    a.setdiag(1.0)
    a[0, :] = 1.0
    a[:, 0] = 1.0
    filled, supernodes = _symbolic(sp.csr_matrix(a))
    assert supernodes == [[0, 1, 2, 3, 4]]
    assert filled.fill_edge_count == (n - 1) * (n - 2) // 2


def test_supernodes_reject_invalid_inputs():
    filled = symbolic_fill_pattern(sp.identity(3, format="csr"))
    with pytest.raises(ValueError):
        build_supernodes_from_filled(filled.parent, filled.columns, max_size=0)
    with pytest.raises(ValueError):
        build_supernodes_from_filled(filled.parent, filled.columns[:2])
