import numpy as np
import pytest
import scipy.sparse as sp

from src.dataStruct import NodeRange
from src.pipeline import extract_local_contribution
from src.symbolic.fill import require_structurally_symmetric, symbolic_fill_pattern


def test_symbolic_fill_adds_elimination_clique_edge():
    matrix = sp.csr_matrix(
        np.array(
            [
                [4.0, 1.0, 1.0],
                [2.0, 5.0, 0.0],
                [3.0, 0.0, 6.0],
            ]
        )
    )
    filled = symbolic_fill_pattern(matrix)
    assert filled.columns == [[0, 1, 2], [1, 2], [2]]
    assert filled.parent.tolist() == [1, 2, -1]
    assert filled.fill_edge_count == 1


def test_numerically_asymmetric_matrix_is_allowed():
    matrix = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 4.0]]))
    require_structurally_symmetric(matrix)


def test_structurally_asymmetric_matrix_is_rejected():
    matrix = sp.csr_matrix(np.array([[1.0, 2.0], [0.0, 4.0]]))
    with pytest.raises(ValueError, match="structurally symmetric"):
        require_structurally_symmetric(matrix)


def test_local_contribution_owns_no_initial_update_update_block():
    matrix = sp.csr_matrix(
        np.array(
            [
                [4.0, 1.0, 2.0],
                [3.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
    )
    local = extract_local_contribution(
        matrix,
        front_indices=[0, 1, 2],
        node_range=NodeRange(start=0, end=1),
    )
    np.testing.assert_array_equal(
        local,
        np.array(
            [
                [4.0, 1.0, 2.0],
                [3.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
            ]
        ),
    )
