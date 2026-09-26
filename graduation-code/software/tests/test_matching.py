import numpy as np
import pytest
import scipy.sparse as sp

from src.config import MatchingConfig
from src.matching import (
    MatchingError,
    canonicalize_csr,
    compute_row_matching,
    gather_rows,
)


def csr_from_dense(dense) -> sp.csr_matrix:
    return sp.csr_matrix(np.asarray(dense, dtype=np.float64))


def test_max_product_differs_from_greedy_and_from_max_sum():
    greedy_trap = csr_from_dense([[10.0, 9.0], [9.0, 1.0]])
    matching = compute_row_matching(greedy_trap, MatchingConfig())
    assert matching.full_matching
    assert matching.row_perm.tolist() == [1, 0]
    assert matching.matched_value.tolist() == [9.0, 9.0]
    assert matching.log_product == pytest.approx(np.log(81.0))

    sum_trap = csr_from_dense([[5.0, 4.0], [4.0, 1.0]])
    matching = compute_row_matching(sum_trap, MatchingConfig())
    assert matching.row_perm.tolist() == [1, 0]
    assert matching.matched_value.tolist() == [4.0, 4.0]
    assert matching.log_product == pytest.approx(np.log(16.0))


def test_log_objective_is_equivalent_to_max_product():
    dense = np.array(
        [
            [3.0, 0.0, 7.0],
            [0.0, 5.0, 1.0],
            [2.0, 4.0, 0.0],
        ]
    )
    matrix = csr_from_dense(dense)
    matching = compute_row_matching(matrix, MatchingConfig())
    product = float(np.prod(np.abs(matching.matched_value)))
    best_product = 0.0
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        values = [abs(dense[row, perm[row]]) for row in range(3)]
        if all(value != 0.0 for value in values):
            best_product = max(best_product, float(np.prod(values)))
    assert product == pytest.approx(best_product)
    assert matching.log_product == pytest.approx(np.log(best_product))


def test_row_permutation_moves_matched_nonzeros_to_the_diagonal():
    dense = np.array(
        [
            [0.0, 2.0, 0.0],
            [3.0, 0.0, 4.0],
            [0.0, 5.0, 6.0],
        ]
    )
    matrix = csr_from_dense(dense)
    matching = compute_row_matching(matrix, MatchingConfig())
    gathered = gather_rows(matrix, matching.row_perm).toarray()
    for index in range(3):
        assert gathered[index, index] != 0.0
        assert gathered[index, index] == dense[matching.row_perm[index], index]
    diagonal_product = float(np.prod(np.abs(np.diag(gathered))))
    assert diagonal_product == pytest.approx(np.prod(np.abs(matching.matched_value)))


def test_row_permutation_enables_static_pivoting_on_zero_diagonal():
    swap_only = csr_from_dense([[0.0, 1.0], [1.0, 0.0]])
    matching = compute_row_matching(swap_only, MatchingConfig())
    assert matching.row_perm.tolist() == [1, 0]
    gathered = gather_rows(swap_only, matching.row_perm).toarray()
    assert np.array_equal(gathered, np.eye(2))


def test_disabled_matching_is_identity_and_not_a_matching_claim():
    matrix = csr_from_dense([[0.0, 2.0], [3.0, 4.0]])
    matching = compute_row_matching(matrix, MatchingConfig(method="none"))
    assert matching.row_perm.tolist() == [0, 1]
    assert matching.full_matching is False
    assert matching.matched_column.tolist() == [0, 1]
    assert matching.matched_value.tolist() == [0.0, 4.0]
    assert matching.log_product is None


def test_incomplete_matching_reports_unmatched_rows():
    no_full_matching = csr_from_dense([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(MatchingError, match="unmatched rows"):
        compute_row_matching(no_full_matching, MatchingConfig())

    structurally_unmatched = csr_from_dense([[0.0, 1.0], [0.0, 2.0]])
    with pytest.raises(MatchingError, match="unmatched rows"):
        compute_row_matching(structurally_unmatched, MatchingConfig())


def test_matching_only_uses_stored_nonzeros_and_requires_finite_values():
    matrix = csr_from_dense([[1.0, 0.0], [0.0, 1.0]])
    matching = compute_row_matching(matrix, MatchingConfig())
    assert matching.matched_column.tolist() == [0, 1]
    assert np.all(matching.matched_value != 0.0)

    with pytest.raises(MatchingError, match="non-finite"):
        compute_row_matching(csr_from_dense([[np.nan, 1.0], [1.0, 1.0]]), MatchingConfig())
    with pytest.raises(MatchingError, match="non-finite"):
        compute_row_matching(csr_from_dense([[np.inf, 1.0], [1.0, 1.0]]), MatchingConfig())


def test_canonicalization_is_deterministic_and_removes_explicit_zeros():
    rows = np.array([0, 1, 1, 0])
    cols = np.array([0, 0, 1, 1])
    data = np.array([1.0, 2.0, 0.0, 3.0])
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(2, 2))
    first = canonicalize_csr(matrix)
    second = canonicalize_csr(matrix.T.T)
    assert first.toarray().tolist() == [[1.0, 3.0], [2.0, 0.0]]
    assert first.indptr.tolist() == second.indptr.tolist()
    assert first.indices.tolist() == second.indices.tolist()
    assert first.data.tolist() == second.data.tolist()
    assert 0.0 not in first.data.tolist()

    matching_first = compute_row_matching(matrix, MatchingConfig())
    matching_second = compute_row_matching(matrix.T.T, MatchingConfig())
    assert matching_first.row_perm.tolist() == matching_second.row_perm.tolist()
    assert matching_first.matched_column.tolist() == (
        matching_second.matched_column.tolist()
    )


def test_gather_rows_follows_gather_convention():
    matrix = csr_from_dense([[1.0, 2.0], [3.0, 4.0]])
    gathered = gather_rows(matrix, np.array([1, 0])).toarray()
    assert np.array_equal(gathered, np.array([[3.0, 4.0], [1.0, 2.0]]))
    with pytest.raises(MatchingError, match="permutation"):
        gather_rows(matrix, np.array([0, 0]))
