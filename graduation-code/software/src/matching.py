"""Host-side numeric preprocessing: global one-to-one weighted row matching.

The stage runs before symbolic analysis and is completely independent of the
golden checker and of the FP64 reference.  It only reorders equations (rows);
it never changes variables (columns).

Objective
---------
``weighted_matching`` computes a *maximum-product perfect matching* on the
nonzero pattern of the input matrix: among all bijections ``sigma`` with
``A[j, sigma(j)] != 0`` it maximizes ``prod_j |A[j, sigma(j)]|``.  Because
``log`` is strictly increasing this is exactly equivalent to maximizing
``sum_j log|A[j, sigma(j)]|``; the implementation solves that equivalent
logarithmic objective with the sparse LAPJVsp assignment solver already
provided by SciPy (``scipy.sparse.csgraph.min_weight_full_bipartite_matching``).

This is deliberately *not* a re-implementation of MC64: no scaling passes, no
partial matchings, no rank-one updates.  It is exactly the max-product perfect
matching stage described above.  Matching edges are restricted to structurally
stored nonzeros of the canonicalized input; explicit zeros are removed first so
they can never be selected.

Permutation convention
----------------------
All permutation arrays in this project use the *gather* direction: with
permutation ``p``, the permuted array is ``y[i] = x[p[i]]`` ("new position
``i`` is sourced from old position ``p[i]``").  For a matrix, row-gathering
``B = P A`` means ``B[i, :] = A[p[i], :]``.

Given the matching ``sigma`` (row ``j`` matched to column ``sigma(j)``), the
returned row permutation is ``p_match = sigma^{-1}`` so that
``B[i, i] = A[p_match[i], i]`` is the matched nonzero of column ``i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from src.config import MatchingConfig


class MatchingError(ValueError):
    """Diagnostic failure of the row-matching preprocessing stage."""


@dataclass(frozen=True)
class RowMatching:
    """Result of the host row-matching stage.

    ``row_perm``: gather permutation for rows, ``B[i, :] = A[row_perm[i], :]``.
    ``matched_column[j]``: column matched to original row ``j`` (identity when
    the stage is disabled; it is then *not* a matching claim).
    ``matched_value[j]``: ``A[j, matched_column[j]]`` (diagonal entry when the
    stage is disabled).
    """

    method: str
    objective: str
    row_perm: np.ndarray
    matched_column: np.ndarray
    matched_value: np.ndarray
    full_matching: bool
    # Objective value sum(log|matched|); None when the stage is disabled.
    log_product: float | None


def canonicalize_csr(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Return a deterministic CSR copy: float64, sorted, deduplicated, no zeros.

    The canonical ordering makes identical input produce identical sparse
    storage order, which is required for byte-reproducible artifacts and for
    deterministic matching under one environment.
    """

    if not sp.isspmatrix(matrix):
        raise TypeError("matrix must be a scipy sparse matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    csr = matrix.tocsr().astype(np.float64)
    csr.sum_duplicates()
    if not np.all(np.isfinite(csr.data)):
        raise MatchingError("matrix contains non-finite values")
    csr.eliminate_zeros()
    csr.sort_indices()
    csr.has_sorted_indices = True  # type: ignore[attr-defined]
    return csr


def compute_row_matching(
    matrix: sp.spmatrix, config: MatchingConfig
) -> RowMatching:
    """Compute the row permutation of the host preprocessing stage."""

    csr = canonicalize_csr(matrix)
    dimension = csr.shape[0]
    if dimension == 0:
        raise MatchingError("matrix must not be empty")

    if config.method == "none":
        diagonal = _diagonal_values(csr)
        return RowMatching(
            method="none",
            objective=config.objective,
            row_perm=np.arange(dimension, dtype=np.int32),
            matched_column=np.arange(dimension, dtype=np.int32),
            matched_value=diagonal,
            full_matching=False,
            log_product=None,
        )
    if config.method != "weighted_matching":
        raise MatchingError(f"unsupported matching method {config.method!r}")
    return _max_product_matching(csr, config)


def _max_product_matching(csr: sp.csr_matrix, config: MatchingConfig) -> RowMatching:
    dimension = csr.shape[0]
    magnitudes = np.abs(csr.data)
    if magnitudes.size == 0:
        raise MatchingError(
            "matrix has no stored nonzeros; no complete matching can exist"
        )
    if not np.all(np.isfinite(magnitudes)) or np.any(magnitudes == 0.0):
        raise MatchingError(
            "matching requires finite nonzero entries; canonicalize first"
        )

    # Equivalent logarithmic objective of max-product matching.  The constant
    # shift keeps every weight strictly positive: the SciPy sparse assignment
    # solver treats explicit zeros as non-edges.
    log_weights = np.log(magnitudes)
    weights = log_weights - log_weights.min() + 1.0
    weighted = csr.copy()
    weighted.data = weights

    try:
        row_ind, col_ind = min_weight_full_bipartite_matching(
            weighted, maximize=True
        )
    except ValueError as exc:
        raise MatchingError(_incomplete_matching_report(csr, exc)) from exc

    row_ind = np.asarray(row_ind, dtype=np.int64)
    col_ind = np.asarray(col_ind, dtype=np.int64)
    if row_ind.size != dimension or not np.array_equal(
        row_ind, np.arange(dimension)
    ):
        raise MatchingError("matching solver did not return a full row assignment")
    if np.unique(col_ind).size != dimension:
        raise MatchingError("matching solver returned a non-bijective assignment")

    matched_value = np.empty(dimension, dtype=np.float64)
    for row in range(dimension):
        begin, end = csr.indptr[row], csr.indptr[row + 1]
        column = int(col_ind[row])
        position = int(np.searchsorted(csr.indices[begin:end], column))
        if (
            begin + position >= end
            or int(csr.indices[begin + position]) != column
        ):
            raise MatchingError(
                f"matched position ({row}, {column}) is not a stored nonzero"
            )
        value = float(csr.data[begin + position])
        if value == 0.0 or not np.isfinite(value):
            raise MatchingError(
                f"matched entry ({row}, {column}) is zero or non-finite"
            )
        matched_value[row] = value

    row_perm = np.empty(dimension, dtype=np.int32)
    row_perm[col_ind] = np.arange(dimension, dtype=np.int32)

    for column in range(dimension):
        source_row = int(row_perm[column])
        if matched_value[source_row] != float(csr[source_row, column]):
            raise MatchingError(
                "row permutation does not move matched entries onto the diagonal"
            )

    log_product = float(np.sum(np.log(np.abs(matched_value))))
    return RowMatching(
        method="weighted_matching",
        objective="max_product",
        row_perm=row_perm,
        matched_column=col_ind.astype(np.int32),
        matched_value=matched_value,
        full_matching=True,
        log_product=log_product,
    )


def _incomplete_matching_report(csr: sp.csr_matrix, cause: Exception) -> str:
    pattern = (csr != 0).astype(np.int8).tocsr()
    unmatched_rows: list[int] = []
    unmatched_cols: list[int] = []
    try:
        by_row = maximum_bipartite_matching(pattern, perm_type="column")
        unmatched_rows = [int(index) for index, value in enumerate(by_row) if value < 0]
        by_col = maximum_bipartite_matching(pattern, perm_type="row")
        unmatched_cols = [int(index) for index, value in enumerate(by_col) if value < 0]
    except Exception:  # pragma: no cover - diagnostics must not mask the failure
        pass
    return (
        "no complete row matching exists on the nonzero pattern "
        f"({cause}); unmatched rows={unmatched_rows[:16]}"
        f"{'...' if len(unmatched_rows) > 16 else ''}, "
        f"unmatched columns={unmatched_cols[:16]}"
        f"{'...' if len(unmatched_cols) > 16 else ''}"
    )


def _diagonal_values(csr: sp.csr_matrix) -> np.ndarray:
    dimension = csr.shape[0]
    diagonal = np.zeros(dimension, dtype=np.float64)
    for row in range(dimension):
        begin, end = csr.indptr[row], csr.indptr[row + 1]
        position = int(np.searchsorted(csr.indices[begin:end], row))
        if begin + position < end and int(csr.indices[begin + position]) == row:
            diagonal[row] = float(csr.data[begin + position])
    return diagonal


def gather_rows(matrix: sp.spmatrix, row_perm: np.ndarray) -> sp.csr_matrix:
    """Row-gather ``B[i, :] = A[row_perm[i], :]`` (``B = P A``)."""

    csr = canonicalize_csr(matrix)
    perm = np.asarray(row_perm, dtype=np.int32)
    dimension = csr.shape[0]
    if perm.ndim != 1 or perm.size != dimension:
        raise MatchingError("row permutation length must match matrix dimension")
    if sorted(int(value) for value in perm) != list(range(dimension)):
        raise MatchingError("row permutation is not a valid 0..N-1 permutation")
    return csr[perm].tocsr()  # type: ignore[index]
