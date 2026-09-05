from __future__ import annotations

import scipy.sparse as sp


def symmetric_sparsity_pattern(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Return the structural envelope ``pattern(A) union pattern(A.T)``.

    Numeric values are deliberately discarded before the union so opposite
    entries cannot cancel.  The result is the single symbolic graph consumed
    by ordering, fill, etree, supernode, and front construction.
    """

    _validate_square_sparse(matrix)
    pattern = (matrix != 0).astype("int8").tocsr()
    envelope = (pattern + pattern.T).astype(bool).tocsr()
    envelope.sum_duplicates()
    envelope.eliminate_zeros()
    return envelope


def is_structurally_symmetric(matrix: sp.spmatrix) -> bool:
    """Report whether the input pattern already equals its transpose."""

    _validate_square_sparse(matrix)
    pattern = (matrix != 0).astype(bool).tocsr()
    return (pattern != pattern.T).nnz == 0


def _validate_square_sparse(matrix: sp.spmatrix) -> None:
    if not sp.isspmatrix(matrix):
        raise TypeError("matrix must be a scipy sparse matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
