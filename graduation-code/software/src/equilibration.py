from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from src.config import EquilibrationConfig


@dataclass(frozen=True)
class EquilibrationResult:
    matrix: sp.csr_matrix
    rhs: np.ndarray
    row_scale_exponents: np.ndarray
    row_max_before_min: float
    row_max_before_max: float
    row_max_after_min: float
    row_max_after_max: float


def equilibrate_system(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: EquilibrationConfig,
) -> EquilibrationResult:
    """Apply a sparsity-preserving power-of-two row equilibration.

    The transformed system is ``D_r A x = D_r b``.  Because there is no
    column scaling, the mathematical solution ``x`` is unchanged.  Restricting
    ``D_r`` to powers of two makes the transform an exponent adjustment rather
    than a mantissa multiplication in the hardware data path.
    """

    csr = matrix.tocsr().astype(np.float64)
    vector = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if csr.shape[0] != vector.size:
        raise ValueError("matrix/RHS dimension mismatch during equilibration")

    row_max_before = _row_max_abs(csr)
    exponents = np.zeros(csr.shape[0], dtype=np.int16)
    if config.mode == "pow2-row":
        nonzero = row_max_before > 0.0
        raw = np.zeros(csr.shape[0], dtype=np.float64)
        raw[nonzero] = -np.rint(np.log2(row_max_before[nonzero]))
        raw = np.clip(
            raw,
            -config.max_scale_exponent,
            config.max_scale_exponent,
        )
        exponents = raw.astype(np.int16)
    elif config.mode != "none":
        raise ValueError(f"unsupported equilibration mode {config.mode!r}")

    scales = np.exp2(exponents.astype(np.float64))
    scaled_matrix = (sp.diags(scales, format="csr") @ csr).tocsr()
    scaled_rhs = scales * vector
    if not np.all(np.isfinite(scaled_matrix.data)) or not np.all(
        np.isfinite(scaled_rhs)
    ):
        raise ValueError("equilibration produced non-finite values")

    row_max_after = _row_max_abs(scaled_matrix)
    before_nonzero = row_max_before[row_max_before > 0.0]
    after_nonzero = row_max_after[row_max_after > 0.0]
    return EquilibrationResult(
        matrix=scaled_matrix,
        rhs=scaled_rhs,
        row_scale_exponents=exponents,
        row_max_before_min=_safe_min(before_nonzero),
        row_max_before_max=_safe_max(before_nonzero),
        row_max_after_min=_safe_min(after_nonzero),
        row_max_after_max=_safe_max(after_nonzero),
    )


def _row_max_abs(matrix: sp.csr_matrix) -> np.ndarray:
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for row in range(matrix.shape[0]):
        begin = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        if begin != end:
            result[row] = float(np.max(np.abs(matrix.data[begin:end])))
    return result


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0
