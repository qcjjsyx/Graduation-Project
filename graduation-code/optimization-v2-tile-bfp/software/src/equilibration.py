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
    column_scale_exponents: np.ndarray
    row_max_before_min: float
    row_max_before_max: float
    row_max_after_min: float
    row_max_after_max: float
    column_max_before_min: float
    column_max_before_max: float
    column_max_after_min: float
    column_max_after_max: float


def equilibrate_system(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    config: EquilibrationConfig,
) -> EquilibrationResult:
    """Apply sparsity-preserving power-of-two equilibration.

    Row-only mode solves ``D_r A x = D_r b``.  Row-column mode solves
    ``D_r A D_c y = D_r b`` and restores ``x = D_c y``.  Restricting both
    diagonal transforms to powers of two makes them exponent adjustments
    rather than general mantissa multiplications.
    """

    csr = matrix.tocsr().astype(np.float64)
    vector = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if csr.shape[0] != vector.size:
        raise ValueError("matrix/RHS dimension mismatch during equilibration")

    row_max_before = _row_max_abs(csr)
    column_max_before = _column_max_abs(csr)
    exponents = np.zeros(csr.shape[0], dtype=np.int16)
    column_exponents = np.zeros(csr.shape[1], dtype=np.int16)
    if config.mode == "pow2-row":
        exponents = _normalization_delta(
            row_max_before,
            np.zeros(csr.shape[0], dtype=np.int64),
            config.max_scale_exponent,
        )
    elif config.mode == "pow2-row-column":
        working = csr
        row_total = np.zeros(csr.shape[0], dtype=np.int64)
        column_total = np.zeros(csr.shape[1], dtype=np.int64)
        for _ in range(config.iterations):
            row_delta = _normalization_delta(
                _row_max_abs(working),
                row_total,
                config.max_scale_exponent,
            ).astype(np.int64)
            row_total += row_delta
            working = (
                sp.diags(np.exp2(row_delta.astype(np.float64)), format="csr")
                @ working
            ).tocsr()

            column_delta = _normalization_delta(
                _column_max_abs(working),
                column_total,
                config.max_scale_exponent,
            ).astype(np.int64)
            column_total += column_delta
            working = (
                working
                @ sp.diags(
                    np.exp2(column_delta.astype(np.float64)),
                    format="csr",
                )
            ).tocsr()
        exponents = row_total.astype(np.int16)
        column_exponents = column_total.astype(np.int16)
    elif config.mode == "pow2-ruiz":
        working = csr
        row_total = np.zeros(csr.shape[0], dtype=np.int64)
        column_total = np.zeros(csr.shape[1], dtype=np.int64)
        for _ in range(config.iterations):
            row_delta = _ruiz_delta(
                _row_max_abs(working),
                row_total,
                config.max_scale_exponent,
            ).astype(np.int64)
            column_delta = _ruiz_delta(
                _column_max_abs(working),
                column_total,
                config.max_scale_exponent,
            ).astype(np.int64)
            row_total += row_delta
            column_total += column_delta
            working = (
                sp.diags(
                    np.exp2(row_delta.astype(np.float64)),
                    format="csr",
                )
                @ working
                @ sp.diags(
                    np.exp2(column_delta.astype(np.float64)),
                    format="csr",
                )
            ).tocsr()
        exponents = row_total.astype(np.int16)
        column_exponents = column_total.astype(np.int16)
    elif config.mode != "none":
        raise ValueError(f"unsupported equilibration mode {config.mode!r}")

    row_scales = np.exp2(exponents.astype(np.float64))
    column_scales = np.exp2(column_exponents.astype(np.float64))
    scaled_matrix = (
        sp.diags(row_scales, format="csr")
        @ csr
        @ sp.diags(column_scales, format="csr")
    ).tocsr()
    scaled_rhs = row_scales * vector
    if not np.all(np.isfinite(scaled_matrix.data)) or not np.all(
        np.isfinite(scaled_rhs)
    ):
        raise ValueError("equilibration produced non-finite values")

    row_max_after = _row_max_abs(scaled_matrix)
    column_max_after = _column_max_abs(scaled_matrix)
    before_nonzero = row_max_before[row_max_before > 0.0]
    after_nonzero = row_max_after[row_max_after > 0.0]
    column_before_nonzero = column_max_before[column_max_before > 0.0]
    column_after_nonzero = column_max_after[column_max_after > 0.0]
    return EquilibrationResult(
        matrix=scaled_matrix,
        rhs=scaled_rhs,
        row_scale_exponents=exponents,
        column_scale_exponents=column_exponents,
        row_max_before_min=_safe_min(before_nonzero),
        row_max_before_max=_safe_max(before_nonzero),
        row_max_after_min=_safe_min(after_nonzero),
        row_max_after_max=_safe_max(after_nonzero),
        column_max_before_min=_safe_min(column_before_nonzero),
        column_max_before_max=_safe_max(column_before_nonzero),
        column_max_after_min=_safe_min(column_after_nonzero),
        column_max_after_max=_safe_max(column_after_nonzero),
    )


def _row_max_abs(matrix: sp.csr_matrix) -> np.ndarray:
    result = np.zeros(matrix.shape[0], dtype=np.float64)
    for row in range(matrix.shape[0]):
        begin = matrix.indptr[row]
        end = matrix.indptr[row + 1]
        if begin != end:
            result[row] = float(np.max(np.abs(matrix.data[begin:end])))
    return result


def _column_max_abs(matrix: sp.csr_matrix) -> np.ndarray:
    return _row_max_abs(matrix.transpose().tocsr())


def _normalization_delta(
    maxima: np.ndarray,
    current_exponents: np.ndarray,
    max_scale_exponent: int,
) -> np.ndarray:
    raw = np.zeros(maxima.size, dtype=np.int64)
    nonzero = maxima > 0.0
    raw[nonzero] = -np.rint(np.log2(maxima[nonzero])).astype(np.int64)
    target = np.clip(
        current_exponents + raw,
        -max_scale_exponent,
        max_scale_exponent,
    )
    return (target - current_exponents).astype(np.int16)


def _ruiz_delta(
    maxima: np.ndarray,
    current_exponents: np.ndarray,
    max_scale_exponent: int,
) -> np.ndarray:
    raw = np.zeros(maxima.size, dtype=np.int64)
    nonzero = maxima > 0.0
    raw[nonzero] = -np.rint(
        0.5 * np.log2(maxima[nonzero])
    ).astype(np.int64)
    target = np.clip(
        current_exponents + raw,
        -max_scale_exponent,
        max_scale_exponent,
    )
    return (target - current_exponents).astype(np.int16)


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0
