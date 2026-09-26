from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def residual_norm(a: sp.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    r = b - a @ x
    return float(np.linalg.norm(r))


def componentwise_backward_error(
    a: sp.spmatrix, x: np.ndarray, b: np.ndarray
) -> float:
    """Componentwise relative backward error of ``A x = b``.

    ``max_i |r_i| / (|b_i| + |A| |x|)_i`` with ``r = b - A x``.  A zero
    denominator with zero numerator counts as zero error; zero denominator
    with nonzero numerator counts as infinite error.
    """

    residual = np.asarray(b, dtype=np.float64).reshape(-1) - a @ np.asarray(
        x, dtype=np.float64
    ).reshape(-1)
    denominator = np.abs(np.asarray(b, dtype=np.float64).reshape(-1)) + abs(a) @ np.abs(
        np.asarray(x, dtype=np.float64).reshape(-1)
    )
    numerator = np.abs(residual)
    worst = 0.0
    for num, den in zip(numerator, denominator):
        if den == 0.0:
            ratio = 0.0 if num == 0.0 else float("inf")
        else:
            ratio = float(num / den)
        worst = max(worst, ratio)
    return worst


def iterative_refinement(
    a: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    solve_fn,
    max_iter: int = 5,
    tol: float = 1e-6,
) -> np.ndarray:
    """Framework for iterative refinement using a provided solver.

    solve_fn should solve A * dx = r and return dx.
    """
    x = x0.copy()
    for _ in range(max_iter):
        r = b - a @ x
        if np.linalg.norm(r) < tol:
            break
        dx = solve_fn(r)
        x += dx
    return x