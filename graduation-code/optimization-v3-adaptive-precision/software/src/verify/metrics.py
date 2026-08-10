from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def residual_norm(a: sp.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    r = b - a @ x
    return float(np.linalg.norm(r))


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