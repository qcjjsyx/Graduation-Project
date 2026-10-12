"""Test-matrix families with a controlled condition number.

Theory: docs/theory/数值误差分析理论基础.md §3.4, §7.7.

* Sparsity patterns come from the repository's real matrices
  (pattern(A) ∪ pattern(Aᵀ) ∪ diag) or from a random symmetric pattern, so
  the symbolic structure (elimination tree, fronts) resembles the real cases.
* F1 (kappa only, growth ~ 1): a weighted graph Laplacian L = D - W on the
  pattern (edge weights U(0.1, 1), so every off-diagonal entry is negative,
  like a circuit nodal matrix), shifted A = L - s*I so that kappa_2(A) =
  lambda_max/lambda_min hits the target exactly.  A stays SPD, so static
  elimination has no element growth.  The near-null direction is the
  constant vector (L*1 = 0, one per connected component); it is spread over
  all nodes and survives any row scaling (D_r*L*1 = 0), so the pipeline's
  equilibration cannot remove the ill-conditioning.
  (Revised 2026-10-11: the first version used a diagonally dominant matrix
  with diagonal = row sum + 1; its smallest eigenvector localized on weakly
  coupled low-degree nodes, and row scaling removed up to 5 decades of the
  target kappa on two of the three random patterns.)
* F2 (kappa + growth): A = S + gamma*||S||_2*K with S an F1 matrix and K a
  skew-symmetric matrix on the same pattern (||K||_2 = 1).  The symmetric part
  is S (positive definite), so static LU never meets a zero pivot, while the
  growth bound n(||T|| + ||K^T T^-1 K||) increases with gamma.
  Measured (2026-10-11): the skew part also lifts the near-null eigenvalue,
  so kappa(F2) collapses to ~10-300 whatever the target; F2 therefore covers
  the "small kappa, large growth" corner only (growth up to ~3e3 at gamma >= 1).
* F3 (kappa controlled, growth measured): M = L + gamma*||L||_2*K is
  nonsymmetric; A = M - mu*I with mu just below the smallest real eigenvalue
  of M, tuned so that kappa_2(A) hits the target.  The symmetric part is no
  longer definite (no zero-pivot guarantee, closer to the real matrices);
  growth is not monotone in gamma and is recorded, not prescribed.
* "real": the repository matrix itself (anchor points).

Dense eigen/SVD computations are used: intended for n <= ~2000.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.matrix_io import read_mat_file

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "example"
REAL_MATRICES = {
    "jj256": "256X256JJ.mat",
    "jj576": "576X576JJ.mat",
    "jj1024": "1024X1024JJ.mat",
}


def load_real(name: str) -> sp.csr_matrix:
    if name not in REAL_MATRICES:
        raise ValueError(f"unknown real matrix {name!r}; known: {sorted(REAL_MATRICES)}")
    return read_mat_file(str(EXAMPLE_DIR / REAL_MATRICES[name])).tocsr().astype(np.float64)


def symmetric_pattern(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Boolean pattern(A) ∪ pattern(Aᵀ) ∪ diag as a 0/1 CSR matrix."""

    a = sp.csr_matrix(matrix)
    a.data = np.ones_like(a.data)
    n = a.shape[0]
    p = (abs(a) + abs(a.T) + sp.eye(n, format="csr")).tocsr()
    p.data = np.ones_like(p.data)
    p.eliminate_zeros()
    return p


def load_pattern(name: str, n: int | None = None, density: float | None = None,
                 seed: int = 0) -> sp.csr_matrix:
    """Pattern of a real matrix (jj256/jj576/jj1024) or a random one."""

    if name == "random":
        if n is None or density is None:
            raise ValueError("random pattern needs n and density")
        rng = np.random.default_rng(seed)
        r = sp.random(n, n, density=density, format="csr", random_state=rng)
        return symmetric_pattern(r)
    return symmetric_pattern(load_real(name))


def laplacian_on_pattern(pattern: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Weighted graph Laplacian L = D - W with the given pattern.

    Edge weights w_ij ~ U(0.1, 1) (bounded away from 0 so no node is weakly
    coupled), W symmetric, D = diag(row sums of W).  L is symmetric positive
    semidefinite with L*1 = 0.
    """

    upper = sp.triu(pattern, k=1).tocoo()
    weights = rng.uniform(0.1, 1.0, upper.nnz)
    w = sp.coo_matrix((weights, (upper.row, upper.col)), shape=pattern.shape)
    w = (w + w.T).tocsr()
    degree = np.asarray(w.sum(axis=1)).ravel()
    return (sp.diags(degree) - w).tocsr()


def extreme_eigenvalues(symmetric: sp.spmatrix) -> tuple[float, float]:
    w = np.linalg.eigvalsh(symmetric.toarray())
    return float(w[0]), float(w[-1])


def shift_to_kappa(b: sp.csr_matrix, kappa: float) -> sp.csr_matrix:
    """A = B - s*I with kappa_2(A) = kappa (B symmetric PSD; A is SPD).

    Eigenvalues shift by -s: (lmax - s) / (lmin - s) = kappa gives
    s = (kappa*lmin - lmax) / (kappa - 1).  Works for kappa below kappa(B)
    as well (s < 0).
    """

    if not kappa > 1.0:
        raise ValueError("kappa must exceed 1")
    lmin, lmax = extreme_eigenvalues(b)
    s = (kappa * lmin - lmax) / (kappa - 1.0)
    return (b - s * sp.eye(b.shape[0], format="csr")).tocsr()


def skew_on_pattern(pattern: sp.csr_matrix, rng: np.random.Generator) -> sp.csr_matrix:
    """Skew-symmetric K (K = -Kᵀ) on the off-diagonal pattern, ||K||_2 = 1."""

    upper = sp.triu(pattern, k=1).tocoo()
    values = rng.uniform(-1.0, 1.0, upper.nnz)
    r = sp.coo_matrix((values, (upper.row, upper.col)), shape=pattern.shape)
    k = (r - r.T).tocsr()
    norm = np.linalg.norm(k.toarray(), 2)
    if norm == 0.0:
        raise ValueError("pattern has no off-diagonal entries")
    return (k / norm).tocsr()


@dataclass(frozen=True)
class MatrixSpec:
    family: str                 # "F1" | "F2" | "F3" | "real"
    pattern: str                # "jj256" | "jj576" | "jj1024" | "random"
    n: int | None = None        # random pattern only
    density: float | None = None
    kappa: float | None = None  # target kappa_2 of the SPD part (F1/F2)
    gamma: float = 0.0          # F2 skew weight (relative to ||S||_2)
    seed: int = 0               # pattern (random) and value seed

    @property
    def case_id(self) -> str:
        if self.family == "real":
            return f"real-{self.pattern}"
        size = f"{self.pattern}{self.n}d{self.density:g}" if self.pattern == "random" else self.pattern
        kappa = f"k{np.log10(self.kappa):.1f}"
        gamma = f"-g{self.gamma:g}" if self.family in ("F2", "F3") else ""
        return f"{self.family}-{size}-{kappa}{gamma}-s{self.seed}"


def build_matrix(spec: MatrixSpec) -> sp.csr_matrix:
    if spec.family == "real":
        return load_real(spec.pattern)
    if spec.kappa is None:
        raise ValueError("F1/F2 need a target kappa")
    rng = np.random.default_rng(spec.seed)
    pattern = load_pattern(spec.pattern, spec.n, spec.density, spec.seed)
    s = shift_to_kappa(laplacian_on_pattern(pattern, rng), spec.kappa)
    if spec.family == "F1":
        return s
    if spec.family == "F2":
        k = skew_on_pattern(pattern, rng)
        s_norm = np.linalg.norm(s.toarray(), 2)
        return (s + spec.gamma * s_norm * k).tocsr()
    if spec.family == "F3":
        b = laplacian_on_pattern(pattern, rng)
        k = skew_on_pattern(pattern, rng)
        m = b.toarray() + spec.gamma * np.linalg.norm(b.toarray(), 2) * k.toarray()
        mu = shift_below_real_eigenvalue(m, spec.kappa)
        return sp.csr_matrix(m - mu * np.eye(m.shape[0]))
    raise ValueError(f"unknown family {spec.family!r}")


def shift_below_real_eigenvalue(m: np.ndarray, kappa: float, rtol: float = 0.01) -> float:
    """mu < lambda (smallest real eigenvalue of m) with kappa_2(m - mu*I) ~ kappa.

    Near a simple real eigenvalue sigma_min(m - mu I) ~ c*(lambda - mu), so
    kappa ~ C/t with t = lambda - mu; iterate t <- t * kappa(t) / target,
    falling back to bisection on log t if the update overshoots.
    """

    eigenvalues = np.linalg.eigvals(m)
    real = np.sort(eigenvalues[np.abs(eigenvalues.imag) <= 1e-9 * np.abs(eigenvalues).max()].real)
    if real.size == 0:
        raise ValueError("matrix has no real eigenvalue to shift towards")
    lam = float(real[0])
    eye = np.eye(m.shape[0])

    def cond(t: float) -> float:
        s = np.linalg.svd(m - (lam - t) * eye, compute_uv=False)
        return float(s[0] / s[-1]) if s[-1] > 0 else np.inf

    scale = float(np.linalg.norm(m, 2))
    t = scale / kappa
    lo, hi = 0.0, np.inf  # bracket on t: cond(lo-side) > target > cond(hi-side)
    for _ in range(60):
        k = cond(t)
        if abs(np.log(k / kappa)) <= rtol:
            return lam - t
        if k > kappa:
            lo = t
        else:
            hi = t
        proposal = t * k / kappa
        if not (lo < proposal < hi):
            proposal = np.sqrt(lo * hi) if lo > 0 and np.isfinite(hi) else (t * 10 if k > kappa else t / 10)
        t = proposal
    raise ValueError(f"could not reach kappa {kappa:.1e} (last {k:.2e})")
