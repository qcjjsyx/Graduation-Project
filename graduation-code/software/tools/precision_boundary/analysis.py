"""Host-side metrics of one compiled artifact (+ optional device run).

Theory and the meaning of every metric: docs/theory/数值误差分析理论基础.md
§9.  Host-only: nothing is fed back into the device image.

Systems:
* original:     A x = b                  (forward error vs x_true lives here)
* transformed:  A_hat x_hat = b_hat,     A_hat = Q^T P D_r A Q (what the device factors)
  kappa(D_r A) == kappa(A_hat) (permutations do not change it).

Iterative refinement emulates the device solve on the host: corrections use
the device's FP32 global L/U with FP32 triangular solves (LAPACK strtrs), the
residual is FP64 and the iterate is accumulated in FP64.  The first iterate is
the device's own solution_data; the gap between it and the host FP32 solve is
reported so the emulation can be judged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg as sl
import scipy.sparse.linalg as spla

from tools.numeric_diagnosis import Artifact, node_diagnosis

F32 = np.float32
U32 = float(np.finfo(np.float32).eps) / 2
U64 = float(np.finfo(np.float64).eps) / 2


def _backward(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Oettli-Prager componentwise backward error (theory doc §5)."""
    r = b - a @ x
    den = np.abs(a) @ np.abs(x) + np.abs(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, np.abs(r) / den, np.where(r == 0, 0.0, np.inf))
    return float(ratio.max())


def _forward(x: np.ndarray, x_true: np.ndarray) -> float:
    return float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true))


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


class Systems:
    def __init__(self, art: Artifact) -> None:
        self.a, self.b = art.original_system()
        self.d = art.row_scale()
        self.a_hat, self.b_hat, self.q = art.transformed_system()
        path = art.art / "x_true_f64.bin"
        self.x_true = np.fromfile(path, "<f8") if path.exists() else None

    def to_original(self, x_hat: np.ndarray) -> np.ndarray:
        x = np.empty_like(x_hat)
        x[self.q] = x_hat
        return x

    def metrics(self, x_hat: np.ndarray) -> dict:
        """Original-equation metrics of a transformed-space iterate."""
        if not _finite(x_hat):
            return {"finite": False}
        x = self.to_original(x_hat)
        out = {"finite": True, "backward_error": _backward(self.a, x, self.b)}
        if self.x_true is not None:
            out["forward_error"] = _forward(x, self.x_true)
        return out


def conditioning(sys_: Systems) -> dict:
    """kappa_2 of A and D_r A (SVD), exact and estimated kappa_1 of D_r A,
    Skeel cond(D_r A, x_true) (theory doc §3)."""

    da = sys_.a * sys_.d[:, None]
    sigma_a = np.linalg.svd(sys_.a, compute_uv=False)
    sigma = np.linalg.svd(da, compute_uv=False)
    rel = sigma / sigma[0]
    inv = np.linalg.inv(da)
    kappa1 = float(np.linalg.norm(da, 1) * np.linalg.norm(inv, 1))
    lu = sl.lu_factor(da)
    n = da.shape[0]
    inv_op = spla.LinearOperator(
        (n, n),
        matvec=lambda v: sl.lu_solve(lu, v),
        rmatvec=lambda v: sl.lu_solve(lu, v, trans=1),
        dtype=np.float64,
    )
    kappa1_est = float(np.linalg.norm(da, 1) * spla.onenormest(inv_op))
    out = {
        "n": int(n),
        "nnz": int(np.count_nonzero(sys_.a)),
        "kappa2_A": float(sigma_a[0] / sigma_a[-1]),
        "kappa2_DrA": float(sigma[0] / sigma[-1]),
        "kappa1_DrA": kappa1,
        "kappa1_DrA_estimate": kappa1_est,
        "smallest_rel_sigma": [float(v) for v in rel[-6:][::-1]],
        "count_rel_sigma_below_u32": int(np.sum(rel < U32)),
    }
    if sys_.x_true is not None:
        x = sys_.x_true
        out["skeel_DrA_xtrue"] = float(
            np.max(np.abs(inv) @ (np.abs(da) @ np.abs(x))) / np.max(np.abs(x))
        )
    return out


def _refine(sys_: Systems, solve32, x0: np.ndarray, steps: int) -> list[dict]:
    """FP64-residual iterative refinement with an FP32 correction solver."""
    history = [{"step": 0, **sys_.metrics(x0)}]
    x = x0.copy()
    for k in range(1, steps + 1):
        if not _finite(x):
            break
        r = sys_.b_hat - sys_.a_hat @ x
        d = solve32(r)
        x = x + d
        entry = {"step": k, **sys_.metrics(x)}
        if _finite(x) and np.linalg.norm(x) > 0:
            entry["correction_norm_ratio"] = float(np.linalg.norm(d) / np.linalg.norm(x))
        history.append(entry)
    return history


def references(sys_: Systems, ir_steps: int) -> dict:
    """FP64 dense solve (lower bound) and FP32 partial-pivoting LU (+IR)."""
    out: dict = {}
    x64 = np.linalg.solve(sys_.a, sys_.b)
    out["fp64_dense"] = {"backward_error": _backward(sys_.a, x64, sys_.b)}
    if sys_.x_true is not None:
        out["fp64_dense"]["forward_error"] = _forward(x64, sys_.x_true)
    a32 = sys_.a_hat.astype(F32)
    p, l, u = sl.lu(a32)
    out["fp32_gepp_growth"] = float(
        (np.abs(l.astype(np.float64)) @ np.abs(u.astype(np.float64))).max()
        / np.abs(sys_.a_hat).max()
    )
    factors = sl.lu_factor(a32)

    def solve32(r: np.ndarray) -> np.ndarray:
        return sl.lu_solve(factors, r.astype(F32)).astype(np.float64)

    out["fp32_gepp_ir"] = _refine(sys_, solve32, solve32(sys_.b_hat), ir_steps)
    return out


def device_factors(art: Artifact) -> tuple[np.ndarray, np.ndarray]:
    """Dense global FP32 L (unit lower) and U in the transformed variables."""
    n = art.n
    lower = np.zeros((n, n), dtype=F32)
    upper = np.zeros((n, n), dtype=F32)
    for key, node in art.manifest["nodes"].items():
        total, pivot = node["total_dim"], node["pivot_dim"]
        g = np.asarray(node["front_indices"])
        lf = art.region(art.image1, f"node.{key}.factor_l", total, pivot).astype(F32)
        uf = art.region(art.image1, f"node.{key}.factor_u", pivot, total).astype(F32)
        for j in range(pivot):
            lower[g[j:], g[j]] = lf[j:, j]
            upper[g[j], g[j:]] = uf[j, j:]
    return lower, upper


def device_analysis(art: Artifact, sys_: Systems, ir_steps: int) -> dict:
    lower, upper = device_factors(art)
    l64, u64 = lower.astype(np.float64), upper.astype(np.float64)
    scale = np.abs(sys_.a_hat).max()
    out = {
        "static_growth": float((np.abs(l64) @ np.abs(u64)).max() / scale),
        "global_lu_backward": float(np.abs(sys_.a_hat - l64 @ u64).max() / scale),
        "min_abs_u_diag_ratio": float(np.min(np.abs(np.diag(u64))) / scale),
    }
    nodes = node_diagnosis(art)
    back = [r["backward_error"] for r in nodes]
    out["node_backward_max"] = float(max(back))
    out["node_backward_median"] = float(np.median(back))
    out["node_min_pivot_ratio"] = float(min(r["pivot_ratio"] for r in nodes))
    out["node_max_abs_l"] = float(max(r["max_abs_l"] for r in nodes))

    def solve32(r: np.ndarray) -> np.ndarray:
        y = sl.solve_triangular(lower, r.astype(F32), lower=True, unit_diagonal=True,
                                check_finite=False)
        return sl.solve_triangular(upper, y, check_finite=False).astype(np.float64)

    host_first = solve32(sys_.b_hat)
    x0 = art.device_solution()
    if x0 is None:
        out["device_solution_available"] = False
        x0 = host_first
    else:
        out["device_solution_available"] = True
        denom = np.linalg.norm(host_first) or 1.0
        out["device_vs_host_fp32_solve"] = float(np.linalg.norm(x0 - host_first) / denom)
    out["ir"] = _refine(sys_, solve32, x0, ir_steps)
    return out


def analyze(art_dir: Path, rep_dir: Path | None, ir_steps: int = 10,
            device_ok: bool = False) -> dict:
    art = Artifact(art_dir, rep_dir if device_ok else None)
    sys_ = Systems(art)
    result = {"conditioning": conditioning(sys_), "references": references(sys_, ir_steps)}
    if device_ok:
        result["device"] = device_analysis(art, sys_, ir_steps)
    return result
