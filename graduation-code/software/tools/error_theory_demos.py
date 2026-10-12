"""Reproduce every numeric table of docs/theory/数值误差分析理论基础.md.

Small dense examples only (numpy/scipy, no device): unit roundoff, the 2x2
condition-number example, the Oettli-Prager construction, forward error vs
kappa, static-pivot growth, and FP32-factor / FP64-residual iterative
refinement.  Run from graduation-code/software:

    python tools/error_theory_demos.py
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sl

F32 = np.float32
U32 = float(np.finfo(np.float32).eps) / 2
U64 = float(np.finfo(np.float64).eps) / 2


def oettli_prager(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    r = b - a @ x
    return float(np.max(np.abs(r) / (np.abs(a) @ np.abs(x) + np.abs(b))))


def rel(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def lu_nopivot_fp32(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    work = a.astype(F32).copy()
    n = len(work)
    lower = np.eye(n, dtype=F32)
    for k in range(n):
        for i in range(k + 1, n):
            lower[i, k] = work[i, k] / work[k, k]
            work[i, k:] -= lower[i, k] * work[k, k:]
            work[i, k] = 0
    return lower, np.triu(work)


def solve_lu_fp32(lower: np.ndarray, upper: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = sl.solve_triangular(lower, b.astype(F32), lower=True, unit_diagonal=True)
    return sl.solve_triangular(upper, y).astype(np.float64)


def growth(a: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    product = np.abs(lower.astype(np.float64)) @ np.abs(upper.astype(np.float64))
    return float(product.max() / np.abs(a).max())


def demo_roundoff() -> None:
    print("[1] unit roundoff")
    print(f"    u_fp32 = {U32:.2e}   u_fp64 = {U64:.2e}")
    print(f"    0.1 stored in FP32 = {float(F32(0.1))!r}")
    print(f"    FP32: 1 - 1e8 = {float(F32(1) - F32(1e8)):.1f}")


def demo_condition_2x2() -> None:
    print("[2] A = [[1,1],[1,1+eps]], x = [1,1], perturb b[1] by 1e-6")
    for eps in (1e-1, 1e-4):
        a = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
        x = np.array([1.0, 1.0])
        b = a @ x
        db = np.array([0.0, 1e-6])
        xp = np.linalg.solve(a, b + db)
        print(
            f"    eps={eps:.0e} kappa={np.linalg.cond(a):.0f} "
            f"rel db={np.linalg.norm(db) / np.linalg.norm(b):.1e} "
            f"rel dx={rel(xp, x):.1e} x'={xp}"
        )


def demo_oettli_prager() -> None:
    print("[3] Oettli-Prager: A=[[2,1],[1,3]], b=[3,4], x_hat=[1.01,0.99]")
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([3.0, 4.0])
    xh = np.array([1.01, 0.99])
    r = b - a @ xh
    d = np.abs(a) @ np.abs(xh) + np.abs(b)
    theta = r / d
    print(f"    r={r}  denominator={d}  ratios={np.abs(theta)}  eta={np.abs(theta).max():.3e}")
    da = theta[:, None] * np.abs(a) * np.sign(xh)[None, :]
    db = -theta * np.abs(b)
    print(f"    dA={da.round(5).tolist()}  db={db.round(5).tolist()}")
    print(f"    (A+dA)x_hat-(b+db) = {(a + da) @ xh - (b + db)}")
    print(f"    max|dA|/|A|={np.max(np.abs(da) / np.abs(a)):.3e}  max|db|/|b|={np.max(np.abs(db) / np.abs(b)):.3e}")


def demo_kappa_sweep() -> None:
    print("[4] FP32 GEPP on 100x100 SPD with prescribed kappa (seed 0), + 10 steps FP64-residual IR")
    rng = np.random.default_rng(0)
    n = 100
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    for kappa in (1e1, 1e3, 1e5, 1e7, 1e9):
        s = np.logspace(0, -np.log10(kappa), n)
        a = (q * s) @ q.T
        x = rng.standard_normal(n)
        b = a @ x
        factors = sl.lu_factor(a.astype(F32))
        xh = sl.lu_solve(factors, b.astype(F32)).astype(np.float64)
        back = oettli_prager(a, xh, b)
        fwd = rel(xh, x)
        xi = xh.copy()
        for _ in range(10):
            d = sl.lu_solve(factors, (b - a @ xi).astype(F32)).astype(np.float64)
            xi = xi + d
        k = np.linalg.cond(a)
        print(
            f"    kappa={k:.1e} backward={back:.1e} forward={fwd:.1e} "
            f"kappa*u32={k * U32:.1e} after_10_IR={rel(xi, x):.1e}"
        )


def demo_growth() -> None:
    print("[5] static (no-pivot) vs partial-pivot FP32 LU, median of 20 random x_true (seed 1)")
    rng = np.random.default_rng(1)

    def run(name: str, a: np.ndarray) -> None:
        rows = []
        lower, upper = lu_nopivot_fp32(a)
        factors = sl.lu_factor(a.astype(F32))
        for _ in range(20):
            x = rng.uniform(0.5, 1.5, len(a))
            b = a @ x
            xs = solve_lu_fp32(lower, upper, b)
            xp = sl.lu_solve(factors, b.astype(F32)).astype(np.float64)
            rows.append((oettli_prager(a, xs, b), rel(xs, x), oettli_prager(a, xp, b), rel(xp, x)))
        med = np.median(np.array(rows), axis=0)
        g = growth(a, lower, upper)
        print(
            f"    {name:12s} kappa={np.linalg.cond(a):5.1f} pivots={np.diag(upper).astype(float).round(6).tolist()} "
            f"g={g:.0e} g*u={g * U32:.0e} | static back={med[0]:.0e} fwd={med[1]:.0e} "
            f"| pivot back={med[2]:.0e} fwd={med[3]:.0e}"
        )

    for eps in (1e-2, 1e-4, 1e-6):
        run(f"2x2 e={eps:.0e}", np.array([[eps, 1.0], [1.0, 1.0]]))
    for d in (1e-1, 1e-3, 1e-5):
        run(f"3x3 d={d:.0e}", np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + d, 1.0], [0.0, 1.0, 1.0]]))


def main() -> None:
    demo_roundoff()
    demo_condition_2x2()
    demo_oettli_prager()
    demo_kappa_sweep()
    demo_growth()


if __name__ == "__main__":
    main()
