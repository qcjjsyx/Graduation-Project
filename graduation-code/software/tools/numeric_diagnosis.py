"""Numerical diagnosis of a compiled artifact and (optionally) a device run.

Answers three separate questions that a single "checker PASS/FAIL" mixes up:

1. How hard is the problem?  2-norm condition number of the original A, of the
   matrix the device actually factors (D_r*A; row permutations do not change
   it) and of two stronger row+column diagonal scalings, plus the smallest
   singular values (how many near-null directions exist).
2. Did the device factor each front well?  Per-node backward error
   ||F - L*U - S||_max / ||F||_max, where F is the node's front assembled from
   the initial image plus the device's OWN child updates, L/U/S are the
   device outputs.  This isolates each node from errors made below it.
   Also reports the smallest static pivot, the original diagonal entry it came
   from (cancellation), and max|L| (element growth).
3. Is the result usable for Ax=b?  Assembles the global FP32 factors from the
   device image, solves on the host, and runs FP64 iterative refinement. When
   the device itself produced a solution (factorization_and_solve artifact and
   a non-zero solution_data region), the device solution is reported as well:
   its residual / backward error / solution error and its deviation from the
   host solving the same system with the same device factors.

Host-only analysis: it never feeds anything back into the device image.
Dense SVD / dense solves are used, so it is meant for n up to a few thousand.

Usage (from graduation-code/software):

    python tools/numeric_diagnosis.py ART_DIR                 # 1 only
    python tools/numeric_diagnosis.py ART_DIR REPORT_DIR      # 1, 2, 3
    python tools/numeric_diagnosis.py ART_DIR REPORT_DIR --json out.json

ART_DIR is the compiler output (manifest.json, memory_image.bin and the FP64
reference files); REPORT_DIR is the tree_factor_device --out directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

FP32_UNIT_ROUNDOFF = 2.0**-24  # ~6.0e-8


class Artifact:
    def __init__(self, art_dir: Path, report_dir: Path | None) -> None:
        self.art = art_dir
        self.manifest = json.loads((art_dir / "manifest.json").read_text(encoding="utf-8"))
        self.regions = self.manifest["memory_image"]["regions"]
        self.n = int(self.manifest["matrix"]["n"])
        self.image0 = np.fromfile(art_dir / "memory_image.bin", dtype=np.uint8)
        self.image1 = (
            np.fromfile(report_dir / "final_memory_image.bin", dtype=np.uint8)
            if report_dir is not None
            else None
        )

    def region(self, image: np.ndarray, name: str, rows: int, cols: int) -> np.ndarray:
        offset = self.regions[name]["offset"]
        raw = image[offset : offset + rows * cols * 4].view("<f4")
        return raw.reshape(rows, cols).astype(np.float64)

    def device_solution(self) -> np.ndarray | None:
        """x_hat the device wrote to solution_data (None for factorization-only
        artifacts or an all-zero region: the region doubles as the solve
        accumulator, so an untouched image is not a device solution)."""

        if self.image1 is None:
            return None
        scope = self.manifest["compiler"].get("operation_scope", "factorization_and_solve")
        if scope != "factorization_and_solve":
            return None
        offset = self.regions["solution_data"]["offset"]
        raw = self.image1[offset : offset + self.n * 4].view("<f4").astype(np.float64)
        if not np.any(raw):
            return None
        return raw

    def original_system(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        a = np.fromfile(self.art / "original_matrix_f64.bin", "<f8").reshape(n, n)
        b = np.fromfile(self.art / "original_rhs_f64.bin", "<f8")
        return a, b

    def row_scale(self) -> np.ndarray:
        return np.exp2(np.fromfile(self.art / "row_scale_e.bin", "<i2").astype(np.float64))

    def transformed_system(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A_hat = Q^T P_match D_r A Q, b_hat, and ordering permutation q."""
        a, b = self.original_system()
        d = self.row_scale()
        transform = self.manifest["preprocess"]["transform"]
        rmap = np.asarray(transform["rhs_row_map"])
        q = np.asarray(transform["ordering_perm"])
        a_hat = (a * d[:, None])[rmap][:, q]
        b_hat = (b * d)[rmap]
        return a_hat, b_hat, q


def _cond_and_sigma(x: np.ndarray) -> tuple[float, np.ndarray]:
    s = np.linalg.svd(x, compute_uv=False)
    return float(s[0] / s[-1]), s / s[0]


def _ruiz(a: np.ndarray, iterations: int = 50) -> np.ndarray:
    b = a.copy()
    for _ in range(iterations):
        b = b / np.sqrt(np.abs(b).max(axis=1))[:, None]
        b = b / np.sqrt(np.abs(b).max(axis=0))[None, :]
    return b


def conditioning(art: Artifact) -> dict:
    a, _ = art.original_system()
    da = a * art.row_scale()[:, None]
    rowcol = a / np.abs(a).max(axis=1)[:, None]
    rowcol = rowcol / np.abs(rowcol).max(axis=0)[None, :]
    k_a, _ = _cond_and_sigma(a)
    k_da, rel = _cond_and_sigma(da)
    k_rc, _ = _cond_and_sigma(rowcol)
    k_ruiz, _ = _cond_and_sigma(_ruiz(a))
    row_max = np.abs(a).max(axis=1)
    return {
        "n": art.n,
        "nnz": int(np.count_nonzero(a)),
        "row_max_range": [float(row_max.min()), float(row_max.max())],
        "cond2_original": k_a,
        "cond2_pipeline_row_scaled": k_da,
        "cond2_row_col_max_scaled": k_rc,
        "cond2_ruiz_row_col": k_ruiz,
        "cond2_times_fp32_u": k_da * FP32_UNIT_ROUNDOFF,
        "smallest_relative_singular_values": [float(v) for v in rel[-6:][::-1]],
        "count_rel_sigma_below_1e-6": int(np.sum(rel < 1e-6)),
        "count_rel_sigma_below_fp32_u": int(np.sum(rel < FP32_UNIT_ROUNDOFF)),
    }


def node_diagnosis(art: Artifact) -> list[dict]:
    assert art.image1 is not None
    nodes = art.manifest["nodes"]
    results = []
    for key in sorted(nodes, key=int):
        node = nodes[key]
        n, p, u = node["total_dim"], node["pivot_dim"], node["update_dim"]
        front = art.region(art.image0, f"node.{key}.front", n, n)
        position = {g: i for i, g in enumerate(node["front_indices"])}
        for child_key, child in nodes.items():
            if child["parent_id"] != int(key):
                continue
            cu = child["update_dim"]
            idx = [position[g] for g in child["front_indices"][child["pivot_dim"] :]]
            front[np.ix_(idx, idx)] += art.region(art.image1, f"node.{child_key}.update", cu, cu)
        l_factor = art.region(art.image1, f"node.{key}.factor_l", n, p)
        u_factor = art.region(art.image1, f"node.{key}.factor_u", p, n)
        schur = np.zeros((n, n))
        if u:
            schur[p:, p:] = art.region(art.image1, f"node.{key}.update", u, u)
        scale = np.abs(front).max()
        backward = np.abs(front - l_factor @ u_factor - schur).max() / scale
        # FP64 static elimination of the same input: where do small pivots come from?
        work = front.copy()
        pivots = np.empty(p)
        for k in range(p):
            pivots[k] = work[k, k]
            work[k + 1 :, k] /= work[k, k]
            work[k + 1 :, k + 1 :] -= np.outer(work[k + 1 :, k], work[k, k + 1 :])
        k_min = int(np.argmin(np.abs(pivots)))
        results.append(
            {
                "node": int(key),
                "n": n,
                "p": p,
                "backward_error": float(backward),
                "max_abs_front": float(scale),
                "min_abs_pivot": float(abs(pivots[k_min])),
                "pivot_ratio": float(abs(pivots[k_min]) / scale),
                "original_diagonal_at_min_pivot": float(abs(front[k_min, k_min])),
                "max_abs_l": float(np.abs(l_factor).max()),
            }
        )
    return results


def solve_diagnosis(art: Artifact, refinement_steps: int = 3) -> dict:
    assert art.image1 is not None
    n = art.n
    rows_l, cols_l, vals_l, rows_u, cols_u, vals_u = [], [], [], [], [], []
    for key, node in art.manifest["nodes"].items():
        total, pivot = node["total_dim"], node["pivot_dim"]
        variables = np.asarray(node["front_indices"])
        l_factor = art.region(art.image1, f"node.{key}.factor_l", total, pivot)
        u_factor = art.region(art.image1, f"node.{key}.factor_u", pivot, total)
        for j in range(pivot):
            tail = variables[j:]
            rows_l += list(tail); cols_l += [variables[j]] * len(tail); vals_l += list(l_factor[j:, j])
            rows_u += [variables[j]] * len(tail); cols_u += list(tail); vals_u += list(u_factor[j, j:])
    lower = sp.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n, n))
    upper = sp.csr_matrix((vals_u, (rows_u, cols_u)), shape=(n, n))
    a, b = art.original_system()
    a_hat, b_hat, q = art.transformed_system()
    x_true_path = art.art / "x_true_f64.bin"
    x_true = np.fromfile(x_true_path, "<f8") if x_true_path.exists() else None

    def lu_solve(r: np.ndarray) -> np.ndarray:
        y = spla.spsolve_triangular(lower, r, lower=True)
        return spla.spsolve_triangular(upper, y, lower=False)

    def metrics(x_hat: np.ndarray) -> dict:
        x = np.empty(n)
        x[q] = x_hat
        residual = a @ x - b
        out = {
            "relative_residual": float(
                np.linalg.norm(residual)
                / (np.linalg.norm(a, "fro") * np.linalg.norm(x) + np.linalg.norm(b))
            ),
            "componentwise_backward_error": float(
                np.max(np.abs(residual) / (np.abs(a) @ np.abs(x) + np.abs(b)))
            ),
        }
        if x_true is not None:
            out["relative_forward_error"] = float(
                np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            )
        return out

    steps = []
    x_hat = lu_solve(b_hat)
    steps.append({"step": "fp32_factors_single_solve", **metrics(x_hat)})
    device = art.device_solution()
    if device is not None:
        # [3] device comparison: the solution the device actually produced vs.
        # the host solving the same system with the same (device) factors.
        steps.append(
            {
                "step": "device_solution_data",
                "relative_to_host_device_factor_solve": float(
                    np.linalg.norm(device - x_hat)
                    / (np.linalg.norm(x_hat) if np.linalg.norm(x_hat) else 1.0)
                ),
                **metrics(device),
            }
        )
    for i in range(1, refinement_steps + 1):
        x_hat = x_hat + lu_solve(b_hat - a_hat @ x_hat)
        steps.append({"step": f"fp64_refinement_{i}", **metrics(x_hat)})
    reference = np.linalg.solve(a, b)
    steps.append({"step": "fp64_dense_reference", **metrics(reference[q])})
    product = (lower @ upper).toarray()
    return {
        "global_backward_error": float(np.abs(a_hat - product).max() / np.abs(a_hat).max()),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("art_dir", type=Path)
    parser.add_argument("report_dir", type=Path, nargs="?")
    parser.add_argument("--json", type=Path, help="also write all results as JSON")
    parser.add_argument("--refinement-steps", type=int, default=3)
    args = parser.parse_args()
    art = Artifact(args.art_dir, args.report_dir)

    result: dict = {"artifact": str(args.art_dir), "conditioning": conditioning(art)}
    c = result["conditioning"]
    print(f"[1] conditioning  n={c['n']} nnz={c['nnz']}")
    for label, field in [
        ("original A", "cond2_original"),
        ("D_r*A (factored by device)", "cond2_pipeline_row_scaled"),
        ("row+col max scaling", "cond2_row_col_max_scaled"),
        ("Ruiz row+col scaling", "cond2_ruiz_row_col"),
    ]:
        print(f"    cond2 {label:28s} {c[field]:.2e}")
    print(f"    cond2(D_r*A) * u_fp32 = {c['cond2_times_fp32_u']:.2e}  (>= 1: FP32 alone cannot resolve x)")
    print("    smallest sigma/sigma_max: " + ", ".join(f"{v:.1e}" for v in c["smallest_relative_singular_values"]))
    print(f"    near-null directions: {c['count_rel_sigma_below_fp32_u']} below FP32 u, "
          f"{c['count_rel_sigma_below_1e-6']} below 1e-6")

    if args.report_dir is not None:
        nodes = node_diagnosis(art)
        result["nodes"] = nodes
        worst = max(nodes, key=lambda r: r["backward_error"])
        smallest = min(nodes, key=lambda r: r["pivot_ratio"])
        print(f"[2] per-node backward error: max {worst['backward_error']:.2e} (node {worst['node']}), "
              f"median {np.median([r['backward_error'] for r in nodes]):.2e}")
        print(f"    smallest pivot ratio {smallest['pivot_ratio']:.2e} at node {smallest['node']} "
              f"(pivot {smallest['min_abs_pivot']:.2e}, original diagonal "
              f"{smallest['original_diagonal_at_min_pivot']:.2e}); "
              f"max|L| over nodes {max(r['max_abs_l'] for r in nodes):.2e}")
        solve = solve_diagnosis(art, args.refinement_steps)
        result["solve"] = solve
        print(f"[3] global ||A_hat - LU|| / ||A_hat|| = {solve['global_backward_error']:.2e}")
        for step in solve["steps"]:
            fwd = step.get("relative_forward_error")
            host_gap = step.get("relative_to_host_device_factor_solve")
            print(f"    {step['step']:28s} residual {step['relative_residual']:.2e}  "
                  f"bwd {step['componentwise_backward_error']:.2e}"
                  + (f"  x err {fwd:.2e}" if fwd is not None else "")
                  + (f"  vs host (device factors) {host_gap:.2e}" if host_gap is not None else ""))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
