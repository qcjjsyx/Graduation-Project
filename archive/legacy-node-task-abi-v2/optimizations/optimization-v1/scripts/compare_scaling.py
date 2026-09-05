#!/usr/bin/env python3
"""Generate and compare the independent scaling optimization variants."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MATRICES = {
    "256": ("256X256JJ.mat", "256fuv.mat"),
    "576": ("576X576JJ.mat", "576fuv.mat"),
    "1024": ("1024X1024JJ.mat", "1024fuv.mat"),
}

VARIANTS = {
    "B0": ("pow2-row", 4),
    "B1": ("pow2-row-column", 4),
    "B2": ("pow2-ruiz", 4),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run B0/B1/B2 scaling A/B experiments",
    )
    parser.add_argument(
        "--software-dir",
        type=Path,
        default=root / "software",
    )
    parser.add_argument(
        "--system-sim",
        type=Path,
        default=root / "systemc" / "build" / "system_sim",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "systemc" / "config" / "default.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--matrices",
        nargs="+",
        choices=tuple(MATRICES),
        default=list(MATRICES),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    software_dir = args.software_dir.resolve()
    system_sim = args.system_sim.resolve()
    config = args.config.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if not system_sim.exists():
        raise SystemExit(f"system_sim does not exist: {system_sim}")

    rows: list[dict] = []
    for matrix_name in args.matrices:
        matrix_file, rhs_file = MATRICES[matrix_name]
        for variant in args.variants:
            mode, iterations = VARIANTS[variant]
            run_dir = args.out / f"{matrix_name}_{variant}"
            artifact_dir = run_dir / "artifact"
            result_dir = run_dir / "result"
            run_dir.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.main",
                    "-mtx",
                    str(software_dir / "example" / matrix_file),
                    "--rhs",
                    str(software_dir / "example" / rhs_file),
                    "--ordering",
                    "amd",
                    "--equilibrate",
                    mode,
                    "--equilibration-iterations",
                    str(iterations),
                    "--out",
                    str(artifact_dir),
                ],
                cwd=software_dir,
                check=True,
            )
            process = subprocess.run(
                [
                    str(system_sim),
                    "--artifact",
                    str(artifact_dir / "manifest.json"),
                    "--config",
                    str(config),
                    "--mode",
                    "fixed",
                    "--out",
                    str(result_dir),
                    "--seed",
                    str(args.seed),
                ],
                check=False,
            )
            summary_path = result_dir / "summary.json"
            summary = (
                json.loads(summary_path.read_text())
                if summary_path.exists()
                else {}
            )
            fixed = summary.get("solve", {}).get("fixed", {})
            rows.append(
                {
                    "matrix": matrix_name,
                    "variant": variant,
                    "equilibration": mode,
                    "return_code": process.returncode,
                    "status": summary.get("status", "launch_failure"),
                    "relative_residual": fixed.get("relative_residual"),
                    "initial_relative_residual":
                        fixed.get("initial_relative_residual"),
                    "scaled_relative_residual":
                        fixed.get("scaled_relative_residual"),
                    "componentwise_backward_error":
                        fixed.get("componentwise_backward_error"),
                    "relative_solution_error":
                        fixed.get("relative_solution_error"),
                    "refinement_iterations":
                        fixed.get("refinement_iterations"),
                    "refinement_stop_reason":
                        fixed.get("refinement_stop_reason"),
                    "precision_rescue_nodes":
                        summary.get("stability", {})
                        .get("precision_rescue_nodes"),
                    "assembly_drop_count":
                        summary.get("stability", {})
                        .get("assembly_drop_count"),
                    "factor_cycles":
                        summary.get("cycles", {}).get("factorization"),
                    "solve_cycles": summary.get("cycles", {}).get("solve"),
                    "total_cycles": summary.get("cycles", {}).get("total"),
                }
            )

    baselines = {
        row["matrix"]: row
        for row in rows
        if row["variant"] == "B0"
    }
    for row in rows:
        baseline = baselines.get(row["matrix"])
        row["residual_improvement_vs_B0"] = improvement(
            baseline, row, "relative_residual"
        )
        row["cycle_improvement_vs_B0"] = improvement(
            baseline, row, "total_cycles"
        )

    (args.out / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    fields = list(rows[0]) if rows else []
    with (args.out / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["status"] == "ok" for row in rows) else 2


def improvement(
    baseline: dict | None,
    candidate: dict,
    field: str,
) -> float | None:
    if baseline is None:
        return None
    before = baseline.get(field)
    after = candidate.get(field)
    if not isinstance(before, (int, float)) or not isinstance(
        after, (int, float)
    ) or before == 0:
        return None
    return (before - after) / abs(before)


if __name__ == "__main__":
    raise SystemExit(main())
