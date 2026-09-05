#!/usr/bin/env python3
"""Run a compact, reproducible SystemC architecture experiment sweep."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-sim", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=("fp64", "fixed", "both"), default="both")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a small representative subset instead of the full Cartesian sweep",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="run the full Cartesian product (very large); default is one factor at a time",
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="run the focused tile-BFP precision/rescue sweep",
    )
    return parser.parse_args()


def configurations(
    base: dict,
    quick: bool,
    cartesian: bool,
    precision: bool,
) -> list[tuple[str, dict]]:
    axes = {
        "buffer_count": [1, 2, 4],
        "ddr_bytes_per_cycle": [16, 32, 64],
        "ddr_base_latency": [10, 20, 50],
        "tile_size": [8, 16, 32],
        "panel_startup": [4, 8, 16],
        "panel_units": [1, 2],
        "trsm_startup": [3, 6, 12],
        "trsm_units": [1, 2],
        "gemm_startup": [5, 10, 20],
        "gemm_units": [1, 2, 4],
        "q_use_bits": [24, 26, 28],
        "frac_bits": [20, 24, 26, 28],
        "accumulator_bits": [48, 64],
        "workspace_guard_bits": [0, 4, 8],
        "fixed_pivot_rel_tol": [1e-7, 1e-5, 1e-3],
        "ir_max_iters": [0, 10, 50],
        "scheduler_policy": ["serial", "resource_aware"],
    }
    if precision:
        selected = [
            ("baseline", {}),
            ("frac_bits-22", {"frac_bits": 22}),
            ("frac_bits-24", {"frac_bits": 24}),
            ("frac_bits-26", {"frac_bits": 26}),
            ("frac_bits-28", {"frac_bits": 28}),
            ("guard_bits-16", {"workspace_guard_bits": 16}),
            ("guard_bits-24", {"workspace_guard_bits": 24}),
            (
                "frac24_guard16",
                {"frac_bits": 24, "workspace_guard_bits": 16},
            ),
            (
                "frac24_guard24",
                {"frac_bits": 24, "workspace_guard_bits": 24},
            ),
            ("factor_tol-5e-7", {"fixed_factor_rel_tol": 5e-7}),
            ("factor_tol-1e-6", {"fixed_factor_rel_tol": 1e-6}),
        ]
    elif quick:
        selected = [
            ("baseline", {}),
            ("buffer_1", {"buffer_count": 1}),
            ("buffer_4", {"buffer_count": 4}),
            ("ddr_bw_16", {"ddr_bytes_per_cycle": 16}),
            ("ddr_bw_64", {"ddr_bytes_per_cycle": 64}),
            ("tile_8", {"tile_size": 8}),
            ("tile_32", {"tile_size": 32}),
            ("serial", {"scheduler_policy": "serial"}),
        ]
    elif cartesian:
        import itertools

        keys = list(axes)
        selected = [
            (
                "_".join(f"{key}-{value}" for key, value in zip(keys, values)),
                dict(zip(keys, values)),
            )
            for values in itertools.product(*(axes[key] for key in keys))
        ]
    else:
        selected = [("baseline", {})]
        for key, values in axes.items():
            for value in values:
                if base.get(key) == value:
                    continue
                selected.append((f"{key}-{value}", {key: value}))
    output: list[tuple[str, dict]] = []
    for name, overrides in selected:
        config = dict(base)
        config.update(overrides)
        output.append((name, config))
    return output


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.base_config.read_text())
    rows: list[dict] = []
    for index, (name, config) in enumerate(
        configurations(
            base, args.quick, args.cartesian, args.precision
        )
    ):
        config["seed"] = args.seed
        run_dir = args.out / f"{index:04d}_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "sim_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        process = subprocess.run(
            [
                str(args.system_sim),
                "--artifact",
                str(args.artifact),
                "--config",
                str(config_path),
                "--mode",
                args.mode,
                "--out",
                str(run_dir),
                "--seed",
                str(args.seed),
            ],
            check=False,
        )
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows.append(
            {
                "name": name,
                "return_code": process.returncode,
                "status": summary.get("status", "launch_failure"),
                "total_cycles": summary.get("cycles", {}).get("total"),
                "factor_cycles": summary.get("cycles", {}).get("factorization"),
                "read_cycles": summary.get("memory", {}).get("read_cycles"),
                "write_cycles": summary.get("memory", {}).get("write_cycles"),
                "fp64_residual": summary.get("solve", {})
                .get("fp64", {})
                .get("relative_residual"),
                "fixed_residual": summary.get("solve", {})
                .get("fixed", {})
                .get("relative_residual"),
                "fixed_scaled_residual": summary.get("solve", {})
                .get("fixed", {})
                .get("scaled_relative_residual"),
                "fixed_solution_error": summary.get("solve", {})
                .get("fixed", {})
                .get("relative_solution_error"),
                "ir_iterations": summary.get("solve", {})
                .get("fixed", {})
                .get("refinement_iterations"),
                "precision_rescue_nodes": summary.get("stability", {})
                .get("precision_rescue_nodes"),
                **config,
            }
        )

    (args.out / "sweep_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (args.out / "sweep_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
