"""FP32 device precision-boundary experiment.

    python -m tools.precision_boundary run --plan smoke --out /tmp/pb [--jobs 4]
    python -m tools.precision_boundary summarize --out /tmp/pb
    python -m tools.precision_boundary plot --csv /tmp/pb/results.csv --out FIG_DIR

Plans (experiments/precision_boundary/实验报告.md §2、§5):
  smoke  random n=128: F1 at 3 kappas + one F2 case (about a minute)
  main   jj256 + random n=256 patterns, 3 seeds:
         F1 kappa 1e1..1e12 every half decade (kappa only, growth ~1);
         F2 gamma {0.3, 1, 3} (kappa collapses to ~1e1-1e2: growth only);
         F3 kappa 1e2..1e12 every decade x gamma {0.1, 0.3, 1} (kappa + measured growth)
  large  jj576 / jj1024 patterns: F1 kappa 1e2..1e12 every 2 decades + F3 gamma 0.3 at 1e4, 1e8
  real   the three repository matrices as anchors

Results are cached per case (<out>/<case_id>/result.json); rerunning a plan
only runs missing cases unless --force.
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the cases run in parallel processes.
for _var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from .analysis import U64  # noqa: E402
from .matrices import MatrixSpec  # noqa: E402
from .runner import run_case  # noqa: E402

RANDOM_DENSITY = {128: 0.03, 256: 0.015}


def _decades(lo: float, hi: float, step: float) -> list[float]:
    count = int(round((hi - lo) / step)) + 1
    return [float(10 ** (lo + i * step)) for i in range(count)]


def plan(name: str) -> list[MatrixSpec]:
    specs: list[MatrixSpec] = []
    if name == "smoke":
        for kappa in (1e2, 1e6, 1e10):
            specs.append(MatrixSpec("F1", "random", 128, RANDOM_DENSITY[128], kappa, 0.0, 0))
        specs.append(MatrixSpec("F2", "random", 128, RANDOM_DENSITY[128], 1e4, 3.0, 0))
        specs.append(MatrixSpec("F3", "random", 128, RANDOM_DENSITY[128], 1e8, 0.3, 0))
    elif name == "main":
        patterns = [("jj256", None, None), ("random", 256, RANDOM_DENSITY[256])]
        for seed in range(3):
            for pattern, n, density in patterns:
                for kappa in _decades(1, 12, 0.5):
                    specs.append(MatrixSpec("F1", pattern, n, density, kappa, 0.0, seed))
                for gamma in (0.3, 1.0, 3.0):
                    specs.append(MatrixSpec("F2", pattern, n, density, 1e4, gamma, seed))
                for kappa in _decades(2, 12, 1):
                    for gamma in (0.1, 0.3, 1.0):
                        specs.append(MatrixSpec("F3", pattern, n, density, kappa, gamma, seed))
    elif name == "large":
        for pattern in ("jj576", "jj1024"):
            for kappa in _decades(2, 12, 2):
                specs.append(MatrixSpec("F1", pattern, None, None, kappa, 0.0, 0))
            for kappa in (1e4, 1e8):
                specs.append(MatrixSpec("F3", pattern, None, None, kappa, 0.3, 0))
    elif name == "real":
        specs = [MatrixSpec("real", p) for p in ("jj256", "jj576", "jj1024")]
    else:
        raise SystemExit(f"unknown plan {name!r}")
    return specs


def _run(args: argparse.Namespace) -> None:
    specs = []
    for name in args.plan:
        specs += plan(name)
    if args.limit:
        specs = specs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(specs)} cases -> {args.out} (jobs={args.jobs})", flush=True)
    kwargs = dict(out_root=args.out, build=args.build, rhs_seed=args.rhs_seed,
                  max_cycles=args.max_cycles, ir_steps=args.ir_steps, force=args.force,
                  keep_artifacts=args.keep_artifacts)
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, spec, **kwargs): spec for spec in specs}
        for future in as_completed(futures):
            spec = futures[future]
            done += 1
            try:
                result = future.result()
                status = "ERROR " + result["error"] if "error" in result else _one_line(result)
            except Exception as error:  # worker crash
                status = f"CRASH {error}"
            print(f"[{done}/{len(specs)}] {spec.case_id}: {status}", flush=True)


def _ir_converged(history: list[dict], n: int, kappa: float) -> int | None:
    """First step meeting the IR success criterion (§4 of the experiment doc)."""
    for entry in history:
        if not entry.get("finite"):
            return None
        if entry["backward_error"] <= 10 * n * U64 and entry.get("forward_error", np.inf) <= 10 * kappa * U64:
            return entry["step"]
    return None


def flatten(result: dict, one_shot_tol: float) -> dict:
    spec = result["spec"]
    row = {
        "case_id": result["case_id"], "family": spec["family"], "pattern": spec["pattern"],
        "kappa_target": spec["kappa"], "gamma": spec["gamma"], "seed": spec["seed"],
        "error": result.get("error", ""),
    }
    comp = result.get("compile", {})
    row.update(compile_ok=comp.get("ok"), nodes=comp.get("nodes"), max_front=comp.get("max_front"),
               device_buffer_bytes=comp.get("device_buffer_bytes"))
    dev = result.get("device_run", {})
    row.update(device_exit=dev.get("exit_code"), control_flow_pass=dev.get("control_flow_pass"),
               numeric_status_pass=dev.get("numeric_status_pass"), cycles=dev.get("cycles"),
               device_seconds=dev.get("seconds"))
    sc = result.get("solve_check", {})
    row.update(solve_pass=sc.get("solve_pass"), eta_f=sc.get("eta_forward"), eta_b=sc.get("eta_backward"))
    fc = result.get("factor_check", {})
    row.update(factor_numeric_pass=fc.get("numeric_pass"))
    ana = result.get("analysis", {})
    cond = ana.get("conditioning", {})
    for key in ("n", "kappa2_A", "kappa2_DrA", "kappa1_DrA", "kappa1_DrA_estimate", "skeel_DrA_xtrue",
                "count_rel_sigma_below_u32"):
        row[key] = cond.get(key)
    ref = ana.get("references", {})
    row["fp64_forward"] = ref.get("fp64_dense", {}).get("forward_error")
    row["gepp_growth"] = ref.get("fp32_gepp_growth")
    gepp = ref.get("fp32_gepp_ir", [])
    if gepp:
        row["gepp_forward_0"] = gepp[0].get("forward_error")
        row["gepp_ir_final_forward"] = gepp[-1].get("forward_error")
    device = ana.get("device")
    n, kappa = cond.get("n"), cond.get("kappa2_DrA")
    if device:
        for key in ("static_growth", "global_lu_backward", "node_backward_max", "node_backward_median",
                    "node_min_pivot_ratio", "node_max_abs_l", "device_vs_host_fp32_solve"):
            row[key] = device.get(key)
        ir = device["ir"]
        row["device_backward"] = ir[0].get("backward_error")
        row["device_forward"] = ir[0].get("forward_error")
        row["ir_final_backward"] = ir[-1].get("backward_error")
        row["ir_final_forward"] = ir[-1].get("forward_error")
        finite = [e.get("forward_error") for e in ir if e.get("finite")]
        row["ir_best_forward"] = min(finite) if finite else None
        row["ir_converged_step"] = _ir_converged(ir, n, kappa) if n and kappa else None
        row["one_shot_ok"] = (row["device_forward"] is not None and row["device_forward"] <= one_shot_tol)
        row["ir_ok"] = row["ir_converged_step"] is not None
    if gepp and n and kappa:
        step = _ir_converged(gepp, n, kappa)
        row["gepp_ir_converged_step"] = step
        row["gepp_ir_ok"] = step is not None
    return row


def _one_line(result: dict) -> str:
    row = flatten(result, 1e-3)
    if not row.get("compile_ok"):
        return f"compile failed: {result.get('compile', {}).get('error', '')[:80]}"
    def f(key: str) -> str:
        v = row.get(key)
        return f"{v:.1e}" if isinstance(v, float) else str(v)
    return (f"k2(DrA)={f('kappa2_DrA')} g={f('static_growth')} eta_f={f('eta_f')} "
            f"fwd={f('device_forward')} IR={f('ir_final_forward')}@{row.get('ir_converged_step')} "
            f"dev={row.get('device_seconds')}s")


def _summarize(args: argparse.Namespace) -> None:
    rows = []
    for path in sorted(args.out.glob("*/result.json")):
        rows.append(flatten(json.loads(path.read_text()), args.one_shot_tol))
    if not rows:
        raise SystemExit(f"no results under {args.out}")
    keys: list[str] = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    csv_path = args.out / "results.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} cases -> {csv_path}")
    header = f"{'case':34s} {'k2(DrA)':>8s} {'g_static':>8s} {'eta_f':>8s} {'eta_b':>8s} {'fwd':>8s} {'IR fwd':>8s} {'IRstep':>6s} {'GEPP fwd':>8s}"
    print(header)
    for row in rows:
        def f(key: str) -> str:
            v = row.get(key)
            return f"{v:8.1e}" if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None else f"{str(v):>8s}"
        print(f"{row['case_id'][:34]:34s} {f('kappa2_DrA')} {f('static_growth')} {f('eta_f')} {f('eta_b')} "
              f"{f('device_forward')} {f('ir_final_forward')} {str(row.get('ir_converged_step')):>6s} {f('gepp_forward_0')}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tools.precision_boundary",
                                     description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--plan", nargs="+", required=True, choices=["smoke", "main", "large", "real"])
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--build", type=Path, default=Path("/tmp/gp-rel"))
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--rhs-seed", type=int, default=1)
    run.add_argument("--max-cycles", type=int, default=40_000_000)
    run.add_argument("--ir-steps", type=int, default=10)
    run.add_argument("--limit", type=int, default=0, help="run only the first N cases")
    run.add_argument("--force", action="store_true")
    run.add_argument("--keep-artifacts", action="store_true")
    summ = sub.add_parser("summarize")
    summ.add_argument("--out", type=Path, required=True)
    summ.add_argument("--one-shot-tol", type=float, default=1e-3)
    plot = sub.add_parser("plot")
    plot.add_argument("--csv", type=Path, required=True)
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plot":
        from .plot import make_figures

        for path in make_figures(args.csv, args.out):
            print(path)
        return
    if args.command == "run":
        if not (args.build / "tree_factor_device").exists():
            raise SystemExit(f"{args.build}/tree_factor_device not found (build Release first)")
        _run(args)
    else:
        _summarize(args)


if __name__ == "__main__":
    sys.exit(main())
