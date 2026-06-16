from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from hardware import (
    HardwareConfig,
    HardwareSystem,
    TILE_SIZE,
    generate_test_matrix,
    summarize_ops,
)


DEFAULT_MODES = ("stable", "pivot_stress", "random", "large_value")
DEFAULT_SIZES = (128, 256, 512)
DEFAULT_SEEDS = (1, 2, 3)


@dataclass
class RunSummary:
    case_id: str
    mode: str
    n: int
    seed: int
    status: str
    error: str
    elapsed_s: float
    e_node: int | None
    swaps: int | None
    lu_abs_error: float
    lu_rel_error: float
    direct_residual: float
    direct_rel_x_error: float
    ir_initial_residual: float
    ir_final_residual: float
    ir_rel_x_error: float
    ir_converged: bool
    ir_iterations: int
    residual_improvement: float
    total_cycles: int | None
    panel_fact_ops: int
    trsm_u_ops: int
    trsm_l_ops: int
    gemm_schur_ops: int
    align_drop_count: int | None
    align_shift_max: int | None
    asm_overflow_count: int | None
    requant_sat_count: int | None
    maxabs_acc: int | None


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return [int(value) for value in values]


def parse_str_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def sci(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(value_f):
        return "-"
    return f"{value_f:.{digits}e}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not math.isfinite(denominator):
        return math.nan
    return numerator / denominator


def run_case(
    n: int,
    mode: str,
    seed: int,
    config: HardwareConfig,
    ir_iters: int,
    tol: float,
) -> tuple[RunSummary, list[dict[str, float | int | str]]]:
    case_id = f"{mode}_n{n}_seed{seed}"
    start = perf_counter()
    try:
        rng = np.random.default_rng(seed)
        a = generate_test_matrix(n, mode, rng)
        x_true = rng.uniform(-10.0, 10.0, n)
        b = a @ x_true

        hw = HardwareSystem(config)
        lu = hw.factorize_dense(a)

        reconstructed = lu.l_float @ lu.u_float
        permuted_a = a[lu.row_map, :]
        lu_abs_error = float(np.linalg.norm(permuted_a - reconstructed, ord="fro"))
        lu_rel_error = safe_ratio(lu_abs_error, float(np.linalg.norm(a, ord="fro")))

        x_direct = hw.solve_quantized_lu(lu, b)
        direct_residual = float(np.linalg.norm(b - a @ x_direct))
        direct_rel_x_error = safe_ratio(
            float(np.linalg.norm(x_direct - x_true)),
            float(np.linalg.norm(x_true)),
        )

        ir = hw.iterative_refine(a, b, lu, max_iters=ir_iters, tol=tol)
        ir_final_residual = float(np.linalg.norm(b - a @ ir.x))
        ir_rel_x_error = safe_ratio(
            float(np.linalg.norm(ir.x - x_true)),
            float(np.linalg.norm(x_true)),
        )
        ir_initial_residual = (
            float(ir.residual_history[0]) if ir.residual_history else math.nan
        )
        residual_improvement = safe_ratio(direct_residual, ir_final_residual)
        op_counts = summarize_ops(lu.op_log)
        elapsed = perf_counter() - start

        history_rows: list[dict[str, float | int | str]] = []
        for iteration, residual in enumerate(ir.residual_history):
            residual_f = float(residual)
            history_rows.append(
                {
                    "case_id": case_id,
                    "mode": mode,
                    "n": n,
                    "seed": seed,
                    "iteration": iteration,
                    "residual_abs": residual_f,
                    "residual_relative_to_initial": safe_ratio(
                        residual_f, ir_initial_residual
                    ),
                }
            )

        summary = RunSummary(
            case_id=case_id,
            mode=mode,
            n=n,
            seed=seed,
            status="ok",
            error="",
            elapsed_s=elapsed,
            e_node=lu.e_node,
            swaps=lu.swap_count,
            lu_abs_error=lu_abs_error,
            lu_rel_error=lu_rel_error,
            direct_residual=direct_residual,
            direct_rel_x_error=direct_rel_x_error,
            ir_initial_residual=ir_initial_residual,
            ir_final_residual=ir_final_residual,
            ir_rel_x_error=ir_rel_x_error,
            ir_converged=ir.converged,
            ir_iterations=max(0, len(ir.residual_history) - 1),
            residual_improvement=residual_improvement,
            total_cycles=lu.total_cycles,
            panel_fact_ops=op_counts.get("PANEL_FACT", 0),
            trsm_u_ops=op_counts.get("TRSM_U", 0),
            trsm_l_ops=op_counts.get("TRSM_L", 0),
            gemm_schur_ops=op_counts.get("GEMM_SCHUR", 0),
            align_drop_count=lu.stats.align_drop_count,
            align_shift_max=lu.stats.align_shift_max,
            asm_overflow_count=lu.stats.asm_overflow_count,
            requant_sat_count=lu.stats.requant_sat_count,
            maxabs_acc=lu.stats.maxabs_acc,
        )
        return summary, history_rows

    except Exception as exc:  # Keep batch runs useful even when a stress case fails.
        elapsed = perf_counter() - start
        summary = RunSummary(
            case_id=case_id,
            mode=mode,
            n=n,
            seed=seed,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            elapsed_s=elapsed,
            e_node=None,
            swaps=None,
            lu_abs_error=math.nan,
            lu_rel_error=math.nan,
            direct_residual=math.nan,
            direct_rel_x_error=math.nan,
            ir_initial_residual=math.nan,
            ir_final_residual=math.nan,
            ir_rel_x_error=math.nan,
            ir_converged=False,
            ir_iterations=0,
            residual_improvement=math.nan,
            total_cycles=None,
            panel_fact_ops=0,
            trsm_u_ops=0,
            trsm_l_ops=0,
            gemm_schur_ops=0,
            align_drop_count=None,
            align_shift_max=None,
            asm_overflow_count=None,
            requant_sat_count=None,
            maxabs_acc=None,
        )
        return summary, []


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summaries: list[RunSummary]) -> None:
    write_csv(path, [asdict(summary) for summary in summaries])


def prepare_plotting(out_dir: Path):
    os.environ.setdefault("XDG_CACHE_HOME", str(out_dir / ".cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    (out_dir / ".cache").mkdir(parents=True, exist_ok=True)
    (out_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_residual_convergence(
    fig_dir: Path,
    histories: list[dict[str, float | int | str]],
    modes: list[str],
    plt,
) -> None:
    by_mode: dict[str, list[list[float]]] = {mode: [] for mode in modes}
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in histories:
        grouped.setdefault(str(row["case_id"]), []).append(row)

    for rows in grouped.values():
        rows = sorted(rows, key=lambda item: int(item["iteration"]))
        mode = str(rows[0]["mode"])
        curve = [
            max(float(item["residual_relative_to_initial"]), 1e-18) for item in rows
        ]
        by_mode.setdefault(mode, []).append(curve)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=180)
    for mode in modes:
        curves = by_mode.get(mode, [])
        if not curves:
            continue
        max_len = max(len(curve) for curve in curves)
        arr = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for row_idx, curve in enumerate(curves):
            arr[row_idx, : len(curve)] = curve
            if len(curve) < max_len:
                arr[row_idx, len(curve):] = curve[-1]
        x = np.arange(max_len)
        median = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax.plot(x, median, marker="o", linewidth=2.0, label=mode)
        ax.fill_between(x, q25, q75, alpha=0.12)

    ax.set_yscale("log")
    ax.set_xlabel("Iterative refinement iteration")
    ax.set_ylabel("Residual / initial residual")
    ax.set_title("Residual convergence across matrix modes")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "residual_convergence_by_mode.png")
    plt.close(fig)


def plot_final_residuals(fig_dir: Path, summaries: list[RunSummary], modes: list[str], plt) -> None:
    ok = [summary for summary in summaries if summary.status == "ok"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    for idx, mode in enumerate(modes):
        values = [
            max(summary.ir_final_residual, 1e-18)
            for summary in ok
            if summary.mode == mode and math.isfinite(summary.ir_final_residual)
        ]
        if not values:
            continue
        xs = np.full(len(values), idx, dtype=np.float64)
        jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0.0]
        ax.scatter(xs + jitter, values, s=34, alpha=0.78)
        ax.plot(
            [idx - 0.24, idx + 0.24],
            [float(np.median(values)), float(np.median(values))],
            color="black",
            linewidth=2.0,
        )
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Final residual after IR")
    ax.set_title("Final residual distribution")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "final_residual_by_mode.png")
    plt.close(fig)


def plot_op_counts(fig_dir: Path, summaries: list[RunSummary], plt) -> None:
    ok = [summary for summary in summaries if summary.status == "ok"]
    sizes = sorted({summary.n for summary in ok})
    if not sizes:
        return

    ops = ("panel_fact_ops", "trsm_u_ops", "trsm_l_ops", "gemm_schur_ops")
    labels = ("PANEL_FACT", "TRSM_U", "TRSM_L", "GEMM_SCHUR")
    data = {op: [] for op in ops}
    for n in sizes:
        subset = [summary for summary in ok if summary.n == n]
        for op in ops:
            data[op].append(float(np.mean([getattr(summary, op) for summary in subset])))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    bottom = np.zeros(len(sizes), dtype=np.float64)
    x = np.arange(len(sizes))
    for op, label in zip(ops, labels):
        values = np.asarray(data[op], dtype=np.float64)
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("Matrix dimension")
    ax.set_ylabel("Operation count")
    ax.set_title("Hardware behavior model op counts by size")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "op_counts_by_size.png")
    plt.close(fig)


def plot_quant_risks(fig_dir: Path, summaries: list[RunSummary], modes: list[str], plt) -> None:
    ok = [summary for summary in summaries if summary.status == "ok"]
    risk_fields = (
        ("align_drop_count", "Align drop"),
        ("asm_overflow_count", "Assembly overflow"),
        ("requant_sat_count", "Requant saturation"),
    )
    values = np.zeros((len(risk_fields), len(modes)), dtype=np.float64)
    for field_idx, (field, _) in enumerate(risk_fields):
        for mode_idx, mode in enumerate(modes):
            values[field_idx, mode_idx] = sum(
                float(getattr(summary, field) or 0)
                for summary in ok
                if summary.mode == mode
            )

    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    x = np.arange(len(modes))
    width = 0.24
    for idx, (_, label) in enumerate(risk_fields):
        ax.bar(x + (idx - 1) * width, values[idx], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=20, ha="right")
    ax.set_ylabel("Total events")
    ax.set_title("Quantization and assembly risk counters")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "quantization_risk_counters.png")
    plt.close(fig)


def write_report(
    out_dir: Path,
    summaries: list[RunSummary],
    histories: list[dict[str, float | int | str]],
    args: argparse.Namespace,
) -> None:
    ok = [summary for summary in summaries if summary.status == "ok"]
    failed = [summary for summary in summaries if summary.status != "ok"]
    effective_modes = list(dict.fromkeys(summary.mode for summary in summaries))
    effective_sizes = sorted({summary.n for summary in summaries})
    effective_seeds = sorted({summary.seed for summary in summaries})

    final_res = [summary.ir_final_residual for summary in ok if math.isfinite(summary.ir_final_residual)]
    direct_res = [summary.direct_residual for summary in ok if math.isfinite(summary.direct_residual)]
    lu_rel = [summary.lu_rel_error for summary in ok if math.isfinite(summary.lu_rel_error)]
    improvement = [
        summary.residual_improvement
        for summary in ok
        if math.isfinite(summary.residual_improvement)
    ]

    max_n = max(effective_sizes) if effective_sizes else 0
    representatives: list[RunSummary] = []
    for mode in effective_modes:
        candidates = [
            summary
            for summary in ok
            if summary.mode == mode and summary.n == max_n
        ]
        if candidates:
            representatives.append(sorted(candidates, key=lambda item: item.seed)[0])

    lines = [
        "# 硬件行为级仿真实验汇总",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 实验目的",
        "",
        "本脚本批量运行 `hardware.py` 中的硬件行为模型，用于展示在加入 ATU、HPU 与量化数据格式后，定点 LU、TRSM、Schur 更新和迭代求精的整体行为。默认覆盖多种矩阵模式、多个矩阵规模和多个随机种子，避免只展示单一特定矩阵。",
        "",
        "## 实验配置",
        "",
        f"- 矩阵模式：{', '.join(effective_modes)}",
        f"- 矩阵规模：{', '.join(str(n) for n in effective_sizes)}",
        f"- 随机种子：{', '.join(str(seed) for seed in effective_seeds)}",
        f"- tile size：{TILE_SIZE}",
        f"- q_use_bits：{args.q_use_bits}",
        f"- frac_bits：{args.frac_bits}",
        f"- IR 最大迭代次数：{args.ir_iters}",
        f"- IR residual tolerance：{args.tol:g}",
        "",
        "## 总体结果",
        "",
        f"- 成功用例：{len(ok)} / {len(summaries)}",
        f"- 失败用例：{len(failed)}",
        f"- 直接求解 residual 中位数：{sci(float(np.median(direct_res)) if direct_res else math.nan)}",
        f"- IR 后 residual 中位数：{sci(float(np.median(final_res)) if final_res else math.nan)}",
        f"- LU 相对误差中位数：{sci(float(np.median(lu_rel)) if lu_rel else math.nan)}",
        f"- residual 改善倍数中位数：{sci(float(np.median(improvement)) if improvement else math.nan)}",
        "",
        "## PPT 可用代表性结果",
        "",
        "| 矩阵模式 | 规模 | seed | swap 次数 | LU 相对误差 | 直接 residual | IR 后 residual | IR 收敛 | 估算周期 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for summary in representatives:
        lines.append(
            "| "
            + " | ".join(
                [
                    summary.mode,
                    str(summary.n),
                    str(summary.seed),
                    str(summary.swaps),
                    sci(summary.lu_rel_error),
                    sci(summary.direct_residual),
                    sci(summary.ir_final_residual),
                    "是" if summary.ir_converged else "否",
                    str(summary.total_cycles),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            "- `figures/residual_convergence_by_mode.png`：不同矩阵模式下 residual 相对初始 residual 的中位收敛曲线，阴影为 25%-75% 分位区间。",
            "- `figures/final_residual_by_mode.png`：不同矩阵模式下 IR 后最终 residual 分布。",
            "- `figures/op_counts_by_size.png`：不同规模下硬件行为模型的操作数量。",
            "- `figures/quantization_risk_counters.png`：量化/装配风险计数统计。",
            "",
            "## 数据文件",
            "",
            "- `run_summary.csv`：每个实验用例的指标汇总。",
            "- `residual_history.csv`：每个用例每次迭代的 residual 历史，可用于重新绘图。",
            "- `op_counts.csv`：硬件操作数量统计。",
            "- `quant_stats.csv`：量化与装配风险指标。",
        ]
    )

    if failed:
        lines.extend(["", "## 失败或异常用例", ""])
        lines.append("| case_id | error |")
        lines.append("|---|---|")
        for summary in failed:
            lines.append(f"| {summary.case_id} | {summary.error} |")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready results for hardware.py behavior model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modes",
        type=parse_str_list,
        default=list(DEFAULT_MODES),
        help="comma-separated matrix modes",
    )
    parser.add_argument(
        "--sizes",
        type=parse_int_list,
        default=list(DEFAULT_SIZES),
        help="comma-separated matrix dimensions",
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=list(DEFAULT_SEEDS),
        help="comma-separated random seeds",
    )
    parser.add_argument("--ir-iters", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--frac-bits", type=int, default=20)
    parser.add_argument("--q-use-bits", type=int, default=27)
    parser.add_argument("--no-cycle-model", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory; default is sim/results/hardware_behavior_latest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_dir = args.out or (script_dir / "results" / "hardware_behavior_latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    config = HardwareConfig(
        q_use_bits=args.q_use_bits,
        frac_bits=args.frac_bits,
        enable_cycle_model=not args.no_cycle_model,
    )

    summaries: list[RunSummary] = []
    histories: list[dict[str, float | int | str]] = []
    total_cases = len(args.modes) * len(args.sizes) * len(args.seeds)
    case_index = 0
    for mode in args.modes:
        for n in args.sizes:
            for seed in args.seeds:
                case_index += 1
                print(f"[{case_index}/{total_cases}] mode={mode} n={n} seed={seed}")
                summary, history_rows = run_case(
                    n=n,
                    mode=mode,
                    seed=seed,
                    config=config,
                    ir_iters=args.ir_iters,
                    tol=args.tol,
                )
                summaries.append(summary)
                histories.extend(history_rows)
                if summary.status != "ok":
                    print(f"  failed: {summary.error}")

    write_summary_csv(out_dir / "run_summary.csv", summaries)
    write_csv(out_dir / "residual_history.csv", histories)
    write_csv(
        out_dir / "op_counts.csv",
        [
            {
                "case_id": summary.case_id,
                "mode": summary.mode,
                "n": summary.n,
                "seed": summary.seed,
                "panel_fact_ops": summary.panel_fact_ops,
                "trsm_u_ops": summary.trsm_u_ops,
                "trsm_l_ops": summary.trsm_l_ops,
                "gemm_schur_ops": summary.gemm_schur_ops,
                "total_cycles": summary.total_cycles,
            }
            for summary in summaries
        ],
    )
    write_csv(
        out_dir / "quant_stats.csv",
        [
            {
                "case_id": summary.case_id,
                "mode": summary.mode,
                "n": summary.n,
                "seed": summary.seed,
                "e_node": summary.e_node,
                "align_drop_count": summary.align_drop_count,
                "align_shift_max": summary.align_shift_max,
                "asm_overflow_count": summary.asm_overflow_count,
                "requant_sat_count": summary.requant_sat_count,
                "maxabs_acc": summary.maxabs_acc,
            }
            for summary in summaries
        ],
    )

    if not args.no_plots:
        try:
            plt = prepare_plotting(out_dir)
            plot_residual_convergence(fig_dir, histories, args.modes, plt)
            plot_final_residuals(fig_dir, summaries, args.modes, plt)
            plot_op_counts(fig_dir, summaries, plt)
            plot_quant_risks(fig_dir, summaries, args.modes, plt)
        except Exception as exc:
            print(f"plot generation failed: {type(exc).__name__}: {exc}")

    write_report(out_dir, summaries, histories, args)

    ok_count = sum(1 for summary in summaries if summary.status == "ok")
    print(f"done: {ok_count}/{len(summaries)} cases succeeded")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
