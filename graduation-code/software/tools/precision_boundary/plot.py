"""Figures of the precision-boundary experiment (static PNG for the thesis).

Reads the summarized results.csv (so the figures can be regenerated from the
copy kept under experiments/precision_boundary/results/2026-10-11/) and writes four figures:

  fig1_f1_boundary.png      F1: forward error vs kappa (one solve / IR / FP64)
  fig2_success_rates.png    success rate per kappa decade, F1 vs F3 with growth > 10
  fig3_kappa_growth.png     (kappa, static growth) plane coloured by IR outcome
  fig4_device_eta.png       device eta_f / eta_b vs kappa (device correctness)

Palette: the validated reference categorical slots 1-3 (blue, orange, aqua),
all-pairs CVD-safe; every series also has its own marker shape and a direct
label (aqua is below 3:1 contrast, so labels are mandatory).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

U32 = 2.0**-24
U64 = 2.0**-53

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _setup():
    import logging

    import matplotlib

    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        # Per-glyph fallback: Latin/maths/minus from DejaVu Sans, CJK from Hiragino.
        "font.family": ["DejaVu Sans", "Hiragino Sans GB"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 6,
        "mathtext.fontset": "dejavusans",
    })
    return plt


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with csv_path.open() as stream:
        for row in csv.DictReader(stream):
            for key, value in row.items():
                if value in ("", "None"):
                    row[key] = None
                elif value in ("True", "False"):
                    row[key] = value == "True"
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def _col(rows, key):
    return np.array([r[key] if isinstance(r.get(key), float) else np.nan for r in rows])


def _kappa_line(ax, unit, label, xs, at, style="--"):
    """Reference line y = kappa*unit, labelled at kappa = at (inside the axes)."""
    ax.plot(xs, xs * unit, style, color=INK2, linewidth=1)
    ax.annotate(label, (at, at * unit), xytext=(-6, 6), textcoords="offset points",
                ha="right", va="bottom", color=INK2, fontsize=9)


def fig1(rows, out: Path, plt) -> None:
    f1 = [r for r in rows if r["family"] == "F1" and isinstance(r.get("device_forward"), float)]
    k = _col(f1, "kappa2_DrA")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    series = [
        ("device_forward", BLUE, "o", "设备求解一次"),
        ("ir_final_forward", ORANGE, "^", "设备因子 + FP64 迭代求精（10 步）"),
        ("fp64_forward", AQUA, "s", "FP64 直接求解（参考）"),
    ]
    for key, color, marker, label in series:
        y = np.maximum(_col(f1, key), 1e-17)
        ax.scatter(k, y, s=26, color=color, marker=marker, label=label, linewidths=0, alpha=0.85)
    xs = np.logspace(0, 12, 50)
    _kappa_line(ax, U32, r"$\kappa\cdot u_{32}$", xs, at=1e8)
    _kappa_line(ax, U64, r"$\kappa\cdot u_{64}$", xs, at=1e11, style=":")
    ax.axhline(1e-3, color=INK2, linewidth=0.8)
    ax.annotate("可用阈值 1e-3", (1.5, 1e-3), xytext=(0, 3), textcoords="offset points",
                color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e12); ax.set_ylim(1e-17, 1e3)
    ax.set_xlabel("κ₂(DᵣA)（设备实际分解的矩阵）")
    ax.set_ylabel("解的相对误差 ‖x̂ − x‖ / ‖x‖")
    ax.set_title("F1（对称正定，增长因子 g = 1）：解误差随条件数的变化")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig1_f1_boundary.png", dpi=200)
    plt.close(fig)


def _rates(rows, select, key):
    by = {}
    for r in rows:
        if not select(r) or not isinstance(r.get("kappa2_DrA"), float) or r.get(key) is None:
            continue
        b = int(np.floor(np.log10(r["kappa2_DrA"])))
        by.setdefault(b, []).append(bool(r[key]))
    bins = sorted(by)
    return np.array([10 ** (b + 0.5) for b in bins]), np.array([np.mean(by[b]) for b in bins])


GROWTH_SPLIT = 10.0


def fig2(rows, out: Path, plt) -> None:
    f1 = lambda r: r["family"] == "F1"  # noqa: E731
    grown = lambda r: (r["family"] == "F3" and isinstance(r.get("static_growth"), float)  # noqa: E731
                       and r["static_growth"] > GROWTH_SPLIT)
    g_med = np.median([r["static_growth"] for r in rows if grown(r)])
    label_g = f"F3，静态 g > {GROWTH_SPLIT:g}（中位数 {g_med:.0e}）".replace("e+0", "e")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), sharey=True)
    panels = [
        (axes[0], "求解一次可用（误差 ≤ 1e-3）", [
            (f1, "one_shot_ok", BLUE, "o", "F1（g = 1）"),
            (grown, "one_shot_ok", ORANGE, "^", label_g),
        ]),
        (axes[1], "迭代求精成功（10 步内）", [
            (f1, "ir_ok", BLUE, "o", "F1，静态主元（g = 1）"),
            (grown, "ir_ok", ORANGE, "^", label_g),
            (grown, "gepp_ir_ok", AQUA, "s", "同一批 F3，FP32 部分选主元（参考）"),
        ]),
    ]
    for ax, title, series in panels:
        for select, key, color, marker, label in series:
            x, y = _rates(rows, select, key)
            ax.plot(x, y, color=color, marker=marker, label=label)
        ax.axvline(1 / U32, color=INK2, linewidth=0.8, linestyle="--")
        ax.annotate(r"$1/u_{32}$", (1 / U32, 0.93), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=9)
        ax.set_xscale("log"); ax.set_xlim(1, 1e13); ax.set_ylim(-0.03, 1.05)
        ax.set_title(title)
        ax.set_xlabel("κ₂(DᵣA)（按数量级分箱）")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5)
    axes[0].set_ylabel("成功比例")
    fig.tight_layout()
    fig.savefig(out / "fig2_success_rates.png", dpi=200)
    plt.close(fig)


def fig3(rows, out: Path, plt) -> None:
    pts = [r for r in rows if isinstance(r.get("static_growth"), float) and r.get("ir_ok") is not None]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for ok, color, marker, label in ((True, BLUE, "o", "迭代求精成功"), (False, ORANGE, "x", "迭代求精失败")):
        sel = [r for r in pts if r["ir_ok"] is ok and r["family"] != "real"]
        ax.scatter(_col(sel, "kappa2_DrA"), _col(sel, "static_growth"), s=22 if ok else 26,
                   color=color, marker=marker, label=label, linewidths=1.2 if not ok else 0, alpha=0.8)
    xs = np.logspace(0, 14, 50)
    ax.plot(xs, 1 / (xs * U32), "--", color=INK2, linewidth=1)
    ax.annotate("κ·g·u₃₂ = 1（理论边界）", (2e3, 1 / (2e3 * U32)), xytext=(6, 2),
                textcoords="offset points", color=INK2, fontsize=9)
    for r in pts:
        if r["family"] == "real":
            ax.scatter([r["kappa2_DrA"]], [r["static_growth"]], s=70, facecolors="none",
                       edgecolors=INK, linewidths=1.4, zorder=5)
            ax.annotate(r["case_id"].replace("real-jj", "") + "JJ", (r["kappa2_DrA"], r["static_growth"]),
                        xytext=(-8, 6), textcoords="offset points", ha="right", fontsize=9, color=INK)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e14); ax.set_ylim(0.5, 1e6)
    ax.set_xlabel(r"$\kappa_2(D_r A)$")
    ax.set_ylabel("静态主元增长因子 g = max(|L||U|) / max|Â|")
    ax.set_title("条件数与增长因子平面上的迭代求精结果（F1/F2/F3 + 真实矩阵）")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_kappa_growth.png", dpi=200)
    plt.close(fig)


def fig4(rows, out: Path, plt) -> None:
    pts = [r for r in rows if isinstance(r.get("eta_f"), float)]
    k = _col(pts, "kappa2_DrA")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.scatter(k, _col(pts, "eta_f"), s=20, color=BLUE, marker="o", label="前代（L）后向误差", linewidths=0, alpha=0.8)
    ax.scatter(k, _col(pts, "eta_b"), s=22, color=ORANGE, marker="^", label="回代（U）后向误差", linewidths=0, alpha=0.8)
    ax.axhline(U32, color=INK2, linewidth=0.8, linestyle="--")
    ax.annotate(r"$u_{32}$", (2, U32), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=9)
    ax.axhline(1e-4, color=INK2, linewidth=0.8)
    ax.annotate("checker 容限 1e-4", (2, 1e-4), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e14); ax.set_ylim(1e-9, 1e-3)
    ax.set_xlabel(r"$\kappa_2(D_r A)$")
    ax.set_ylabel("后向误差")
    ax.set_title("设备三角求解的后向误差与条件数无关（全部用例）")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig4_device_eta.png", dpi=200)
    plt.close(fig)


def make_figures(csv_path: Path, out: Path) -> list[Path]:
    plt = _setup()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    for fig in (fig1, fig2, fig3, fig4):
        fig(rows, out, plt)
    return sorted(out.glob("fig*.png"))
