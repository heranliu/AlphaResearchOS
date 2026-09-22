"""Render the sector ETF benchmark from public derived data, with no model calls.

Run from any directory:
    uv run --extra plot python benchmarks/sector-etf/plot_benchmark.py

Plot brief: exact numerical comparisons; white background; green strategy and
neutral baselines; zero-based bars; four complete equity paths; no smoothing,
invented confidence intervals, downloaded data, or rewritten research results.
Inputs are results.json and the 841-row derived curves.csv beside this script.
Outputs are three PNG/SVG pairs in docs/assets by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORDER = ["alphaos", "equal_weight", "momentum_20", "momentum_60"]
COLORS = {"alphaos": "#09866B", "equal_weight": "#526477",
          "momentum_20": "#A2ABB5", "momentum_60": "#81908A"}
TEXT = "#22313B"
MUTED = "#62727A"
GRID = "#E6EBE9"


def load_data():
    result = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    with (HERE / "curves.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", *ORDER]:
            raise ValueError("curves.csv must contain date and exactly four named equity series")
        rows = list(reader)
    dates = [datetime.strptime(row["date"], "%Y-%m-%d") for row in rows]
    period = result["evaluation_period"]
    if (len(rows) != period["sessions"] or dates != sorted(set(dates))
            or rows[0]["date"] != period["start"] or rows[-1]["date"] != period["end"]):
        raise ValueError("Equity dates differ from the stated evaluation period")
    curves = {key: np.array([float(row[key]) for row in rows]) for key in ORDER}
    for key, equity in curves.items():
        if not np.isfinite(equity).all() or (equity <= 0).any():
            raise ValueError("Equity must be finite and strictly positive")
        # Recompute the plotted financial metrics from the public equity series.
        # Initial cash is 1; no first-day cost or initial drawdown is discarded.
        changes = equity / np.r_[1.0, equity[:-1]] - 1
        std = np.std(changes, ddof=1)
        computed = {
            "total_return": equity[-1] - 1,
            "cagr": equity[-1] ** (252 / len(equity)) - 1,
            "sharpe": float(np.mean(changes) / std * math.sqrt(252)) if std else 0.0,
            "max_drawdown": float(np.min(equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1)),
            "annual_vol": float(std * math.sqrt(252)),
        }
        for metric, actual in computed.items():
            if not math.isclose(actual, result["strategies"][key]["metrics"][metric], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{key}.{metric} differs between the plotted curve and results.json")
    if [row["cost_bps"] for row in result["cost_sensitivity"]] != [0, 10, 20, 40]:
        raise ValueError("Expected fixed transaction-cost scenarios: 0, 10, 20, 40 bps")
    for scenario in result["cost_sensitivity"]:
        for key in ["alphaos", "equal_weight"]:
            if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scenario[key].values()):
                raise ValueError("Cost sensitivity must contain finite numeric metrics")
            if scenario["cost_bps"] == result["method"]["cost_bps_one_way"]:
                if scenario[key] != result["strategies"][key]["metrics"]:
                    raise ValueError("Base-case costs disagree with the headline metrics")
    return result, dates, curves


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": TEXT,
        "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.linewidth": .8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.titlecolor": TEXT,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
        "svg.fonttype": "none", "svg.hashsalt": "alpharesearchos-sector-etf",
        "lines.solid_capstyle": "round", "axes.axisbelow": True,
    })


def heading(fig, title, subtitle):
    fig.text(.06, .965, "ALPHARESEARCHOS  /  SECTOR ETF BENCHMARK", color=COLORS["alphaos"],
             fontsize=10, weight="bold", va="top")
    fig.text(.06, .92, title, fontsize=24, weight="bold", va="top")
    fig.text(.06, .861, subtitle, fontsize=11, color=MUTED, va="top")


def period_label(result):
    period = result["evaluation_period"]
    return f'{period["start"]} – {period["end"]}  ·  {period["sessions"]} trading sessions  ·  9 US sector ETFs'


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "svg"]:
        metadata = {"Software": "AlphaResearchOS / Matplotlib"} if suffix == "png" else {"Date": None, "Creator": "AlphaResearchOS / Matplotlib"}
        fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight", pad_inches=.2, metadata=metadata)
    plt.close(fig)


def performance(result, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.3))
    fig.subplots_adjust(left=.19, right=.945, top=.745, bottom=.12, wspace=.68, hspace=.62)
    heading(fig, "Higher returns than all three baselines", period_label(result))
    labels = [result["strategies"][key]["label"] for key in ORDER]
    specifications = [
        ("total_return", "Cumulative return", 100, 140, 40, "%", False),
        ("cagr", "Annualized return", 100, 32, 10, "%", False),
        ("sharpe", "Sharpe ratio", 1, 1.9, .5, "", False),
        ("max_drawdown", "Maximum drawdown · lower is better", 100, 25, 5, "%", True),
    ]
    for ax, (metric, title, scale, ceiling, step, unit, magnitude) in zip(axes.flat, specifications, strict=True):
        values = [result["strategies"][key]["metrics"][metric] * scale for key in ORDER]
        if magnitude:
            values = [abs(value) for value in values]
        ax.barh(np.arange(4), values, height=.5, color=[COLORS[key] for key in ORDER], zorder=3)
        ax.set_yticks(np.arange(4), labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, ceiling)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position, unit=unit: f"{value:g}{unit}"))
        ax.grid(axis="x", color=GRID, linewidth=.8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.set_title(title, loc="left", pad=16, fontsize=13)
        for row, value in enumerate(values):
            ax.text(value + ceiling * .025, row, f"{value:.2f}{unit}", va="center", fontsize=12,
                    color=COLORS["alphaos"] if row == 0 else TEXT, weight="bold" if row == 0 else "normal")
        ax.get_yticklabels()[0].set_color(COLORS["alphaos"])
        ax.get_yticklabels()[0].set_weight("bold")
    fig.text(.06, .045, "10 bps one-way costs  ·  Rebalance every 5 sessions  ·  Top 3 holdings  ·  Equal weight holds all 9",
             fontsize=10, color=MUTED)
    save(fig, out, "benchmark-performance")


def equity(result, dates, curves, out):
    fig, ax = plt.subplots(figsize=(14, 7.1))
    fig.subplots_adjust(left=.085, right=.95, top=.715, bottom=.16)
    heading(fig, "Growth of $1", period_label(result))
    styles = {"alphaos": "-", "equal_weight": "-", "momentum_20": "--", "momentum_60": "-."}
    for key in ORDER[::-1]:
        ax.plot(dates, curves[key], color=COLORS[key], lw=2.5 if key == "alphaos" else 1.5,
                linestyle=styles[key], zorder=5 if key == "alphaos" else 3)
    handles = [Line2D([], [], color=COLORS[key], lw=2.6 if key == "alphaos" else 1.7, linestyle=styles[key],
                      label=f'{result["strategies"][key]["label"]}  +{result["strategies"][key]["metrics"]["total_return"] * 100:.2f}%')
               for key in ORDER]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.077, .805), ncol=2,
               fontsize=11, columnspacing=3.5, handlelength=3.2)
    ax.axhline(1, color="#B7C1BC", linewidth=.8, linestyle=":", zorder=1)
    ax.set_ylim(.78, 2.35)
    ax.set_xlim(dates[0], dates[-1])
    ax.yaxis.set_major_locator(MultipleLocator(.4))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"${value:.1f}"))
    ticks = [dates[0], datetime(2024, 1, 1), datetime(2025, 1, 1), datetime(2026, 1, 1), dates[-1]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="both", length=0, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.scatter(dates[-1], curves["alphaos"][-1], s=30, color=COLORS["alphaos"], zorder=6, clip_on=False)
    ax.annotate(f'${curves["alphaos"][-1]:.2f}', (dates[-1], curves["alphaos"][-1]), xytext=(-10, 10),
                textcoords="offset points", ha="right", color=COLORS["alphaos"], weight="bold", fontsize=13)
    fig.text(.06, .065, "Net of 10 bps one-way costs  ·  Rebalance every 5 sessions  ·  2-session signal lag",
             fontsize=10, color=MUTED)
    save(fig, out, "benchmark-equity")


def costs(result, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(left=.08, right=.95, top=.68, bottom=.22, wspace=.27)
    heading(fig, "Returns remain ahead as trading costs rise", period_label(result))
    fig.legend(handles=[Patch(color=COLORS[key], label=result["strategies"][key]["label"])
                        for key in ["alphaos", "equal_weight"]],
               loc="upper left", bbox_to_anchor=(.074, .805), ncol=2, fontsize=11, columnspacing=2.5)
    scenarios = result["cost_sensitivity"]
    x = np.arange(len(scenarios))
    width = .32
    for key, offset in [("alphaos", -width / 2), ("equal_weight", width / 2)]:
        values = np.array([scenario[key]["total_return"] * 100 for scenario in scenarios])
        left.bar(x + offset, values, width=width, color=COLORS[key], zorder=3)
        for location, value in zip(x + offset, values, strict=True):
            left.text(location, value + 3, f"{value:.1f}%", ha="center", fontsize=10,
                      color=COLORS[key], weight="bold" if key == "alphaos" else "normal")
        sharpe = [scenario[key]["sharpe"] for scenario in scenarios]
        right.plot([scenario["cost_bps"] for scenario in scenarios], sharpe, color=COLORS[key],
                   marker="o", markersize=5, linewidth=2, zorder=3)
        for scenario, value in zip(scenarios, sharpe, strict=True):
            right.annotate(f"{value:.2f}", (scenario["cost_bps"], value), xytext=(0, 10 if key == "alphaos" else -17),
                           textcoords="offset points", ha="center", fontsize=10, color=COLORS[key])
    left.set_title("Cumulative return", loc="left", pad=16)
    left.set_ylim(0, 145)
    left.set_xticks(x, ["0", "10", "20", "40"])
    left.yaxis.set_major_locator(MultipleLocator(40))
    left.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}%"))
    right.set_title("Sharpe ratio", loc="left", pad=16)
    right.set_ylim(0, 1.95)
    right.set_xlim(-3, 43)
    right.set_xticks([0, 10, 20, 40])
    right.yaxis.set_major_locator(MultipleLocator(.5))
    for ax in [left, right]:
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.set_xlabel("One-way transaction cost (bps)", labelpad=12, fontsize=10)
    advantage = scenarios[-1]["alphaos"]["total_return"] - scenarios[-1]["equal_weight"]["total_return"]
    fig.text(.06, .095, f"At 40 bps: +{advantage * 100:.2f} percentage points of cumulative return versus equal weight.",
             fontsize=11, color=COLORS["alphaos"], weight="bold")
    fig.text(.06, .04, "Same selected strategy and training across cost scenarios",
             fontsize=10, color=MUTED)
    save(fig, out, "benchmark-costs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/assets", help="Output directory for the three PNG/SVG pairs")
    args = parser.parse_args(argv)
    result, dates, curves = load_data()
    configure()
    performance(result, args.out)
    equity(result, dates, curves, args.out)
    costs(result, args.out)
    print(json.dumps({"figures": ["benchmark-performance", "benchmark-equity", "benchmark-costs"],
                      "formats": ["png", "svg"], "curve_rows": len(dates), "plotted_metrics_checked": 20,
                      "output": str(args.out), "model_calls": 0, "network_requests": 0}, indent=2))


if __name__ == "__main__":
    main()
