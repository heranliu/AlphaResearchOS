"""Derive transparent diagnostics from the recorded benchmark evidence.

Run: uv run --locked --extra plot python benchmarks/sector-etf/plot_diagnostics.py
No downloads, model calls, backtest reruns, or edits to results.json / curves.csv.
The report hierarchy takes inspiration from QuantStats and pyfolio tearsheets;
the calculations and matplotlib figures here are independently implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from plot_benchmark import load_data  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KEYS = ["alphaos", "equal_weight", "momentum_20", "momentum_60"]
LABELS = ["AlphaResearchOS", "Equal weight", "20-day momentum", "60-day momentum"]
COLORS = ["#07866D", "#506981", "#C18B3B", "#8F80B7"]
TEXT, MUTED, GRID = "#18303F", "#627582", "#E4EBEE"


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "text.color": TEXT, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlecolor": TEXT, "axes.titlesize": 12.5,
        "axes.axisbelow": True, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "alphaos-diagnostics-v1",
    })


def grouped_returns(dates, returns, by_month=False):
    groups = {}
    for index, date in enumerate(dates):
        period = date.strftime("%Y-%m") if by_month else str(date.year)
        groups.setdefault(period, []).append(index)
    return {
        period: {key: float(np.prod(1 + returns[key][indices]) - 1) for key in KEYS}
        for period, indices in groups.items()
    }


def derive(result, dates, curves):
    daily = {key: equity / np.r_[1.0, equity[:-1]] - 1 for key, equity in curves.items()}
    annual = grouped_returns(dates, daily)
    monthly = grouped_returns(dates, daily, by_month=True)
    # Both boundary months are partial in this fixed historical benchmark.
    # Excluding them makes the denominator explicit, without assuming a holiday calendar.
    complete = list(monthly)[1:-1]
    if complete != [f"{year}-{month:02}" for year in range(2023, 2027)
                    for month in range(1, 13) if "2023-06" <= f"{year}-{month:02}" <= "2026-08"]:
        raise ValueError("Unexpected monthly coverage; review boundary-month exclusions")
    for key in KEYS:
        for grouped in (annual, monthly):
            compounded = math.prod(1 + row[key] for row in grouped.values()) - 1
            if not math.isclose(compounded, curves[key][-1] - 1, abs_tol=1e-12):
                raise ValueError("Grouped compounding does not reconcile to the full equity curve")
    paired = {}
    for key in KEYS[1:]:
        differences = [monthly[period]["alphaos"] - monthly[period][key] for period in complete]
        wins = sum(value > 0 for value in differences)
        paired[key] = {
            "months": len(complete), "wins": wins,
            "ties": sum(value == 0 for value in differences),
            "win_fraction": wins / len(complete),
            "median_return_difference_pp": float(np.median(differences) * 100),
        }
    return {
        "schema_version": 1,
        "evidence_status": "Historical holdout benchmark from recorded results and daily curves",
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("results.json", "curves.csv")
        },
        "evaluation_period": result["evaluation_period"],
        "annual_returns": annual,
        "partial_years": {"2023": "2023-05-11 to 2023-12-29", "2026": "2026-01-02 to 2026-09-17"},
        "monthly_returns": monthly,
        "complete_months": complete,
        "paired_monthly_comparisons": paired,
        "definitions": {
            "daily_return": "equity[t] / equity[t-1] - 1, with initial cash equity 1",
            "period_return": "product of (1 + daily return) within the period, minus 1",
            "monthly_win": "AlphaResearchOS monthly net return strictly exceeds paired baseline return",
            "monthly_coverage": "39 full interior months; partial May 2023 and September 2026 excluded",
            "drawdown": "equity / running peak including initial equity 1, minus 1",
            "annualization": "252 sessions; sample daily-return standard deviation; zero risk-free rate",
        },
    }


def heading(fig, title, subtitle):
    fig.text(.065, .96, "ALPHARESEARCHOS  /  HISTORICAL EVIDENCE", color=COLORS[0],
             size=10, weight="bold", va="top")
    fig.text(.065, .915, title, size=22, weight="bold", va="top")
    fig.text(.065, .866, subtitle, color=MUTED, size=10.5, va="top")


def save(fig, output, name):
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else None
        destination = output / f"{name}.{extension}"
        fig.savefig(destination, dpi=160, metadata=metadata)
        if extension == "svg":
            destination.write_text("\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n")
    plt.close(fig)


def diagnostics(result, dates, curves, metrics, output):
    fig = plt.figure(figsize=(14.2, 10.3))
    heading(fig, "Performance beyond the headline", "2023-05-11 — 2026-09-17 · 841 sessions · 10 bps one-way costs")
    handles = [Line2D([0], [0], color=color, lw=3, label=label)
               for label, color in zip(LABELS, COLORS, strict=True)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.06, .835), ncol=4,
               handlelength=1.8, columnspacing=2.2)
    grid = fig.add_gridspec(2, 2, left=.115, right=.96, bottom=.14, top=.755,
                           wspace=.34, hspace=.54, height_ratios=[1, .94])
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("01  Return versus variability", loc="left", pad=17)
    offsets = [(10, -4), (-8, -28), (10, -15), (-10, 18)]
    for index, key in enumerate(KEYS):
        item = result["strategies"][key]["metrics"]
        x, y = item["annual_vol"] * 100, item["cagr"] * 100
        ax.scatter(x, y, s=160 if index == 0 else 95, color=COLORS[index], zorder=3)
        ax.annotate(f"{LABELS[index]}\n{y:.1f}% CAGR", (x, y), xytext=offsets[index],
                    textcoords="offset points", fontsize=9.5,
                    ha="right" if index in (1, 3) else "left", color=COLORS[index])
    ax.set(xlim=(10.8, 18.2), ylim=(9, 30), xlabel="Annualized volatility (%)", ylabel="CAGR (%)")
    ax.grid(color=GRID)
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("02  Drawdowns through the full holdout", loc="left", pad=17)
    for index in [1, 2, 3, 0]:
        equity = curves[KEYS[index]]
        dd = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1
        ax.plot(dates, dd, color=COLORS[index], lw=1.8 if index == 0 else 1.1,
                alpha=1 if index == 0 else .8)
    ax.set_ylim(-.245, .008)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID)
    ax.set_ylabel("Below prior equity peak")
    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("03  Calendar-period net returns", loc="left", pad=17)
    values = np.array([[metrics["annual_returns"][year][key] * 100
                        for year in metrics["annual_returns"]] for key in KEYS])
    cmap = LinearSegmentedColormap.from_list("net-return", ["#F9FBFA", "#B8DFD1", "#07866D"])
    ax.imshow(values, cmap=cmap, vmin=0, vmax=35, aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{values[i, j]:.1f}%", ha="center", va="center", weight="bold" if i == 0 else "normal",
                    color="white" if values[i, j] > 24 else TEXT, size=11)
    ax.set_xticks(range(4), ["2023*", "2024", "2025", "2026 YTD*"])
    ax.set_yticks(range(4), ["AlphaOS", "Equal weight", "20d momentum", "60d momentum"])
    ax.tick_params(length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0, -.23, "* Partial periods: May 11–Dec 29, 2023; Jan 2–Sep 17, 2026.",
            transform=ax.transAxes, color=MUTED, fontsize=9)
    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("04  Months ahead of each baseline", loc="left", pad=17)
    for index, key in enumerate(KEYS[1:]):
        item = metrics["paired_monthly_comparisons"][key]
        win = item["win_fraction"] * 100
        ax.barh(index, 100, height=.44, color="#EEF3F5")
        ax.barh(index, win, height=.44, color=COLORS[index + 1])
        ax.text(win - 2, index, f"{item['wins']}/39  ·  {win:.1f}%", color="white",
                va="center", ha="right", weight="bold", fontsize=10)
    ax.set_yticks(range(3), ["vs equal weight", "vs 20d momentum", "vs 60d momentum"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.axvline(50, color=MUTED, lw=.8, linestyle="--", alpha=.6)
    ax.tick_params(axis="y", length=0)
    ax.text(0, -.23, "39 complete months · Jun 2023–Aug 2026 · no ties", transform=ax.transAxes,
            color=MUTED, fontsize=9)
    fig.text(.065, .027, "Same dates and costs for all portfolios. Historical comparisons use the recorded holdout results.",
             color=MUTED, size=9.5)
    save(fig, output, "benchmark-diagnostics")


def implementation(result, output):
    fig = plt.figure(figsize=(14.2, 6.7))
    heading(fig, "Trading frictions and portfolio activity", "Frozen strategy · identical scoring and selection · costs varied after selection")
    grid = fig.add_gridspec(1, 2, left=.08, right=.96, bottom=.19, top=.72, wspace=.39)
    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("01  Cost-matched net return", loc="left", pad=15)
    scenarios = result["cost_sensitivity"]
    costs = [item["cost_bps"] for item in scenarios]
    a = np.array([item["alphaos"]["total_return"] * 100 for item in scenarios])
    b = np.array([item["equal_weight"]["total_return"] * 100 for item in scenarios])
    ax.fill_between(costs, a, b, color=COLORS[0], alpha=.09)
    ax.plot(costs, a, color=COLORS[0], marker="o", lw=2.5, label="AlphaResearchOS")
    ax.plot(costs, b, color=COLORS[1], marker="o", lw=2, label="Equal weight")
    for x, y, base in zip(costs, a, b, strict=True):
        ax.annotate(f"+{y - base:.1f} pp", (x, (y + base) / 2), ha="center", va="center", color=COLORS[0], size=10)
    ax.set(xlim=(-2, 42), ylim=(0, 145), xlabel="One-way transaction costs (bps)", ylabel="Cumulative net return (%)")
    ax.set_xticks(costs)
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("02  Daily traded value / portfolio NAV", loc="left", pad=15)
    values = [result["strategies"][key]["metrics"]["avg_turnover"] * 100 for key in KEYS]
    ax.barh(range(4), values, height=.49, color=COLORS)
    for i, value in enumerate(values):
        ax.text(value + .25, i, f"{value:.2f}%", va="center", color=COLORS[i], weight="bold")
    ax.set_yticks(range(4), LABELS)
    ax.invert_yaxis()
    ax.set_xlim(0, 15.8)
    ax.set_xlabel("Average daily turnover (%) · at 10 bps")
    ax.grid(axis="x", color=GRID)
    ax.tick_params(axis="y", length=0)
    fig.text(.065, .083, "At 40 bps: +92.15% net return; +31.00 pp over cost-matched equal weight.",
             color=TEXT, weight="bold", size=11)
    fig.text(.065, .042, "Turnover includes entry, drift-aware rebalancing and liquidation. AlphaOS trades less than both momentum baselines, more than equal weight.",
             color=MUTED, size=9.4)
    save(fig, output, "benchmark-frictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--metrics-path", type=Path, default=HERE / "analysis_metrics.json")
    args = parser.parse_args()
    result, dates, curves = load_data()  # Validates all headline metrics against the daily curves.
    metrics = derive(result, dates, curves)
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics(result, dates, curves, metrics, args.output_dir)
    implementation(result, args.output_dir)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Validated 841 sessions; wrote diagnostics, frictions and {args.metrics_path.name}.")


if __name__ == "__main__":
    main()
