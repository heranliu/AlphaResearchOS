"""Causal, long-only close-to-close research backtests with explicit costs.

Return row t earns close[t]/close[t-1]-1. At a scheduled rebalance the order
executes at close[t-1], based on scores[t-signal_lag] (lag >= 2). Thus a
close-derived signal is never assumed executable at that same close.
Each [start, end) fold starts in cash at close[start-1] and liquidates at
close[end-1]. The equal-weight benchmark follows identical conventions.

One-way turnover is actual gross traded notional / row-opening NAV, including
the initial purchase and final liquidation. Costs reduce investable NAV;
weights subsequently drift with asset returns. No leverage or filling of
missing prices is permitted. Dividends require total-return adjusted inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = 10.0
    top_k: int = 3
    rebalance_every: int = 5
    signal_lag: int = 2
    annualization: int = 252

    def __post_init__(self) -> None:
        if not math.isfinite(self.cost_bps) or not 0 <= self.cost_bps <= 1000:
            raise ValueError("cost_bps must be finite and between 0 and 1000.")
        for name in ("top_k", "rebalance_every", "signal_lag", "annualization"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.signal_lag < 2:
            raise ValueError("signal_lag must be at least 2 for close-derived signals.")


def _rebalance(weights: np.ndarray, target: np.ndarray, cost: float) -> tuple[np.ndarray, float, float]:
    """Return post-cost dollar holdings / pre-trade NAV, NAV ratio, turnover.

    Solve k + cost * sum(abs(k * target - weights)) = 1. Since
    cost <= 0.1 and sum(target) <= 1, the fixed point is a contraction.
    """
    if cost == 0:
        return target.copy(), 1.0, float(np.abs(target - weights).sum())
    remaining = 1.0
    for _ in range(32):
        updated = 1.0 - cost * float(np.abs(remaining * target - weights).sum())
        if abs(updated - remaining) < 1e-14:
            remaining = updated
            break
        remaining = updated
    holdings = remaining * target
    turnover = float(np.abs(holdings - weights).sum())
    return holdings, remaining, turnover


def _check_inputs(scores: pd.DataFrame, close: pd.DataFrame, start: int, end: int) -> None:
    if not isinstance(scores, pd.DataFrame) or not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("scores and close must be nonempty aligned DataFrames.")
    if not scores.index.equals(close.index) or not scores.columns.equals(close.columns):
        raise ValueError("scores and close must have exactly matching dates and assets.")
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.has_duplicates or close.index.hasnans or not close.index.is_monotonic_increasing:
        raise ValueError("Dates must be a unique ascending DatetimeIndex.")
    if close.columns.has_duplicates:
        raise ValueError("Asset columns must be unique.")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(close):
        raise ValueError("Require 0 <= start < end <= number of rows.")
    prices = close.to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be positive and complete; missing data cannot be filled implicitly.")


def backtest(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    config: BacktestConfig,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Simulate an independent fold with cold start and terminal liquidation.

    Finite scores are eligible; top-k ties break by lexical asset name. A
    scheduled rebalance with no valid scores moves to cash. Invalid scores
    between rebalances do not trigger an unscheduled trade.
    """
    _check_inputs(scores, close, start, end)
    n_assets = len(close.columns)
    prices = close.to_numpy(dtype=float)
    signals = scores.to_numpy(dtype=float)
    asset_names = np.asarray([str(asset) for asset in close.columns])
    cost = config.cost_bps / 10_000.0
    strategy_weights = np.zeros(n_assets)
    benchmark_weights = np.zeros(n_assets)
    benchmark_target = np.full(n_assets, 1.0 / n_assets)
    records = []
    equity = benchmark_equity = 1.0

    for row in range(start, end):
        target = None
        # A row-zero return does not exist. The schedule stays anchored at start.
        rebalance_now = row > 0 and (row - start) % config.rebalance_every == 0
        if rebalance_now:
            target = np.zeros(n_assets)
            signal_row = row - config.signal_lag
            if signal_row >= 0:
                eligible = np.flatnonzero(np.isfinite(signals[signal_row]))
                if eligible.size:
                    ranked = eligible[np.lexsort((asset_names[eligible], -signals[signal_row, eligible]))]
                    selected = ranked[: config.top_k]
                    target[selected] = 1.0 / len(selected)
        asset_return = np.zeros(n_assets) if row == 0 else prices[row] / prices[row - 1] - 1.0

        def advance(weights: np.ndarray, desired: np.ndarray | None, asset_return=asset_return, row=row) -> tuple[np.ndarray, float, float, float]:
            if desired is None:
                holdings, remaining, turnover = weights.copy(), 1.0, 0.0
            else:
                holdings, remaining, turnover = _rebalance(weights, desired, cost)
            cash = max(0.0, remaining - float(holdings.sum()))
            ending_holdings = holdings * (1.0 + asset_return)
            ending_nav = cash + float(ending_holdings.sum())
            # Exposure is the post-trade allocation during this return interval.
            exposure = float(holdings.sum() / remaining) if remaining else 0.0
            if row == end - 1:
                liquidation = float(ending_holdings.sum())
                # Express both trades in the row's opening NAV units.
                turnover += liquidation
                ending_nav -= cost * liquidation
                ending_holdings[:] = 0.0
            if not np.isfinite(ending_nav) or ending_nav <= 0:
                raise ValueError("Portfolio NAV became nonpositive or nonfinite.")
            return ending_holdings / ending_nav, ending_nav - 1.0, turnover, exposure

        strategy_weights, daily_return, turnover, exposure = advance(strategy_weights, target)
        benchmark_weights, benchmark_return, benchmark_turnover, _ = advance(
            benchmark_weights, benchmark_target if rebalance_now else None
        )
        equity *= 1.0 + daily_return
        benchmark_equity *= 1.0 + benchmark_return
        if not math.isfinite(equity) or not math.isfinite(benchmark_equity):
            raise ValueError("Cumulative equity overflowed; check input price scales and returns.")
        records.append({
            "return": daily_return,
            "benchmark_return": benchmark_return,
            "turnover": turnover,
            "benchmark_turnover": benchmark_turnover,
            "exposure": exposure,
            "equity": equity,
            "benchmark_equity": benchmark_equity,
        })
    return pd.DataFrame(records, index=close.index[start:end])


def _sharpe(returns: np.ndarray, annualization: int) -> float:
    if len(returns) < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    return float(np.mean(returns) / deviation * np.sqrt(annualization)) if deviation > 1e-12 else 0.0


def summarize(curve: pd.DataFrame, annualization: int = 252) -> dict[str, Any]:
    """Return finite, JSON-safe metrics; drawdown includes initial equity 1."""
    if type(annualization) is not int or annualization < 1:
        raise ValueError("annualization must be a positive integer.")
    if curve.empty:
        raise ValueError("Cannot summarize an empty curve.")
    returns = curve["return"].to_numpy(dtype=float)
    benchmark_returns = curve["benchmark_return"].to_numpy(dtype=float)
    if not np.isfinite(returns).all() or not np.isfinite(benchmark_returns).all() or (returns <= -1).any() or (benchmark_returns <= -1).any():
        raise ValueError("Curve returns must be finite and greater than -1.")
    log_growth = float(np.log1p(returns).sum())
    equity = np.exp(np.cumsum(np.log1p(returns)))
    total_return = float(np.expm1(log_growth))
    benchmark_total = float(np.expm1(np.log1p(benchmark_returns).sum()))
    peak = np.maximum.accumulate(np.concatenate(([1.0], equity)))[1:]
    annual_log_growth = log_growth * annualization / len(returns)
    # A short extreme series can mathematically overflow annualization. Keep
    # JSON valid without pretending such a CAGR is an ordinary finite number.
    cagr = float(np.expm1(annual_log_growth)) if annual_log_growth < 700 else None
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": _sharpe(returns, annualization),
        "max_drawdown": float(np.min(equity / peak - 1.0)),
        "annual_vol": float(np.std(returns, ddof=1) * np.sqrt(annualization)) if len(returns) > 1 else 0.0,
        "avg_turnover": float(curve["turnover"].mean()),
        "days": int(len(returns)),
        "benchmark_total_return": benchmark_total,
        "excess_return": total_return - benchmark_total,
    }


def _rank_ic(scores: pd.DataFrame, close: pd.DataFrame, signal_lag: int, rows: list[int]) -> float:
    """Daily Spearman IC against the return the lagged signal can first earn."""
    future_returns = close.div(close.shift(1)) - 1.0
    lagged = scores.shift(signal_lag).replace([np.inf, -np.inf], np.nan)
    values = []
    for row in rows:
        score = lagged.iloc[row]
        realized = future_returns.iloc[row]
        mask = score.notna() & realized.notna()
        if int(mask.sum()) < 2:
            continue
        x = score[mask].rank(method="average").to_numpy()
        y = realized[mask].rank(method="average").to_numpy()
        if np.std(x) > 0 and np.std(y) > 0:
            values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else 0.0


def evaluate_development(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    config: BacktestConfig,
    folds: list[tuple[int, int]],
) -> dict[str, Any]:
    """Score fixed chronological development folds, never a final holdout.

    score = mean(fold Sharpe) - 0.5 * std(fold Sharpe)
            + 2 * mean(fold total-return minus benchmark total-return).
    Complexity and trial-budget penalties belong to the research controller.
    Fold curves are chained for descriptive aggregate metrics only; each fold
    pays its own cold-start and liquidation costs. Rank IC is diagnostic.
    """
    if not folds:
        raise ValueError("At least one development fold is required.")
    previous_end = -1
    results, curves, rows = [], [], []
    for start, end in folds:
        if start < previous_end:
            raise ValueError("Development folds must be chronological and nonoverlapping.")
        curve = backtest(scores, close, config, start, end)
        metrics = summarize(curve, config.annualization)
        results.append({"start": int(start), "end": int(end), "metrics": metrics})
        curves.append(curve)
        rows.extend(range(start, end))
        previous_end = end
    sharpes = np.asarray([result["metrics"]["sharpe"] for result in results])
    excess = np.asarray([result["metrics"]["excess_return"] for result in results])
    score = float(sharpes.mean() - 0.5 * sharpes.std(ddof=0) + 2.0 * excess.mean())
    combined = pd.concat(curves)
    combined["equity"] = (1.0 + combined["return"]).cumprod()
    combined["benchmark_equity"] = (1.0 + combined["benchmark_return"]).cumprod()
    return {
        "score": score,
        "metrics": summarize(combined, config.annualization),
        "folds": results,
        "rank_ic": _rank_ic(scores, close, config.signal_lag, rows),
        "stability": float(np.mean(excess > 0)),
    }


def bootstrap_sharpe_interval(returns: Any, seed: int = 42) -> list[float]:
    """95% circular-block bootstrap interval (1,000 resamples, 252 days/year).

    Blocks of round(sqrt(n)) retain some serial dependence. This describes
    realized returns, without correcting selection bias or regime changes.
    """
    values = np.asarray(returns, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap returns must be nonempty and finite.")
    if len(values) < 2 or np.std(values) < 1e-12:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    block_size = max(2, round(np.sqrt(len(values))))
    block_count = math.ceil(len(values) / block_size)
    starts = rng.integers(0, len(values), size=(1000, block_count))
    positions = (starts[:, :, None] + np.arange(block_size)[None, None, :]) % len(values)
    samples = values[positions.reshape(1000, -1)[:, : len(values)]]
    deviations = samples.std(axis=1, ddof=1)
    estimates = np.divide(samples.mean(axis=1), deviations, out=np.zeros(1000), where=deviations > 1e-12) * np.sqrt(252)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]
