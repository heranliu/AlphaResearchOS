# US sector ETF benchmark

[Home](../README.md) · [Research methods](METHODS.md) · [Benchmark data](../benchmarks/sector-etf/results.json)

AlphaResearchOS achieved **117.53% total return**, **26.22% CAGR**, and a **1.5608 Sharpe ratio** on nine US sector ETFs during **2023-05-11 to 2026-09-17**, after **10 bps one-way transaction costs**. Development scoring selected a two-feature Ridge strategy before holdout evaluation.

**Evidence status:** this benchmark predates the optional [Jev review gate](JEV.md). The figures below are derived from the existing fixed results and daily curves, not a rerun with Jev. No live Jev evaluation or on/off performance ablation has been measured. The comparisons describe one universe and holdout period, not a statistical guarantee of future outperformance.

## Data and portfolio settings

| Setting | Value |
| --- | --- |
| Market | Nine US sector ETFs: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY |
| Data | Yahoo Finance adjusted daily OHLCV, 2010-01-04 → 2026-09-17; 4,202 sessions |
| Training history | Available observations through 2023-05-08, with matured prediction labels |
| Development evaluation | 2011-01-03 → 2023-05-08; three chronological folds, 3,107 sessions |
| Warmup / separation | 252 sessions / 2 sessions |
| Holdout | 2023-05-11 → 2026-09-17; 841 sessions |
| Portfolio | Long-only; top three equal-weight positions; rebalance every five sessions |
| Execution | Two-session signal lag; 10 bps one-way traded value; cash start and final liquidation |
| Baselines | All-nine-asset equal weight; 20-day and 60-day momentum, each holding the top three |
| Research | Eight attempts; Codex `gpt-6-astra`, medium reasoning; separate proposal and review sessions |

All portfolio comparisons share evaluation dates, signal timing, costs, and liquidation rules. Annualized statistics use 252 trading days and a zero risk-free rate.

## Performance

![Return, risk, and risk-adjusted performance](assets/benchmark-performance.png)

| Strategy | Total return | CAGR | Sharpe | Max drawdown | Average daily turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| **AlphaResearchOS · Ridge** | **117.53%** | **26.22%** | **1.5608** | -16.21% | 4.92% |
| Equal weight | 63.22% | 15.81% | 1.2662 | **-15.61%** | 0.51% |
| 20-day momentum | 58.22% | 14.74% | 1.0615 | -21.21% | 13.11% |
| 60-day momentum | 65.63% | 16.32% | 1.1823 | -18.14% | 7.86% |

The strategy's total-return advantage was **54.32 percentage points over equal weight**, **59.31 over 20-day momentum**, and **51.90 over 60-day momentum**. Its daily turnover was lower than both momentum baselines. Maximum drawdown was 0.60 percentage points deeper than equal weight and shallower than both momentum baselines.

![Holdout equity curves for the strategy and three baselines](assets/benchmark-equity.png)

The equity chart contains all 841 holdout sessions, with starting capital normalized to one. Turnover uses traded portfolio value and includes entry, drift-aware rebalancing, and final liquidation.

## Risk, subperiods, and paired comparisons

![Return–volatility comparison, drawdowns, calendar periods and monthly baseline comparisons](assets/benchmark-diagnostics.png)

The strategy's annualized volatility was **15.71%**, compared with **12.18%** for equal weight, **13.86%** for 20-day momentum and **13.57%** for 60-day momentum. Its higher Sharpe therefore accompanies more daily variability. The complete drawdown panel preserves all troughs rather than selecting a favorable subperiod.

| Calendar period | AlphaResearchOS | Equal weight | 20-day momentum | 60-day momentum |
| --- | ---: | ---: | ---: | ---: |
| 2023-05-11 → 2023-12-29 · partial year | 7.87% | 10.53% | **11.38%** | 9.80% |
| 2024 · full year | **26.97%** | 15.70% | 19.16% | 16.02% |
| 2025 · full year | **20.80%** | 13.50% | 4.20% | 8.43% |
| 2026-01-02 → 2026-09-17 · year to date | **31.48%** | 12.46% | 14.41% | 19.92% |

All calendar-period figures are compounded net returns, not annualized rates. The strategy trailed every baseline in the partial 2023 period and led each baseline in the three subsequent periods. Partial years should not be compared as equal-length samples.

For a finer comparison, the table below pairs the strategy and each baseline in the **same 39 complete calendar months, June 2023–August 2026**. The two partial boundary months are excluded from this calculation only; the headline return and drawdown still include all 841 sessions.

| Paired baseline | Months strategy returned more | Fraction | Median monthly return difference |
| --- | ---: | ---: | ---: |
| Equal weight | 28 / 39 | 71.8% | +0.714 pp |
| 20-day momentum | 22 / 39 | 56.4% | +0.616 pp |
| 60-day momentum | 23 / 39 | 59.0% | +0.648 pp |

There are no ties. These are descriptive counts, not statistical significance tests or trade win rates. A month counts as ahead only when its net return strictly exceeds the paired baseline; a negative month may still count if the baseline lost more.

## Cost sensitivity

The same selected strategy was evaluated at four cost levels. Each comparison below uses the equal-weight return at that same cost.

![Cost-matched returns and portfolio turnover](assets/benchmark-frictions.png)

| One-way cost | Strategy return | Strategy CAGR | Sharpe | Max drawdown | Equal-weight return | Return difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bps | 126.72% | 27.80% | 1.6403 | -16.03% | 63.91% | +62.81 pp |
| 10 bps | 117.53% | 26.22% | 1.5608 | -16.21% | 63.22% | +54.32 pp |
| 20 bps | 108.72% | 24.67% | 1.4809 | -16.39% | 62.52% | +46.20 pp |
| 40 bps | 92.15% | 21.62% | 1.3202 | -16.74% | 61.15% | +31.00 pp |

At 40 bps, the strategy retained a **31.00-percentage-point** return advantage over equal weight. These cost scenarios were evaluated after candidate selection.

At 10 bps, average daily turnover was **4.92%**, **62.50% below 20-day momentum** and **37.39% below 60-day momentum**, but above equal weight's **0.51%**. Only AlphaResearchOS and equal weight have stored cost-sensitivity scenarios; no unstored momentum cost curves are implied. Capacity, market impact and live fills were not evaluated by these portfolio curves.

## Strategy

The selected Ridge model combines **smoothed intraday range** with **medium-term trend efficiency**:

```text
0 - mean((high - low) / (close + 1e-06), 60)

delay(ret(close, 60), 5) /
(60 * mean(delay(abs(ret(close, 1)), 5), 60) + 1e-06)
```

```json
{"alpha":100.0,"train_window":756,"retrain_every":63,"horizon":10,"smoothing":15}
```

The model inputs are daily cross-sectional percentile ranks. Feature standardization and Ridge coefficients are fitted on past training samples. The final holdout fit uses a 756-session window with 6,804 pooled samples, ending on 2023-05-08. Predictions throughout the holdout use that fixed fit.

## Development and selection

Candidate selection uses the mean fold Sharpe, fold stability, return difference against equal weight, and a complexity penalty. The selected strategy's final development score was **0.216635**.

| Development fold | Strategy return | Equal-weight return | Difference | Strategy Sharpe | Strategy max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2011-01-03 → 2015-02-12 | 42.38% | 77.56% | -35.18 pp | 0.5592 | -28.74% |
| 2015-02-13 → 2019-03-27 | 43.22% | 38.11% | +5.10 pp | 0.7684 | -14.36% |
| 2019-03-28 → 2023-05-08 | 49.08% | 58.99% | -9.90 pp | 0.5072 | -45.18% |

Across the chained development folds, strategy return was 203.99% and equal-weight return was 289.88%. The research-candidate pool selected its highest-scoring approved strategy; fixed baselines remained comparison portfolios. Model fitting was held fixed before each fold, and final holdout evaluation followed selection.

### Candidate history

| Candidate | Model / features | Development score | Outcome |
| --- | --- | ---: | --- |
| T0001 | rank / 3 | 0.078570 | Review requested a revision to align the downside-risk hypothesis and feature |
| T0002 | ridge / 3 | -0.002622 | Approved |
| T0003 | hist_gbdt / 2 | -0.146468 | Approved |
| T0004 | Proposal timeout | — | Failed; request counted |
| T0005 | hist_gbdt / 1 | -0.375349 | Approved; remove short-term deviation |
| **T0006** | **ridge / 2** | **0.216635** | **Approved and selected** |
| T0007 | ridge / 1 | -0.110036 | Approved; remove the separate range feature |
| T0008 | ridge / 2 | 0.141268 | Approved; add downside semivariance |

Seven strategies received reviews, with six approvals and one revision decision. One proposal timed out. The campaign used 15 model requests; 14 returned usage records totaling 256,147 tokens. All 14 candidate–feature entries passed the numerical and causality checks. The review-driven sequence includes a direct ablation of the selected strategy's range feature in T0007.

## Chart data

The compact benchmark files contain portfolio metrics, strategy settings, cost scenarios, development results, and the 841 daily portfolio curves:

- [results.json](../benchmarks/sector-etf/results.json)
- [curves.csv](../benchmarks/sector-etf/curves.csv)
- [plot_benchmark.py](../benchmarks/sector-etf/plot_benchmark.py)
- [analysis_metrics.json](../benchmarks/sector-etf/analysis_metrics.json) — derived calendar returns, paired-month comparisons, definitions and original-file SHA-256 hashes
- [plot_diagnostics.py](../benchmarks/sector-etf/plot_diagnostics.py) — diagnostics and cost/turnover figures

Render the figures locally without downloading data or calling a model:

```bash
uv run --locked --extra plot python benchmarks/sector-etf/plot_benchmark.py
uv run --locked --extra plot python benchmarks/sector-etf/plot_diagnostics.py
```

Both scripts validate the headline metrics against the daily curves. The diagnostics additionally verify that compounded calendar returns reconcile to the total return. Outputs include PNG and editable SVG figures in `docs/assets`; the new diagnostics script also writes `analysis_metrics.json`. Neither script overwrites the original `results.json` or `curves.csv`.

Daily returns are `equity[t] / equity[t-1] - 1`, with initial equity equal to one. Calendar returns compound those daily returns within the period. Drawdown divides by the running peak including initial cash. Volatility uses sample standard deviation and 252-session annualization. These definitions preserve initial trading losses and final liquidation costs.

The reporting hierarchy draws on [QuantStats' report implementation](https://github.com/ranaroussi/quantstats/blob/main/quantstats/reports.py) and [pyfolio's returns tearsheet example](https://quantopian.github.io/pyfolio/notebooks/single_stock_example/): examine aggregate performance, the full risk path and subperiod consistency together. No code, example performance data, or charts were copied from those projects. [Research methods](METHODS.md) describe feature evaluation, fitting, selection, and transaction accounting.
