<div align="center">

<h1>AlphaResearchOS</h1>
<p><strong>Agentic Quant Research</strong></p>
<p>Turn market data into research hypotheses, predictive models, and auditable strategies.</p>

<p><strong>English</strong> &nbsp;|&nbsp; <a href="README.zh-CN.md">简体中文</a></p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11%E2%80%933.13-3776AB?logo=python&logoColor=white" alt="Python 3.11–3.13">
  <img src="https://img.shields.io/badge/Models-Codex%20%2B%20API-16877B" alt="Codex and compatible APIs">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-64748B" alt="MIT License"></a>
</p>

<p><a href="#quick-start">Quick start</a> · <a href="#benchmark">Benchmark</a> · <a href="docs/USER_GUIDE.md">User guide</a></p>

</div>

AlphaResearchOS brings the quantitative research loop into one local workbench. Import a standard OHLCV CSV, connect Codex or a compatible model API, and describe the strategy you want to investigate.

**Hypothesis → Features + model → Development evaluation → Independent review → Iteration → Holdout**

![AlphaResearchOS workbench](docs/assets/workbench.png)

## Features

| Research | Review | Analyze |
| --- | --- | --- |
| Generate hypotheses and 1–6 feature expressions | Review each valid candidate in a fresh model session | Compare strategies with equal-weight and momentum baselines |
| Compare rank ensembles, Ridge, and histogram gradient boosting | Inspect parent trajectories, rationale, and follow-up experiments | Explore equity, drawdown, turnover, and cost sensitivity |
| Guide search with development memory and UCB parent selection | Set request, candidate, time, and token-admission budgets | Manage a factor library and export complete research reports |

Choose a **dark or light theme** from the top bar. The workbench follows your system on first use and remembers your choice.

<details>
<summary>Dark and light theme preview</summary>

Interface preview with synthetic demonstration data.

![Dark theme](docs/assets/workbench-dark.png)

![Light theme](docs/assets/workbench-light.png)

</details>

## Quick start

Python 3.11–3.13 · macOS / Linux · Windows through WSL2

```bash
uv sync --locked
uv run alphaos serve
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765/), then:

1. **Inspect and import your data.** Click **Import CSV / 导入 CSV**, review coverage and validation results, then confirm the import.
2. **Connect a model.** Choose your authenticated **Codex CLI**, a DeepSeek/Gemini/Ollama API preset, or a custom compatible endpoint. Enter a model ID, save, and test its structured response.
3. **Start research.** Select the imported dataset, enter a direction and budget, and follow candidates as they are proposed, trained, reviewed, and evaluated.

Your CSV uses these columns:

```csv
date,symbol,open,high,low,close,volume
```

Use UTF-8, **3–100 assets**, aligned daily observations, and consistently adjusted prices. Research with the default settings needs **at least 468 common trading days**; the importer accepts datasets from 400 days. Each date–asset pair must be unique. See the [data guide](docs/USER_GUIDE.md#standard-csv).

New research defaults to **6 candidates, 12 model requests, and 900 seconds**. The workbench keeps the features, fitted-model settings, review decisions, and development results together. [Model connections and controls →](docs/USER_GUIDE.md#model-connections)

## Benchmark

On **nine US sector ETFs**, AlphaResearchOS's selected two-feature Ridge strategy achieved **117.53% total return**, **26.22% CAGR**, and a **1.5608 Sharpe ratio** over **2023-05-11 to 2026-09-17**, after **10 bps one-way transaction costs**.

The dataset spans **2010-01-04 to 2026-09-17**. Model training and candidate selection use history through **2023-05-08**; the table and charts below cover the subsequent 841-session holdout. Strategies rebalance every five sessions. AlphaResearchOS and the momentum baselines hold the top three assets; equal weight holds all nine.

![Sector ETF holdout equity curves](docs/assets/benchmark-equity.png)

| Strategy | Total return | CAGR | Sharpe | Max drawdown | Annual volatility |
| --- | ---: | ---: | ---: | ---: | ---: |
| **AlphaResearchOS · Ridge** | **117.53%** | **26.22%** | **1.5608** | -16.21% | 15.71% |
| Equal weight | 63.22% | 15.81% | 1.2662 | **-15.61%** | **12.18%** |
| 20-day momentum | 58.22% | 14.74% | 1.0615 | -21.21% | 13.86% |
| 60-day momentum | 65.63% | 16.32% | 1.1823 | -18.14% | 13.57% |

The return lead was **+54.32 pp over equal weight**, **+59.31 pp over 20-day momentum**, and **+51.90 pp over 60-day momentum**. Sharpe was higher than all three baselines, alongside higher volatility; maximum drawdown was 0.60 pp deeper than equal weight.

![Return and volatility, full drawdowns, calendar-period returns, and paired monthly comparisons](docs/assets/benchmark-diagnostics.png)

Across **39 complete months**, the strategy beat equal weight in **28 months (71.8%)**, 20-day momentum in **22 (56.4%)**, and 60-day momentum in **23 (59.0%)**. It led all three in 2024, 2025 and 2026 through September 17, but trailed them in the partial 2023 window. The chart preserves every year and the full drawdown path.

<details>
<summary><strong>Implementation: costs and turnover</strong></summary>

At **40 bps one-way costs**, the strategy returned **92.15%**, compared with **61.15%** for cost-matched equal weight. At the 10 bps benchmark setting, average daily turnover was **4.92%**: **62.50% lower** than 20-day momentum and **37.39% lower** than 60-day momentum, but above equal weight's **0.51%**.

![Cost-matched returns and daily portfolio turnover](docs/assets/benchmark-frictions.png)

</details>

[Methods, development results, and chart reproduction](docs/BENCHMARK.md) · [Immutable benchmark data](benchmarks/sector-etf/results.json) · [Derived diagnostics](benchmarks/sector-etf/analysis_metrics.json)

## The research process

Features run through a bounded OHLCV expression language. Predictive training uses matured labels, with model fitting fixed before each development fold and the final holdout. A separate model session reviews development evidence and suggests the next experiment. Approved candidates compete by development score before the final strategy is evaluated.

The workbench supports executable window, field, and turnover constraints, plus pause/resume and a searchable factor library. [Research methods →](docs/METHODS.md)

## Development

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check src/alpharesearchos scripts
node --check src/alpharesearchos/static/app.js
uv build
```

[Contributing](CONTRIBUTING.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

## License

[MIT](LICENSE). Third-party attribution is maintained in [NOTICE](NOTICE.md), with numerical component details in [UPSTREAM.md](docs/UPSTREAM.md).
