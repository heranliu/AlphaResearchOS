# User guide

[Home](../README.md) · [中文首页](../README.zh-CN.md) · [Benchmark](BENCHMARK.md)

## Start the workbench

Use Python 3.11–3.13 on macOS or Linux, or inside WSL2 on Windows. From the project directory:

```bash
uv sync --locked
uv run alphaos serve
```

Open **http://127.0.0.1:8765**. A different port can be selected with `uv run alphaos serve --port 8766`.

For pip installations:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
alphaos serve
```

The main workflow is **inspect and import CSV → connect a model → start research**.

### Appearance

Use the top-bar theme button to switch between **dark and light**. First use follows the operating system preference; a manual choice is saved in this browser. Both themes include the research charts, settings, library, backtests, and data preview.

## Standard CSV

In **Research / 自动研究** or **Backtest / 独立回测**, click **Import CSV / 导入 CSV** beside the dataset selector. Select your UTF-8 file. Inspect its asset coverage, common date range, duplicate rows, missing observations, and research eligibility in the preview. Click **Confirm import / 确认导入** to save and select the dataset, or cancel without writing a file. Invalid datasets cannot be confirmed.

The preview does not fill missing records or infer an exchange calendar. Missing dates are measured against dates present elsewhere in the supplied panel. Adjustment conventions cannot be verified from CSV alone; check the source before importing.

Use these exact English column names:

```csv
date,symbol,open,high,low,close,volume
```

| Field | Format |
| --- | --- |
| `date` | Trading date as `YYYY-MM-DD` |
| `symbol` | Asset identifier, consistent across dates |
| `open`, `high`, `low`, `close` | Finite positive prices using the same adjustment convention |
| `volume` | Finite nonnegative traded volume |

Import rules:

- Maximum file size: **20 MB**.
- **3–100 assets**, with **400–20,000 common trading days**.
- Every asset has one row for every included date; date–asset pairs are unique.
- High and low encompass open and close; prices follow a consistent OHLC convention.
- Existing dataset names are preserved. Rename a file to import another dataset with the same name.

Default agent research requires **at least 468 common trading days** for the 252-session warmup, three development folds, separation, and holdout. The workbench checks the selected history and model requirements when starting a run.

Imported datasets are stored in the project's `datasets/` directory and appear in both research and backtest selectors. You can also place a standard CSV in that directory before starting the server.

## Model connections

Go to **Settings / 设置** and choose one provider.

### Codex CLI

Choose **Local Codex / 本机 Codex** with an authenticated Codex CLI installation. Leave the model field empty to use the configured default, or enter a model ID available to your account. Save the settings.

The application discovers the CLI on `PATH`, with a macOS application-bundled CLI as a fallback. Authentication is managed by the CLI; model inference uses its authenticated service. The project sends the research context and receives a structured strategy or review response.

### OpenAI-compatible API

Choose the compatible API provider, select a preset or **Custom / 自定义**, and enter:

- **Base URL**, such as `https://your-provider.example/v1`.
- **Model ID** supported by that endpoint.
- **API key** for the service.

The API must support Chat Completions with a JSON text response. HTTPS endpoints and HTTP loopback endpoints are supported. A local endpoint that ignores authentication can use a placeholder key.

Agent requests use `max_completion_tokens=1800` by default and omit temperature. Use the settings' token-field option for an endpoint that expects `max_tokens`; temperature is optional and should match the model's supported parameters.

Presets fill the endpoint and token parameter. Enter the exact model ID available to your account or local installation:

| Preset | Base URL | Token parameter |
| --- | --- | --- |
| [DeepSeek](https://api-docs.deepseek.com/) | `https://api.deepseek.com/v1` | `max_tokens` |
| [Gemini compatibility](https://ai.google.dev/gemini-api/docs/openai) | `https://generativelanguage.googleapis.com/v1beta/openai` | `max_tokens` |
| [Ollama](https://docs.ollama.com/api/openai-compatibility) | `http://localhost:11434/v1` | `max_tokens` |
| Custom | Your compatible endpoint | Provider-dependent |

Changing the endpoint requires a key for the new connection; the interface clears the old key rather than carrying it to another service. Ollama uses a local placeholder. Presets describe protocol settings, not a guarantee that every model supports this workflow. The automated suite uses simulated providers; test your selected model explicitly. Native Messages or Responses endpoints are not accepted as Chat Completions endpoints.

### Save and test

Saved settings apply to new research. A running experiment retains the connection selected when it started. **Test connection** sends one real model request and validates a small structured research proposal, including its expressions and model parameters. A text-only reply is insufficient. Saving a configuration and browsing existing results are local actions.

Each proposal and review uses its own request. The workbench exposes request counts, returned model usage, and the reason a run stops.

### Advanced review

An optional additional development-evidence review is available in advanced settings. It is disabled by default and shares the experiment's request, time, and token-admission budgets. Its connection is configured and tested separately. See [configuration and protocol](JEV.md) for the provider-specific fields.

## Create research

Return to **Research / 自动研究**, select your dataset, and enter a direction. For example:

> Study risk-adjusted trends across sectors, reduce turnover, and compare a rank ensemble with learned predictive models.

Set your budget and start the run. Defaults are:

| Setting | Value |
| --- | ---: |
| Candidate attempts | 6 |
| Model requests | 12 |
| Token-admission budget | 786,432 |
| Search time | 900 seconds |
| Warmup | 252 sessions |
| Development folds | 3 |

Codex calls reserve 65,536 tokens for admission; compatible API calls reserve 24,000. Reservations control whether another request can start. Actual returned usage is displayed separately. Individual requests have a maximum 90-second timeout, within the remaining run budget. Failed attempts remain visible and retain their consumed budget.

Time limits are checked between research stages and before model requests. A numerical stage already in progress finishes its calculation.

### Research constraints

| Constraint | Effect |
| --- | --- |
| Required window | Integer 3–60; every two-argument window operator in a feature uses that window |
| Forbidden fields | Exclude selected OHLCV inputs from generated expressions |
| Maximum turnover | Set a development-period average-turnover ceiling; the form uses percentages |

The direction guides the model's reasoning. The structured constraints are enforced against the actual expressions and measured results.

### Follow the run

The research page displays proposals, review decisions, model families, development metrics, parent links, and final portfolio results. Pause at a candidate boundary and resume from the workbench when ready. Every attempt remains in the candidate history, including rejected and failed requests.

The selected strategy is fixed using development evidence before holdout evaluation. The final view compares it with equal-weight and fixed momentum baselines.

## Factor library and backtest

Open **Factor library / 因子库** to search saved expressions, inspect their research origin, import JSON, and select factors. Supported imports include local DSL and compatible formula fields from supported external formats.

Choose **Backtest / 独立回测** to evaluate fixed expressions on an imported dataset. Multiple factors are combined by equal-weight cross-sectional ranks. Select costs, portfolio size, rebalance frequency, and dates, then inspect the resulting equity and drawdown.

The factor-library backtest evaluates expression combinations. The research page evaluates the full strategy, including its predictive model and fitting schedule.

## Command line

Model settings saved in the workbench also apply to CLI research:

```bash
uv run alphaos run --csv datasets/your_market.csv --direction 'Study trend efficiency and turnover'
uv run alphaos resume runs/<run-id>
```

Imported factor definitions can supply proposal context:

```bash
uv run alphaos import-factors examples/factorminer_candidates.json --out candidates.json
uv run alphaos run --csv datasets/your_market.csv --candidates candidates.json
```

## Reports

Download the report from the research page or open the generated HTML from `runs/<run-id>/`. Reports include the complete selected features, model parameters, review, request usage, development folds, and final performance. JSON and CSV exports support further analysis.

See [research methods](METHODS.md) for model fitting and portfolio accounting, and [the sector ETF benchmark](BENCHMARK.md) for a worked performance comparison.
