# Contributing

Contributions are welcome in English or Chinese. Start with a reproducible bug,
a focused user workflow, or a measurable experiment improvement.

## Development setup

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --locked --extra dev --extra market
uv run alphaos serve
```

Open `http://127.0.0.1:8765`, import a standard OHLCV CSV, and configure a model
connection in Settings to start research.

## Checks

```bash
uv run pytest -q
uv run ruff check src/alpharesearchos scripts
node --check src/alpharesearchos/static/app.js
uv run python scripts/check_release.py
uv build
```

Browser workflows and themes have a dedicated Chromium suite with a local model
HTTP fixture. Install and run it with:

```bash
uv sync --locked --extra dev --extra market --extra e2e
uv run playwright install chromium
uv run pytest -q tests/e2e
```

To check the distribution in a fresh virtual environment after `uv build`:

```bash
ALPHAOS_TEST_WHEEL=1 uv run pytest -q tests/test_distribution.py
```

This uses the dependencies already present in uv's cache and performs no live
model calls. The standard unit job excludes `tests/e2e`; the dedicated browser
job installs Chromium and runs the complete browser suite.

## Pull requests

Describe the user-visible result, the change, and how it was checked. Include a
screenshot for interface changes. For numerical changes, include a small causal
example and compare results at matching dates, transaction costs, and budgets.

Keep runs, downloaded datasets, model settings, and credentials in the ignored
workspace directories. Use deterministic fixtures in tests. Preserve the original
benchmark evidence; add a separately identified experiment for new results.

## Useful areas to work on

- Data provider adapters with explicit adjustment rules and actionable diagnostics.
- Research replay, experiment comparison, and navigable strategy lineage.
- Reviewer context containing parent metrics and training evidence.
- Market, time-window, and mechanism-ablation benchmark suites.
- Model-provider integrations and English interface translations.

Discuss changes to the research protocol or public API in an issue before a
large implementation. Small fixes can go directly into a pull request.
