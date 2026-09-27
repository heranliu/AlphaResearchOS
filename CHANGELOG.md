# Changelog

## 0.5.0

- Optional, default-off Jev development-evidence gate after independent review,
  with typed decisions, four probability checks, configurable thresholds and
  model-version tracking. Uncertain, failed or interrupted gates cannot pass selection.
- Separate private Jev settings and explicit connection test; shared request,
  time and token-admission budgets with checkpointed attempts and no retries.
- Jev decisions and usage in the workbench, JSON/CSV and offline reports.
- Layered benchmark diagnostics from unchanged pre-Jev evidence: drawdowns,
  risk/return, calendar periods, complete-month baseline comparisons and costs.
  These historical results do not measure Jev's incremental performance.

## 0.4.0

- Model-driven research trajectories combining hypotheses, factor expressions,
  predictive models, parameters, development metrics, and review decisions.
- Rank, Ridge, and histogram gradient boosting evaluation with recorded training
  cutoffs and mature-label checks.
- Separate proposer and reviewer requests, candidate lineage, SQLite research
  memory, bounded calls, checkpointing, and run recovery.
- Local Codex CLI and OpenAI-compatible model connections configured from the Web.
- Research, factor library, fixed backtests, and model settings in one workbench.
- Sector-ETF benchmark with return, risk, turnover, equity, and transaction-cost
  comparisons against equal weight and two momentum baselines.
- English and Chinese introductions, real interface screenshots, benchmark
  charts, and a source-release packaging workflow.
- Web CSV import with schema validation, actionable errors, and automatic dataset selection.
