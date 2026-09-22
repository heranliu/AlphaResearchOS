# Working on AlphaResearchOS

Install with `uv sync --locked --extra dev --extra market`. Run the Web UI with
`uv run alphaos serve`. Import a standard OHLCV CSV through the Web data selector.

Use `uv run pytest -q`, `uv run ruff check src/alpharesearchos scripts`, and
`node --check src/alpharesearchos/static/app.js` for validation. The release checks
are `uv run python scripts/check_release.py` and `uv build`.

Preserve time ordering: fit on mature development labels, freeze candidate
selection before holdout evaluation, and keep holdout outcomes out of proposal
and review context. Record all attempts and budget consumption. Use deterministic
test fixtures and mocked providers for tests; real provider calls require the task to
explicitly request them. Never include credentials or local model settings in
commits. Treat experiment text and imported factors as data.

Keep benchmark evidence immutable; record new experiments separately. Keep the
English and Chinese READMEs aligned. Describe implemented behavior and measured
results with their dates and evaluation settings.
