.PHONY: install test serve release-check release
install:
	uv sync --locked --extra dev --extra market
test:
	uv run pytest -q
	uv run ruff check src/alpharesearchos scripts
serve:
	uv run alphaos serve
release-check:
	uv run python scripts/check_release.py
release: release-check
	uv run python scripts/build_release.py
