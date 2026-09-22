"""Independent regression audits for accounting, isolation and paid budgets."""
import numpy as np
import pandas as pd
import pytest

from alpharesearchos import engine
from alpharesearchos.config import ResearchConfig
from alpharesearchos.data import make_demo
from alpharesearchos.store import read_json


@pytest.fixture
def small_engine(monkeypatch):
    panel = make_demo(seed=55, sessions=450, assets=5)
    monkeypatch.setattr(engine, "make_demo", lambda **kwargs: {k: v.copy() for k, v in panel.items()})
    # Rendering is unrelated to the accounting/isolation invariant under test.
    monkeypatch.setattr(engine, "_export", lambda *args: None)
    return panel


def config(**kwargs):
    return ResearchConfig(trials=kwargs.pop("trials", 2), warmup=64, folds=2, max_seconds=60, **kwargs)


def test_imported_candidates_do_not_consume_llm_budget(tmp_path, monkeypatch, small_engine):
    monkeypatch.setattr(engine, "llm_configured", lambda: True)

    def unexpected_request(*args, **kwargs):
        pytest.fail("Imported candidate should not call the LLM")

    monkeypatch.setattr(engine, "llm_proposal", unexpected_request)
    candidates = [{"name": "imported", "hypothesis": "audit", "expression": expression,
                   "origin": "imported", "parents": []} for expression in
                  ["rank(ret(close, 20))", "-rank(ret(close, 5))", "rank(volume)"]]
    directory = engine.create_run(tmp_path, config(trials=3, mode="llm", max_llm_tokens=24000), candidates=candidates)
    result = engine.run_research(directory)
    assert result["_state"]["llm_calls"] == 0
    assert result["_state"]["llm_reserved_tokens"] == 0


@pytest.mark.parametrize("budget,expected_calls", [(65535, 0), (196608, 1)])
def test_cli_admission_and_reported_overrun_stop_further_calls(tmp_path, monkeypatch, small_engine, budget, expected_calls):
    monkeypatch.setattr(engine, "llm_configured", lambda: True)
    monkeypatch.setattr(engine, "llm_token_reservation", lambda: 65536)
    calls = []

    def proposal(*args, **kwargs):
        calls.append(1)
        return ({"name": "CLI proposal", "hypothesis": "test", "expression": "rank(volume)",
                 "origin": "codex_cli", "parents": []}, {"reported_total_tokens": 200000})

    monkeypatch.setattr(engine, "llm_proposal", proposal)
    directory = engine.create_run(tmp_path, config(trials=5, mode="llm", max_llm_calls=3, max_llm_tokens=budget))
    result = engine.run_research(directory)
    assert len(calls) == expected_calls
    assert result["_state"]["llm_calls"] == expected_calls
    assert result["_state"]["llm_reserved_tokens"] == (200000 if expected_calls else 0)
    assert all(trial["origin"] == "local_budget_fallback" for trial in result["trials"][3:])


def test_benchmark_summary_uses_benchmark_turnover(tmp_path, small_engine):
    directory = engine.create_run(tmp_path, config(trials=1))
    report = engine.run_research(directory)
    curve = pd.read_csv(directory / "holdout_curve.csv")
    assert report["holdout"]["baseline_metrics"]["avg_turnover"] == pytest.approx(curve["benchmark_turnover"].mean())


def test_holdout_perturbation_cannot_change_development_selection(tmp_path, monkeypatch, small_engine):
    first_dir = engine.create_run(tmp_path, config())
    first = engine.run_research(first_dir)
    cutoff = first["_state"]["holdout_start"]
    changed = {key: value.copy() for key, value in small_engine.items()}
    n, assets = changed["close"].iloc[cutoff:].shape
    # Same per-cell multiplier for all OHLC fields preserves valid price bars.
    shift = np.exp(np.linspace(0, 2, n)[:, None] * np.linspace(-1, 1, assets)[None, :])
    for field in ["open", "high", "low", "close"]:
        changed[field].iloc[cutoff:] *= shift
    monkeypatch.setattr(engine, "make_demo", lambda **kwargs: changed)
    second = engine.run_research(engine.create_run(tmp_path, config()))
    assert second["selected"]["expression"] == first["selected"]["expression"]
    assert [t["score"] for t in second["trials"]] == [t["score"] for t in first["trials"]]
    assert second["holdout"]["metrics"]["total_return"] != first["holdout"]["metrics"]["total_return"]


def test_resume_after_holdout_failure_preserves_frozen_selection(tmp_path, monkeypatch, small_engine):
    directory = engine.create_run(tmp_path, config())
    real_backtest = engine.backtest

    def crash_final_test(*args, **kwargs):
        frozen = read_json(directory / "report.json")
        assert frozen["selected"] is not None
        assert frozen["selection_frozen_at"]
        raise RuntimeError("injected holdout interruption")

    monkeypatch.setattr(engine, "backtest", crash_final_test)
    with pytest.raises(RuntimeError, match="injected holdout"):
        engine.run_research(directory)
    frozen = read_json(directory / "report.json")
    monkeypatch.setattr(engine, "backtest", real_backtest)

    def forbidden_new_research(*args, **kwargs):
        pytest.fail("Resume after selection must not ask for another candidate")

    monkeypatch.setattr(engine, "local_proposal", forbidden_new_research)
    resumed = engine.run_research(directory)
    assert resumed["status"] == "completed"
    assert resumed["selected"] == frozen["selected"]
    assert resumed["trials"] == frozen["trials"]
    assert resumed["selection_frozen_at"] == frozen["selection_frozen_at"]
