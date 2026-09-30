"""Protocol-level tests: frozen inputs, restart parity, and honest failures."""

import importlib.util
import json
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from alpharesearchos import cli, engine
from alpharesearchos.backtest import BacktestConfig, backtest, summarize
from alpharesearchos.config import ResearchConfig
from alpharesearchos.data import FIELDS, load_csv, make_demo, save_panel, to_long
from alpharesearchos.factors import evaluate_expression
from alpharesearchos.store import atomic_json, read_json, run_lock


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "prices.csv"
    save_panel(make_demo(seed=17, sessions=400, assets=4), path)
    return path


def _run(directory, **kwargs):
    # During parallel development the renderer may not exist yet. This never
    # stubs evaluation, persistence, selection, or final-holdout computation.
    renderer_ready = importlib.util.find_spec("alpharesearchos.reports") is not None
    context = nullcontext() if renderer_ready else patch.object(engine, "_export", lambda *_: None)
    with context:
        return engine.run_research(directory, **kwargs)


def _config(**overrides):
    values = {"trials": 4, "seed": 17, "warmup": 64, "folds": 2, "top_k": 2}
    values.update(overrides)
    return ResearchConfig(**values)


def test_demo_seed_and_csv_snapshot_are_reproducible(csv_path):
    first = make_demo(seed=17, sessions=400, assets=4)
    second = make_demo(seed=17, sessions=400, assets=4)
    for field in FIELDS:
        pd.testing.assert_frame_equal(first[field], second[field], check_exact=True)
        np.testing.assert_allclose(load_csv(csv_path)[field], first[field], rtol=1e-10)
    assert not first["close"].equals(make_demo(seed=18, sessions=400, assets=4)["close"])


@pytest.mark.parametrize("problem", ["missing_cell", "missing_price", "duplicate", "bad_date", "intraday", "bad_high", "negative_volume", "missing_column"])
def test_csv_rejects_malformed_and_missing_observations(tmp_path, problem):
    frame = to_long(make_demo(seed=17, sessions=400, assets=4))
    if problem == "missing_cell":
        frame = frame.iloc[1:]
    elif problem == "missing_price":
        frame.loc[0, "close"] = np.nan
    elif problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif problem == "bad_date":
        frame["date"] = frame["date"].astype(str)
        frame.loc[4, "date"] = "not-a-date"
    elif problem == "intraday":
        frame["date"] = frame["date"].astype(str)
        frame.loc[0, "date"] = "2020-01-02 12:00:00"
    elif problem == "bad_high":
        frame.loc[0, "high"] = 0.5 * frame.loc[0, "close"]
    elif problem == "negative_volume":
        frame.loc[0, "volume"] = -1
    else:
        frame = frame.drop(columns="volume")
    path = tmp_path / f"{problem}.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_csv(path)


@pytest.mark.parametrize("kwargs", [
    {"trials": True}, {"trials": 2.5}, {"trials": 0}, {"seed": -1},
    {"folds": 1}, {"top_k": 0}, {"rebalance_every": 0},
    {"cost_bps": True}, {"cost_bps": "10"}, {"cost_bps": float("nan")},
    {"holdout_fraction": float("inf")}, {"max_llm_calls": -1},
    {"max_llm_tokens": 500001}, {"mode": "unknown"}, {"direction": "   "},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)


def test_atomic_checkpoint_retains_old_json_after_replace_failure(tmp_path):
    destination = tmp_path / "checkpoint.json"
    atomic_json(destination, {"completed": 2})
    with patch("alpharesearchos.store.os.replace", side_effect=OSError("simulated disk failure")):
        with pytest.raises(OSError, match="simulated disk failure"):
            atomic_json(destination, {"completed": 3})
    assert read_json(destination) == {"completed": 2}
    assert list(tmp_path.glob(".checkpoint-*.tmp")) == []
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(tmp_path):
                pass


@pytest.mark.parametrize("pause_reason", ["checkpoint", "user_pause"])
def test_stop_resume_matches_uninterrupted_research(csv_path, tmp_path, monkeypatch, pause_reason):
    resumed_dir = engine.create_run(tmp_path / "runs", _config(), csv_path)
    options = {"stop_after": 2} if pause_reason == "checkpoint" else {
        "should_pause": lambda: len(read_json(resumed_dir / "report.json")["trials"]) >= 2}
    paused = _run(resumed_dir, **options)
    assert paused["status"] == "paused"
    assert paused["stop_reason"] == pause_reason
    assert paused["selected"] is None and paused["holdout"] is None
    assert paused["progress"]["completed"] == 2
    assert read_json(resumed_dir / "report.json")["_state"]["pending"] is None
    propose = engine.local_proposal

    def checked_proposal(*args, **kwargs):
        persisted = read_json(resumed_dir / "report.json")
        assert persisted["status"] == "running" and persisted["stop_reason"] is None
        return propose(*args, **kwargs)

    with monkeypatch.context() as resumed_checks:
        resumed_checks.setattr(engine, "local_proposal", checked_proposal)
        resumed = _run(resumed_dir)
    direct_dir = engine.create_run(tmp_path / "runs", _config(), csv_path)
    direct = _run(direct_dir)
    assert resumed["status"] == direct["status"] == "completed"
    assert resumed["stop_reason"] == direct["stop_reason"] == "trial_budget"
    keys = ["id", "expression", "status", "score", "metrics", "reason"]
    assert [{key: trial.get(key) for key in keys} for trial in resumed["trials"]] == [
        {key: trial.get(key) for key in keys} for trial in direct["trials"]
    ]
    assert resumed["selected"]["expression"] == direct["selected"]["expression"]
    assert resumed["holdout"] == direct["holdout"]
    # Completed calls are idempotent: no additional trials or holdout searching.
    assert _run(resumed_dir) == resumed


def test_snapshot_tampering_rejected_before_research(csv_path, tmp_path):
    directory = engine.create_run(tmp_path / "runs", _config(trials=1), csv_path)
    with (directory / "snapshot.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum changed"):
        _run(directory)
    assert read_json(directory / "report.json")["progress"]["completed"] == 0


def test_interrupted_attempt_consumes_budget_without_retry(csv_path, tmp_path):
    directory = engine.create_run(tmp_path / "runs", _config(trials=2), csv_path)
    paused = _run(directory, stop_after=1)
    paused["_state"]["pending"] = {"id": "T0002", "started_at": "interrupted"}
    paused["_state"]["llm_calls"] = 1
    paused["_state"]["llm_reserved_tokens"] = 22700
    atomic_json(directory / "report.json", paused)
    with patch.object(engine, "local_proposal", side_effect=AssertionError("Attempt was retried")):
        resumed = _run(directory)
    assert resumed["status"] == "completed"
    assert len(resumed["trials"]) == 2
    assert resumed["trials"][1]["id"] == "T0002"
    assert resumed["trials"][1]["status"] == "failed"
    assert resumed["_state"]["llm_calls"] == 1
    assert resumed["_state"]["llm_reserved_tokens"] == 22700


def test_future_holdout_prices_cannot_change_development_selection(csv_path, tmp_path):
    original = _run(engine.create_run(tmp_path / "runs", _config(), csv_path))
    changed = load_csv(csv_path)
    start = original["_state"]["holdout_start"]
    multiplier = np.exp(0.001 * np.arange(1, len(changed["close"]) - start + 1))[:, None]
    for field in ("open", "high", "low", "close"):
        changed[field].iloc[start:] *= multiplier
    changed_path = tmp_path / "changed-holdout.csv"
    save_panel(changed, changed_path)
    altered = _run(engine.create_run(tmp_path / "runs", _config(), changed_path))
    assert original["selected"]["expression"] == altered["selected"]["expression"]
    assert [(t["expression"], t["status"], t.get("score"), t.get("metrics")) for t in original["trials"]] == [
        (t["expression"], t["status"], t.get("score"), t.get("metrics")) for t in altered["trials"]
    ]
    assert original["holdout"]["metrics"]["total_return"] != altered["holdout"]["metrics"]["total_return"]


def test_all_rejected_candidates_produce_failed_not_completed_run(csv_path, tmp_path):
    candidates = [
        {"name": "constant", "hypothesis": "invalid", "expression": "1", "origin": "test"},
        {"name": "future", "hypothesis": "invalid", "expression": "delay(close, -1)", "origin": "test"},
        {"name": "undefined", "hypothesis": "invalid", "expression": "close / 0", "origin": "test"},
    ]
    directory = engine.create_run(tmp_path / "runs", _config(trials=3), csv_path, candidates)
    report = _run(directory)
    assert report["status"] == "failed"
    assert report["stop_reason"] == "no_valid_candidates"
    assert report["selected"] is None and report["holdout"] is None
    assert len(report["trials"]) == 3
    assert all(trial["status"] == "rejected" for trial in report["trials"])


def test_frozen_holdout_replay_matches_saved_metrics(csv_path, tmp_path):
    directory = engine.create_run(tmp_path / "runs", _config(trials=2), csv_path)
    report = _run(directory)
    panel = load_csv(directory / "snapshot.csv")
    scores = evaluate_expression(report["selected"]["expression"], panel)
    curve = backtest(scores, panel["close"], BacktestConfig(top_k=2), report["_state"]["holdout_start"], len(panel["close"]))
    assert summarize(curve) == report["holdout"]["metrics"]
    if importlib.util.find_spec("alpharesearchos.reports") is not None:
        assert engine.verify_run(directory)["ok"]
        assert cli.main(["replay", str(directory)]) == 0
        with (directory / "holdout_curve.csv").open("a") as stream:
            stream.write("\n")
        assert not engine.verify_run(directory)["ok"]
        assert cli.main(["replay", str(directory)]) == 2


def test_cli_returns_nonzero_for_invalid_inputs_and_failed_search(csv_path, tmp_path, capsys):
    assert cli.main(["run", "--csv", str(csv_path), "--trials", "0"]) == 2
    assert cli.main(["run", "--csv", str(tmp_path / "absent.csv")]) == 2
    # Test orchestration failure separately from validation: all valid syntax,
    # but the candidate has no cross-sectional variation and must be rejected.
    candidates = tmp_path / "constant.json"
    candidates.write_text(json.dumps([{"expression": "1"}]))
    renderer_ready = importlib.util.find_spec("alpharesearchos.reports") is not None
    context = nullcontext() if renderer_ready else patch.object(engine, "_export", lambda *_: None)
    with context:
        assert cli.main(["run", "--mode", "local", "--csv", str(csv_path), "--runs", str(tmp_path / "cli-runs"), "--trials", "1", "--candidates", str(candidates)]) == 1
    captured = capsys.readouterr()
    assert "alphaos:" in captured.err
    assert '"status": "failed"' in captured.out
