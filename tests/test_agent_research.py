"""Agent orchestration against real causal predictors and synthetic prices.

Only model transport is substituted. No subprocess/network/model account is
used; strategy evaluation, accounting, persistence and exports remain real.
"""

import copy
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

from alpharesearchos import agent_research, codex_provider, engine, proposals, research_client
from alpharesearchos.backtest import BacktestConfig, backtest, summarize
from alpharesearchos.config import ResearchConfig
from alpharesearchos.data import load_csv, make_demo, save_panel
from alpharesearchos.predictive import strategy_scores
from alpharesearchos.research_memory import GraphMemory
from alpharesearchos.store import atomic_json, read_json


def strategy(model="rank"):
    features = {
        "rank": ["ret(close, 5)", "-std(ret(close, 1), 10)"],
        "ridge": ["volume / mean(volume, 5)", "ret(close, 3)"],
        "hist_gbdt": ["(high - low) / close", "ret(close, 10)"],
    }
    return {
        "name": f"审计 {model}", "hypothesis": "检验历史量价信息与可交易收益的关系",
        "rationale": "仅以开发期反馈比较机制", "features": features[model], "model": model,
        "model_params": {"alpha": 1.0, "train_window": 126, "retrain_every": 63, "horizon": 1, "smoothing": 2},
    }


class ModelRecorder:
    def __init__(self):
        self.calls = []
        self.decisions = {}
        self.proposal_error = None
        self.interrupt_review = False
        self.extra_proposal = {}
        self.reported_tokens = 37

    def __call__(self, instruction, context, schema, *, timeout):
        assert timeout > 0
        role = "propose" if "features" in schema["properties"] else "review"
        self.calls.append({"role": role, "context": copy.deepcopy(context)})
        if role == "propose":
            if self.proposal_error:
                raise self.proposal_error
            result = strategy(context["required_model"] or "rank")
            result.update(self.extra_proposal)
        else:
            if self.interrupt_review:
                self.interrupt_review = False
                raise KeyboardInterrupt
            result = {
                "decision": self.decisions.get(context["candidate"]["model"], "approve"),
                "critique": "依据提供的开发期指标进行技术复核", "risks": ["开发期选择偏差"],
                "suggested_action": "检验独立时期的稳定性",
            }
        return result, {"provider": "mock", "reported_total_tokens": self.reported_tokens}


@pytest.fixture
def model(monkeypatch):
    recorder = ModelRecorder()

    def forbidden(*args, **kwargs):
        pytest.fail("Agent tests must not call a live provider or local-proposal fallback")

    monkeypatch.setattr(codex_provider, "codex_status", lambda **kwargs: {
        "available": True, "authenticated": True, "version": "test-cli", "default_model": "test",
    })
    monkeypatch.setattr(codex_provider, "codex_structured", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(engine, "local_proposal", forbidden)
    monkeypatch.setattr(proposals, "local_proposal", forbidden)
    monkeypatch.setattr(research_client, "structured_call", recorder)
    monkeypatch.setattr(research_client, "reservation", lambda: 100)
    # Other agents may edit an unrelated renderer during the test. Source-change
    # rejection itself has separate tests; this fixture fixes that external input.
    monkeypatch.setattr(engine, "code_fingerprint", lambda: "agent-test-frozen-implementation")
    with proposals.provider_context({"provider": "codex_cli", "codex_model": "test", "api_key": "PROVIDER_SECRET_CANARY"}):
        yield recorder


@pytest.fixture
def market(tmp_path):
    path = tmp_path / "market.csv"
    save_panel(make_demo(seed=17, sessions=400, assets=4), path)
    return path


def config(**overrides):
    values = {"mode": "agent", "trials": 2, "seed": 17, "warmup": 128, "folds": 2,
              "top_k": 2, "max_complexity": 150, "max_llm_calls": 6, "max_llm_tokens": 600}
    return ResearchConfig(**(values | overrides))


@pytest.fixture
def jev(monkeypatch, model):
    from alpharesearchos import jev_client
    calls = []
    responses = {"decision": "approve", "error": None, "model": "jev-1.13.0"}

    def gate(context, *, settings, timeout):
        assert timeout > 0 and settings["jev_api_key"] == "JEV_SECRET_CANARY"
        calls.append(copy.deepcopy(context))
        if responses["error"]:
            raise responses["error"]
        decision = responses["decision"]
        return {"decision": decision, "confidence": .9, "threshold": .7,
                "probabilities": {key: .9 if key == decision else .05 for key in ["approve", "revise", "reject"]},
                "checks": {key: .85 for key in ["hypothesis_alignment", "cost_support", "fold_consistency", "evidence_sufficiency"]},
                "summary": "Recorded development-evidence judgment"}, {
                    "provider": "typesafe_jev", "model": responses["model"], "reported_total_tokens": 23}

    monkeypatch.setattr(jev_client, "review_gate", gate)
    monkeypatch.setattr(agent_research, "JEV_TOKEN_RESERVATION", 100)
    settings = {**proposals.llm_settings(), "jev_enabled": True, "jev_api_key": "JEV_SECRET_CANARY"}
    with proposals.provider_context(settings):
        yield calls, responses


def test_jev_additional_gate_is_counted_exported_and_has_no_holdout(tmp_path, market, model, jev):
    directory, report = run(tmp_path, market, settings=config(trials=1))
    assert report["status"] == "completed"
    assert len(model.calls) == 2 and len(jev[0]) == 1
    assert report["_state"]["llm_calls"] == 3 and report["_state"]["llm_reserved_tokens"] == 300
    assert report["research"]["budget"]["jev_calls"] == 1
    assert report["research"]["jev"]["actual_model"] == "jev-1.13.0"
    assert report["selected"]["jev_gate"]["decision"] == "approve"
    assert jev[0][0]["execution"]["cost_bps"] == 10
    assert jev[0][0]["review"]["decision"] == "approve"
    sent = json.dumps(jev[0])
    for canary in ["holdout", "JEV_SECRET_CANARY", "PROVIDER_SECRET_CANARY", str(tmp_path)]:
        assert canary not in sent
    for name in ["report.json", "trials.csv", "report.html", "report.md", "selected_candidate.json"]:
        body = (directory / name).read_text()
        assert "JEV_SECRET_CANARY" not in body and "PROVIDER_SECRET_CANARY" not in body
        assert "jev" in body.lower()


@pytest.mark.parametrize("decision", ["revise", "reject"])
def test_jev_veto_excludes_llm_approved_candidate(tmp_path, market, model, jev, decision):
    jev[1]["decision"] = decision
    _, report = run(tmp_path, market, settings=config(trials=1))
    trial = report["trials"][0]
    assert trial["review"]["decision"] == "approve" and trial["status"] == "rejected"
    assert trial["jev_gate"]["decision"] == decision
    assert report["selected"] is None and report["holdout"] is None


def test_reviewer_rejection_does_not_call_jev(tmp_path, market, model, jev):
    model.decisions["rank"] = "reject"
    _, report = run(tmp_path, market, settings=config(trials=1))
    assert jev[0] == [] and report["_state"]["llm_calls"] == 2
    assert report["selected"] is None


@pytest.mark.parametrize("overrides", [{"max_llm_calls": 2}, {"max_llm_tokens": 299}])
def test_jev_requires_budget_for_all_three_requests_before_proposal(tmp_path, market, model, jev, overrides):
    _, report = run(tmp_path, market, settings=config(trials=1, **overrides))
    assert report["stop_reason"] == "model_budget" and not report["trials"]
    assert model.calls == [] and jev[0] == []


def test_failed_jev_request_is_charged_without_fallback(tmp_path, market, model, jev):
    jev[1]["error"] = RuntimeError("Jev connection failed")
    _, report = run(tmp_path, market, settings=config(trials=1))
    assert report["trials"][0]["status"] == "failed"
    assert report["trials"][0]["jev_usage"]["status"] == "attempted"
    assert report["selected"] is None and report["holdout"] is None
    assert len(jev[0]) == 1 and report["_state"]["llm_calls"] == 3


def test_interrupted_jev_request_is_never_replayed(tmp_path, market, model, jev):
    directory = engine.create_run(tmp_path / "runs", config(trials=1), market)
    jev[1]["error"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        agent_research.run_agent_research(directory)
    assert read_json(directory / "report.json")["_state"]["pending"]["phase"] == "jev_gate"
    jev[1]["error"] = None
    report = engine.run_research(directory)
    assert len(jev[0]) == 1 and report["_state"]["llm_calls"] == 3
    assert report["selected"] is None


@pytest.mark.parametrize("change", [{"jev_enabled": False}, {"jev_min_confidence": .8}, {"jev_model": "jev-1.14.0"}])
def test_jev_configuration_frozen_at_creation(tmp_path, market, model, jev, change):
    directory = engine.create_run(tmp_path / "runs", config(trials=1), market)
    with proposals.provider_context({**proposals.llm_settings(), **change}):
        with pytest.raises(ValueError, match="Jev configuration changed"):
            engine.run_research(directory)
    assert model.calls == [] and jev[0] == []


def test_future_prices_cannot_change_jev_evidence_or_decision(tmp_path, market, model, jev):
    _, first = run(tmp_path / "first", market, settings=config(trials=1))
    changed = load_csv(market)
    cut = first["_state"]["development_end"]
    multipliers = np.exp(.003 * np.arange(1, len(changed["close"]) - cut + 1))[:, None]
    for field in ("open", "high", "low", "close"):
        changed[field].iloc[cut:] *= multipliers
    altered_path = tmp_path / "altered.csv"
    save_panel(changed, altered_path)
    _, second = run(tmp_path / "second", altered_path, settings=config(trials=1))
    assert jev[0][0] == jev[0][1]
    assert first["selected"]["jev_gate"] == second["selected"]["jev_gate"]
    assert first["holdout"]["metrics"]["total_return"] != second["holdout"]["metrics"]["total_return"]


def run(tmp_path, market, *, settings=None, **kwargs):
    directory = engine.create_run(tmp_path / "runs", settings or config(), market)
    report = agent_research.run_agent_research(directory, memory_path=tmp_path / "memory.sqlite3", **kwargs)
    return directory, report


def test_three_real_model_families_reviewed_and_fits_frozen_before_each_evaluation(tmp_path, market, model):
    directory, report = run(tmp_path, market, settings=config(trials=3))
    assert report["status"] == "completed", [(t["status"], t.get("reason")) for t in report["trials"]]
    assert [t["status"] for t in report["trials"]] == ["ok"] * 3
    assert report["research"]["model_counts"] == {"rank": 1, "ridge": 1, "hist_gbdt": 1}
    assert [call["role"] for call in model.calls] == ["propose", "review"] * 3
    assert report["_state"]["llm_calls"] == 6
    assert report["_state"]["llm_reserved_tokens"] == 600
    assert report["research"]["policy"] == "model_only"
    for trial in report["trials"]:
        assert trial["causality"]["passed"]
        for audit in trial["training_audit"]["folds"]:
            assert audit["fit_cutoff"] == audit["fold_start"] - 2
            for fit in audit["fit_blocks"]:
                assert fit["label_end_row"] <= fit["fit_row"] < audit["fit_cutoff"]
        if trial["model"] != "rank":
            assert all(audit["fit_count"] > 0 for audit in trial["training_audit"]["folds"])
    final = report["holdout"]["training_audit"]
    assert final["fit_cutoff"] == report["_state"]["development_end"]
    assert all(fit["label_end_row"] < final["fit_cutoff"] for fit in final["fit_blocks"])
    memory = GraphMemory(tmp_path / "memory.sqlite3")
    assert len(memory.list_nodes(report["_state"]["memory_scope"])) == 3
    assert read_json(directory / "research_graph.json") == report["research"]["graph"]
    assert engine.run_research(directory) == report


@pytest.mark.parametrize("overrides", [{"max_llm_calls": 0}, {"max_llm_calls": 1}, {"max_llm_tokens": 199}])
def test_no_two_call_budget_fails_without_any_model_or_fallback(tmp_path, market, model, overrides):
    _, report = run(tmp_path, market, settings=config(**overrides))
    assert report["status"] == "failed" and report["stop_reason"] == "model_budget"
    assert report["trials"] == [] and model.calls == []
    assert report["selected"] is None and report["holdout"] is None
    assert report["_state"]["llm_calls"] == report["_state"]["llm_reserved_tokens"] == 0


def test_provider_failure_is_charged_once_and_never_replaced_by_local_search(tmp_path, market, model):
    model.proposal_error = RuntimeError("transport unavailable")
    _, report = run(tmp_path, market, settings=config(trials=1))
    assert report["status"] == "failed"
    assert [call["role"] for call in model.calls] == ["propose"]
    assert report["trials"][0]["status"] == "failed"
    assert report["selected"] is None
    assert report["_state"]["llm_calls"] == 1 and report["_state"]["llm_reserved_tokens"] == 100


@pytest.mark.parametrize("decision", ["reject", "revise"])
def test_independent_reviewer_veto_cannot_be_selected(tmp_path, market, model, decision):
    model.decisions["rank"] = decision
    _, report = run(tmp_path, market, settings=config(trials=1))
    trial = report["trials"][0]
    assert trial["metrics"] and trial["score"] is not None
    assert trial["status"] == "rejected" and trial["review"]["decision"] == decision
    assert report["status"] == "failed" and report["selected"] is None and report["holdout"] is None
    assert report["_state"]["llm_calls"] == 2


@pytest.mark.parametrize("pause_reason", ["checkpoint", "user_pause"])
def test_pause_resume_preserves_charges_memory_graph_and_dev_feedback(tmp_path, market, model, monkeypatch, pause_reason):
    options = {"stop_after": 1} if pause_reason == "checkpoint" else {"should_pause": lambda: len(model.calls) >= 2}
    directory, paused = run(tmp_path, market, **options)
    assert paused["status"] == "paused" and paused["selected"] is None
    assert paused["stop_reason"] == pause_reason
    assert paused["_state"]["llm_calls"] == 2 and paused["_state"]["llm_reserved_tokens"] == 200
    old_graph = copy.deepcopy(paused["research"]["graph"])
    assert len(old_graph["nodes"]) == 1
    scope = paused["_state"]["memory_scope"]
    memory = GraphMemory(tmp_path / "memory.sqlite3")
    assert len(memory.list_nodes(scope)) == 1
    with proposals.provider_context({"provider": "codex_cli", "codex_model": "changed-model"}):
        with pytest.raises(ValueError, match="provider changed"):
            engine.run_research(directory)
    assert len(model.calls) == 2

    def checked_model(*args, **kwargs):
        persisted = read_json(directory / "report.json")
        assert persisted["status"] == "running" and persisted["stop_reason"] is None
        return model(*args, **kwargs)

    monkeypatch.setattr(research_client, "structured_call", checked_model)
    resumed = engine.run_research(directory)
    assert resumed["status"] == "completed"
    assert resumed["stop_reason"] == "trial_budget"
    assert resumed["_state"]["llm_calls"] == 4 and resumed["_state"]["llm_reserved_tokens"] == 400
    assert resumed["research"]["graph"]["nodes"][0] == old_graph["nodes"][0]
    assert len(memory.list_nodes(scope)) == 2
    context = model.calls[2]["context"]
    assert context["required_model"] == "ridge"
    assert context["development_history"][0]["development_metrics"]["sharpe"] == paused["trials"][0]["metrics"]["sharpe"]
    assert context["parent_trajectories"][0]["review"]["decision"] == "approve"
    assert context["parent_ids"] == [paused["id"] + ":T0001"]
    assert resumed["research"]["graph"]["edges"] == [{"from": paused["id"] + ":T0001", "to": paused["id"] + ":T0002"}]


def test_interrupted_review_keeps_paid_budget_and_is_not_retried_or_selected(tmp_path, market, model):
    directory = engine.create_run(tmp_path / "runs", config(trials=1), market)
    model.interrupt_review = True
    with pytest.raises(KeyboardInterrupt):
        agent_research.run_agent_research(directory, memory_path=tmp_path / "memory.sqlite3")
    interrupted = read_json(directory / "report.json")
    assert interrupted["status"] == "paused"
    assert interrupted["_state"]["pending"]["phase"] == "review"
    assert interrupted["_state"]["llm_calls"] == 2
    resumed = engine.run_research(directory)
    assert resumed["status"] == "failed"
    assert len(model.calls) == 2
    assert resumed["_state"]["llm_reserved_tokens"] == 200
    assert resumed["_state"]["pending"] is None
    assert resumed["trials"][0]["status"] == "failed" and resumed["trials"][0]["score"] is None
    assert resumed["selected"] is None


def test_future_prices_cannot_change_feedback_or_development_winner(tmp_path, market, model):
    _, original = run(tmp_path / "first", market)
    changed = load_csv(market)
    cut = original["_state"]["development_end"]
    # Change the embargo as well as every held-out date, while preserving valid
    # daily OHLC relationships and the entire development history exactly.
    multipliers = np.exp(.003 * np.arange(1, len(changed["close"]) - cut + 1))[:, None]
    for field in ("open", "high", "low", "close"):
        changed[field].iloc[cut:] *= multipliers
    changed_path = tmp_path / "changed.csv"
    save_panel(changed, changed_path)
    first_calls = copy.deepcopy(model.calls)
    model.calls.clear()
    _, altered = run(tmp_path / "second", changed_path)
    keys = ["features", "model", "canonical", "status", "score", "metrics", "folds", "training_audit"]
    assert [{k: t[k] for k in keys} for t in original["trials"]] == [{k: t[k] for k in keys} for t in altered["trials"]]
    assert original["selected"]["canonical"] == altered["selected"]["canonical"]
    assert original["holdout"]["metrics"]["total_return"] != altered["holdout"]["metrics"]["total_return"]
    for before, after in zip(first_calls, model.calls, strict=True):
        # Run-qualified graph identifiers necessarily differ; measured context
        # and the entire reviewer evidence must remain byte-for-byte identical.
        left = json.dumps(before, sort_keys=True).replace(original["id"], "RUN")
        right = json.dumps(after, sort_keys=True).replace(altered["id"], "RUN")
        assert left == right


def test_export_replays_full_multifeature_supervised_strategy_and_costs(tmp_path, market, model):
    model.decisions["rank"] = "reject"
    directory, report = run(tmp_path, market)
    assert report["status"] == "completed" and report["selected"]["model"] == "ridge"
    specification = read_json(directory / "selected_candidate.json")
    assert len(specification["features"]) == 2
    module_spec = importlib.util.spec_from_file_location("exported_agent_strategy", directory / "selected_factor.py")
    exported = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(exported)
    assert exported.STRATEGY == {key: specification[key] for key in ["features", "model", "model_params"]}
    panel = load_csv(directory / "snapshot.csv")
    actual = exported.compute(panel)
    expected, _ = strategy_scores(specification, panel, fit_cutoff=report["_state"]["development_end"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    curve = backtest(actual, panel["close"], BacktestConfig(10, 2, 5), report["_state"]["holdout_start"], len(actual))
    assert summarize(curve) == report["holdout"]["metrics"]
    free_curve = backtest(actual, panel["close"], BacktestConfig(0, 2, 5), report["_state"]["holdout_start"], len(actual))
    assert summarize(curve)["total_return"] < summarize(free_curve)["total_return"]
    assert read_json(directory / "training_audit.json") == report["holdout"]["training_audit"]


def test_context_contains_development_evidence_without_snapshot_or_credentials(tmp_path, market, model):
    directory = engine.create_run(tmp_path / "runs", config(), market, [{"expression": "rank(close)", "api_key": "IMPORT_SECRET_CANARY"}])
    queued = read_json(directory / "report.json")
    queued["source"]["private_metadata"] = {"api_key": "SOURCE_SECRET_CANARY", "holdout": "HELDOUT_CANARY"}
    queued["_state"]["private_credential"] = "STATE_SECRET_CANARY"
    atomic_json(directory / "report.json", queued)
    report = agent_research.run_agent_research(directory, memory_path=tmp_path / "memory.sqlite3")
    assert report["status"] == "completed"
    sent = json.dumps(model.calls, ensure_ascii=False)
    for private in ["PROVIDER_SECRET_CANARY", "IMPORT_SECRET_CANARY", "SOURCE_SECRET_CANARY", "HELDOUT_CANARY", "STATE_SECRET_CANARY", str(tmp_path)]:
        assert private not in sent
    assert model.calls[0]["context"]["imported_features"] == ["rank(close)"]
    assert "development_baselines" in model.calls[0]["context"]
    for call in model.calls:
        assert all(key not in call["context"] for key in ["holdout", "snapshot", "source", "api_key", "_state"])


@pytest.mark.parametrize("reserved", ["id", "parents", "review", "metrics", "score", "status"])
def test_proposer_cannot_overwrite_internal_trial_fields(model, reserved):
    model.extra_proposal = {reserved: {"injected": True}}
    with pytest.raises(ValueError):
        research_client.propose({"required_model": "rank"}, timeout=1)


def test_reported_usage_above_reservation_stops_later_calls(tmp_path, market, model):
    model.reported_tokens = 250
    _, report = run(tmp_path, market, settings=config(trials=3))
    assert report["status"] == "completed" and report["stop_reason"] == "model_budget"
    assert len(report["trials"]) == 1 and len(model.calls) == 2
    assert report["_state"]["llm_reserved_tokens"] == 500
    assert report["research"]["budget"]["llm_reserved_tokens"] == 500


def test_illegal_future_feature_rejected_before_training_or_review(tmp_path, market, model):
    model.extra_proposal = {"features": ["delay(close, -1)"]}
    _, report = run(tmp_path, market, settings=config(trials=1))
    assert report["status"] == "failed" and report["selected"] is None
    assert report["trials"][0]["status"] == "rejected"
    assert "training_audit" not in report["trials"][0]
    assert [call["role"] for call in model.calls] == ["propose"]


def test_cross_run_memory_refreshes_parent_feedback_and_excludes_old_holdout(tmp_path, market, model):
    first_directory, first = run(tmp_path, market)
    assert first["status"] == "completed" and len(first["trials"]) == 2
    assert first["holdout"]["metrics"]
    first["holdout"]["private_note"] = "PRIOR_HOLDOUT_CANARY"
    atomic_json(first_directory / "report.json", first)
    scope = first["_state"]["memory_scope"]
    memory = GraphMemory(tmp_path / "memory.sqlite3")
    original_nodes = {node["id"]: node for node in memory.list_nodes(scope)}
    assert len(original_nodes) == 2
    model.calls.clear()

    _, second = run(tmp_path, market)
    assert second["status"] == "completed"
    assert second["_state"]["memory_scope"] == scope
    assert second["research"]["memory"]["retrieved"] == 2
    assert [trial["status"] for trial in second["trials"]] == ["ok", "ok"]
    contexts = [call["context"] for call in model.calls if call["role"] == "propose"]
    assert len(contexts) == 2
    initial = {node["id"]: node for node in contexts[0]["retrieved_trajectories"]}
    refreshed = {node["id"]: node for node in contexts[1]["retrieved_trajectories"]}
    assert initial == original_nodes
    assert set(refreshed) == set(original_nodes)

    child = second["trials"][0]
    parents = contexts[0]["parent_ids"]
    assert parents and set(parents) <= original_nodes.keys()
    child_reward = (1 + math.tanh(child["score"])) / 2
    for identifier, original in original_nodes.items():
        increment = int(identifier in parents)
        assert refreshed[identifier]["visits"] == original["visits"] + increment
        assert refreshed[identifier]["reward_sum"] == pytest.approx(original["reward_sum"] + increment * child_reward)
    # Persisted planning state is the refreshed snapshot actually used for the
    # second proposal, not the stale counters captured at initial retrieval.
    assert {node["id"]: node for node in second["_state"]["memory_prior"]} == refreshed
    assert len(memory.list_nodes(scope)) == 4

    old_trials = {first["id"] + ":" + trial["id"]: trial for trial in first["trials"]}
    for node in initial.values():
        original = old_trials[node["id"]]
        assert node["review"] == original["review"]
        assert node["development_end"] == first["split"]["development_end"]
        assert node["development_metrics"]["sharpe"] == original["metrics"]["sharpe"]
    assert contexts[0]["parent_trajectories"]
    assert all(node["review"]["decision"] == "approve" for node in contexts[0]["parent_trajectories"])

    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)

    assert not {"holdout", "holdout_metrics", "holdout_curve", "snapshot", "private_note"}.intersection(keys(model.calls))
    assert "PRIOR_HOLDOUT_CANARY" not in json.dumps(model.calls)
