"""Memory persistence must never bypass protocol, chronology or review boundaries."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from alpharesearchos.research_memory import GraphMemory, canonical_strategy, choose_parents


def trial(node_id="run:T1", **changes):
    result = {"id": node_id, "name": "momentum", "hypothesis": "price momentum persists",
              "features": ["rank(ret(close, 20))"], "model": "rank", "model_params": {},
              "development_end": "2024-12-31", "development_metrics": {"sharpe": .7, "avg_turnover": .2},
              "score": .6, "status": "ok", "review": {"decision": "approve", "critique": "stable",
                                                          "risks": ["cost sensitivity"], "suggested_action": "mutate"},
              "parents": [], "action": "explore"}
    result.update(changes)
    return result


def test_canonical_identity_includes_model_features_and_parameters():
    base = trial()
    assert canonical_strategy(base) == canonical_strategy(trial(features=["rank( ret(close,20) )"]))
    assert canonical_strategy(base) == canonical_strategy(trial(score=100, hypothesis="new rationale"))
    assert canonical_strategy(base) != canonical_strategy(trial(model="ridge"))
    assert canonical_strategy(base) != canonical_strategy(trial(model_params={"smoothing": 5}))
    assert canonical_strategy(base) == canonical_strategy(trial(model_params={"alpha": 4.0}))
    assert canonical_strategy(base) != canonical_strategy(trial(features=["rank(ret(close, 40))"]))


def test_persistent_full_trajectory_and_idempotent_feedback(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = GraphMemory(path)
    parent = memory.record_trial("scope", trial())
    child = trial("run:T2", features=["mean(rank(ret(close,20)),5)"], parents=[parent], action="mutate",
                  model="ridge", model_params={"alpha": 1.0, "train_window": 504, "retrain_every": 63,
                                                "horizon": 5, "smoothing": 3})
    memory.record_trial("scope", child)
    memory.record_trial("scope", child)
    nodes = {node["id"]: node for node in GraphMemory(path).list_nodes("scope")}
    assert len(nodes) == 2
    assert nodes[parent]["visits"] == 1
    assert 0 < nodes[parent]["reward_sum"] < 1
    assert nodes["run:T2"]["parents"] == [parent]
    assert nodes["run:T2"]["model_params"]["horizon"] == 5
    assert nodes["run:T2"]["review"]["critique"] == "stable"
    assert nodes["run:T2"]["development_metrics"]["avg_turnover"] == .2
    assert path.stat().st_mode & 0o777 == 0o600


def test_no_holdout_or_secrets_in_database_or_retrieval(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    value = trial()
    sentinel = "SENSITIVE-HOLDOUT-KEY-SENTINEL"
    value.update(holdout={"sharpe": sentinel}, test_metrics={"return": sentinel}, api_key=sentinel,
                 provider={"api_key": sentinel}, canonical=sentinel, raw_response=sentinel)
    value["development_metrics"].update(holdout={"sharpe": sentinel}, test_sharpe=sentinel, api_key=sentinel)
    value["model_params"]["api_key"] = sentinel
    value["review"]["holdout"] = sentinel
    memory.record_trial("scope", value)
    assert sentinel not in json.dumps(memory.retrieve("scope", "momentum"))
    assert sentinel.encode() not in memory.path.read_bytes()
    with sqlite3.connect(memory.path) as db:
        payload = json.loads(db.execute("SELECT payload FROM nodes").fetchone()[0])
    assert set(payload["model_params"]) == set()
    assert "holdout" not in payload["review"]
    assert "test_sharpe" not in payload["development_metrics"]


def test_scope_date_and_feedback_cutoffs(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope-a", trial("parent", development_end="2023-12-31"))
    memory.record_trial("scope-a", trial("future", parents=["parent"], score=100,
                                         features=["rank(volume)"], development_end="2025-01-01"))
    memory.record_trial("scope-b", trial("other-dataset", score=100))
    earlier = memory.retrieve("scope-a", "momentum", asof_date="2024-12-31")
    assert [node["id"] for node in earlier] == ["parent"]
    assert earlier[0]["visits"] == 0 and earlier[0]["reward_sum"] == 0
    assert memory.list_nodes("scope-a")[1]["visits"] == 1
    assert memory.retrieve("unknown-protocol", "momentum") == []
    assert memory.retrieve("scope-a", "momentum", asof_date="2022-12-31") == []


def test_dedup_preserves_trajectories_but_diversifies_context(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope", trial("run1:T1"))
    memory.record_trial("scope", trial("run2:T1", hypothesis="independent replication"))
    memory.record_trial("scope", trial("run2:T2", features=["rank(volume)"], parents=["run2:T1"]))
    assert len(memory.list_nodes("scope")) == 3
    results = memory.retrieve("scope", "", limit=4)
    assert len(results) == 2
    assert len({node["canonical"] for node in results}) == 2


def test_lexical_retrieval_prefers_relevant_development_evidence(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope", trial("quality", features=["rank(volume)"], hypothesis="volume quality", score=10))
    memory.record_trial("scope", trial("momentum", hypothesis="momentum trend", score=-1))
    assert memory.retrieve("scope", "momentum", limit=1)[0]["id"] == "momentum"


def test_failed_candidate_is_retrievable_and_consumes_parent_visit(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope", trial("parent"))
    memory.record_trial("scope", trial("failed", features=["rank("], score=None, status="rejected",
                                         parents=["parent"], review={"decision": "reject", "critique": "invalid syntax"}))
    nodes = {node["id"]: node for node in memory.list_nodes("scope")}
    assert nodes["parent"]["visits"] == 1 and nodes["parent"]["reward_sum"] == 0
    assert "failed" in {node["id"] for node in memory.retrieve("scope", "invalid")}
    assert choose_parents(list(nodes.values())) == ["parent"]


@pytest.mark.parametrize("parent", ["self", "missing", "foreign"])
def test_rejects_self_missing_or_cross_scope_parent(tmp_path, parent):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("other", trial("foreign"))
    with pytest.raises(ValueError):
        memory.record_trial("scope", trial("self", parents=[parent]))
    assert memory.list_nodes("scope") == []


def test_cycles_and_future_data_edges_are_rejected_atomically(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope", trial("a"))
    memory.record_trial("scope", trial("b", parents=["a"]))
    with pytest.raises(ValueError, match="rewritten"):
        memory.record_trial("scope", trial("a", parents=["b"]))
    with pytest.raises(ValueError, match="future"):
        memory.record_trial("scope", trial("c", development_end="2023-12-31", parents=["b"]))
    assert len(memory.list_nodes("scope")) == 2
    with sqlite3.connect(memory.path) as db:
        assert db.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_two_parent_crossover_counts_once_per_parent(tmp_path):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    memory.record_trial("scope", trial("a"))
    memory.record_trial("scope", trial("b", features=["rank(volume)"]))
    memory.record_trial("scope", trial("cross", parents=["a", "b"], action="crossover"))
    nodes = {node["id"]: node for node in memory.list_nodes("scope")}
    assert nodes["a"]["visits"] == nodes["b"]["visits"] == 1


def test_ucb_uses_child_feedback_and_visits():
    a = trial("a", score=.1)
    b = trial("b", score=.1, features=["rank(volume)"])
    assert len(choose_parents([a, b])) == 1
    failures = [trial(f"failed{i}", score=None, status="failed", parents=["b"]) for i in range(10)]
    assert choose_parents([a, b, *failures])[0] == "a"
    success = trial("success", score=10, parents=["b"], review={"decision": "reject"})
    assert choose_parents([a, b, success], exploration=0)[0] == "a"
    # Both have been explored; equal visit counts make feedback decisive.
    a.update(visits=2, reward_sum=0)
    b.update(visits=2, reward_sum=2)
    assert choose_parents([a, b], exploration=0) == ["b", "a"]


def test_ucb_excludes_review_veto_and_empty_search():
    assert choose_parents([]) == []
    assert choose_parents([trial(review={"decision": "reject"})]) == []
    assert choose_parents([trial(review={"veto": True})]) == []
    assert choose_parents([trial(review={"decision": "revise"})]) == []


@pytest.mark.parametrize("change", [
    {"development_end": "2024-02-30"}, {"score": float("nan")}, {"score": float("inf")},
    {"features": []}, {"model": "python"}, {"model_params": {"alpha": -1}},
    {"model_params": {"horizon": 1.5}}, {"parents": ["x", "x"]},
    {"action": "trade"}, {"review": {"decision": "ship"}},
])
def test_invalid_memory_inputs_are_rejected(tmp_path, change):
    memory = GraphMemory(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError):
        memory.record_trial("scope", trial(**change))
    assert memory.list_nodes("scope") == []


def test_refuses_symlinks(tmp_path):
    target = tmp_path / "real.sqlite"
    target.touch()
    path = tmp_path / "link.sqlite"
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        GraphMemory(path)


def test_concurrent_duplicate_recording_is_idempotent(tmp_path):
    path = tmp_path / "memory.sqlite"
    memory = GraphMemory(path)
    memory.record_trial("scope", trial("a"))
    child = trial("b", parents=["a"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: memory.record_trial("scope", child), range(8)))
    assert results == ["b"] * 8
    assert len(memory.list_nodes("scope")) == 2
    assert next(node for node in memory.list_nodes("scope") if node["id"] == "a")["visits"] == 1
