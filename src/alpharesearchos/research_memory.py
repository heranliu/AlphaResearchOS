"""Development-only trajectory DAG and deterministic, bounded research retrieval.

Scopes identify a dataset *and* an evaluation protocol. Nodes are immutable;
insertion can only reference existing parents. No report or provider dictionary
is serialized wholesale. This is an independently authored implementation.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

_PARAMS = {"alpha", "train_window", "retrain_every", "horizon", "smoothing"}
_METRICS = {"sharpe", "annual_return", "annualized_return", "total_return", "net_return",
            "excess_return", "max_drawdown", "volatility", "annual_volatility", "turnover",
            "average_turnover", "rank_ic", "ic", "coverage", "finite_fraction", "complexity",
            "avg_turnover", "mean_sharpe", "std_sharpe", "mean_excess_return", "score", "observations",
            "cagr", "annual_vol", "days", "benchmark_total_return"}
_ACTIONS = {"explore", "mutate", "crossover", "model_switch"}


def _text(value, name, maximum=1000, *, empty=True):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise ValueError(f"Invalid {name}")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError(f"Invalid {name}")
    return value.strip()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid finite numeric {name}")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"Invalid finite numeric {name}") from None
    if not math.isfinite(result):
        raise ValueError(f"Invalid finite numeric {name}")
    return result


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("development_end/asof_date must be YYYY-MM-DD")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError("Invalid development_end/asof_date") from None


def _strategy(trial):
    if not isinstance(trial, dict):
        raise ValueError("strategy must be an object")
    features = trial.get("features")
    if features is None:
        features = [trial.get("expression")]
    if not isinstance(features, list) or not 1 <= len(features) <= 6:
        raise ValueError("Memory strategy requires one to six features")
    features = [_text(value, "feature", empty=False) for value in features]
    model = trial.get("model", "rank")
    if not isinstance(model, str) or model not in {"rank", "ridge", "hist_gbdt"}:
        raise ValueError("Unknown memory model")
    source = trial.get("model_params", {})
    if not isinstance(source, dict):
        raise ValueError("model_params must be an object")
    params = {}
    for key in sorted(_PARAMS & source.keys()):
        value = _number(source[key], key)
        low, high = {"alpha": (1e-6, 1e4), "train_window": (63, 1008), "retrain_every": (5, 252),
                     "horizon": (1, 20), "smoothing": (1, 20)}[key]
        if not low <= value <= high or (key != "alpha" and type(source[key]) is not int):
            raise ValueError("Invalid model parameter range/type")
        params[key] = value if key == "alpha" else int(value)
    return {"features": features, "model": model, "model_params": params}


def canonical_strategy(trial):
    """Canonical feature syntax + model configuration; excludes measured results.

    Invalid candidate syntax still has an identity so failures can be remembered.
    This function parses expressions but never executes or certifies them.
    """
    strategy = _strategy(trial)
    defaults = {"alpha": 1.0, "train_window": 504, "retrain_every": 63, "horizon": 5, "smoothing": 3}
    params = {**defaults, **strategy["model_params"]}
    strategy["model_params"] = {"smoothing": params["smoothing"]} if strategy["model"] == "rank" else params
    normalized = []
    for feature in strategy["features"]:
        try:
            normalized.append(ast.dump(ast.parse(feature, mode="eval"), include_attributes=False))
        except (SyntaxError, ValueError, RecursionError):
            normalized.append("invalid:" + feature)
    strategy["features"] = normalized
    encoded = json.dumps(strategy, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _review(source):
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError("review must be an object")
    result = {}
    if "decision" in source:
        if not isinstance(source["decision"], str) or source["decision"] not in {"approve", "revise", "reject"}:
            raise ValueError("Unknown review decision")
        result["decision"] = source["decision"]
    for key in ("passed", "veto"):
        if key in source:
            if type(source[key]) is not bool:
                raise ValueError("Review flags must be boolean")
            result[key] = source[key]
    for key in ("critique", "suggested_action"):
        if key in source:
            result[key] = _text(source[key], "review text", 1200)
    if "risks" in source:
        if not isinstance(source["risks"], list) or len(source["risks"]) > 10:
            raise ValueError("Review risks must be a short list")
        result["risks"] = [_text(value, "review risk", 300) for value in source["risks"]]
    return result


def _trial(source):
    if not isinstance(source, dict):
        raise ValueError("trial must be an object")
    strategy = _strategy(source)
    node = {"id": _text(source.get("id"), "node id", 160, empty=False), **strategy,
            "development_end": _day(source.get("development_end")),
            "hypothesis": _text(source.get("hypothesis", ""), "hypothesis", 1200),
            "name": _text(source.get("name", ""), "name", 150),
            "action": source.get("action", "explore"), "status": source.get("status", "ok"),
            "review": _review(source.get("review")), "score": None}
    if (not isinstance(node["action"], str) or node["action"] not in _ACTIONS
            or not isinstance(node["status"], str) or node["status"] not in {"ok", "rejected", "failed"}):
        raise ValueError("Invalid research action/status")
    if source.get("score") is not None:
        node["score"] = _number(source["score"], "score")
    if node["status"] == "ok" and node["score"] is None:
        raise ValueError("An evaluated trial requires a development score")
    parents = source.get("parents", [])
    if not isinstance(parents, list) or len(parents) > 2:
        raise ValueError("A trajectory accepts at most two parents")
    node["parents"] = [_text(value, "parent id", 160, empty=False) for value in parents]
    if len(node["parents"]) != len(set(node["parents"])) or node["id"] in node["parents"]:
        raise ValueError("Duplicate or self parent")
    metrics = source.get("development_metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("development_metrics must be an object")
    node["development_metrics"] = {key: _number(metrics[key], "development metric")
                                   for key in sorted(_METRICS & metrics.keys()) if metrics[key] is not None}
    node["canonical"] = canonical_strategy(strategy)
    return node


def _approved(trial):
    if not isinstance(trial, dict):
        return False
    review = trial.get("review") or {}
    if not isinstance(review, dict):
        return False
    try:
        _number(trial.get("score"), "score")
    except ValueError:
        return False
    return (trial.get("status") == "ok" and review.get("decision", "approve") == "approve"
            and review.get("passed", True) is not False and not review.get("veto", False))


def _reward(trial):
    return (1 + math.tanh(trial["score"])) / 2 if _approved(trial) else 0.0


def _tokens(text):
    # Chinese bigrams keep relevance useful without adding a tokenizer dependency.
    tokens = set(re.findall(r"[a-z0-9_]+", text.lower()))
    for phrase in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.update(phrase[index:index + 2] for index in range(max(1, len(phrase) - 1)))
    return tokens


class GraphMemory:
    """SQLite trajectory history; each method uses its own short transaction."""

    def __init__(self, path):
        self.path = Path(os.path.abspath(os.fspath(path)))
        for item in (self.path, *self.path.parents):
            if item.is_symlink():
                raise ValueError("Memory path must not contain symbolic links")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Memory must be a regular SQLite file")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    scope TEXT NOT NULL, node_id TEXT NOT NULL, canonical TEXT NOT NULL,
                    payload TEXT NOT NULL, development_end TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    visits INTEGER NOT NULL DEFAULT 0, reward_sum REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (scope, node_id));
                CREATE INDEX IF NOT EXISTS nodes_scope_date ON nodes(scope, development_end);
                CREATE TABLE IF NOT EXISTS edges (
                    scope TEXT NOT NULL, parent_id TEXT NOT NULL, child_id TEXT NOT NULL,
                    PRIMARY KEY(scope, parent_id, child_id),
                    FOREIGN KEY(scope, parent_id) REFERENCES nodes(scope, node_id),
                    FOREIGN KEY(scope, child_id) REFERENCES nodes(scope, node_id));
            """)

    @contextmanager
    def _connect(self):
        if self.path.is_symlink():
            raise ValueError("Memory path must not be a symbolic link")
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def record_trial(self, scope, trial):
        """Insert one immutable trajectory; exact replay does not consume visits."""
        scope = _text(scope, "scope", 128, empty=False)
        node = _trial(trial)
        payload = json.dumps(node, sort_keys=True, ensure_ascii=False, allow_nan=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT payload FROM nodes WHERE scope=? AND node_id=?", (scope, node["id"])).fetchone()
            if old:
                if old["payload"] != payload:
                    raise ValueError("A trajectory id cannot be rewritten")
                return node["id"]
            for parent in node["parents"]:
                row = db.execute("SELECT development_end FROM nodes WHERE scope=? AND node_id=?", (scope, parent)).fetchone()
                if row is None:
                    raise ValueError("Parents must already exist in the same research scope")
                if row["development_end"] > node["development_end"]:
                    raise ValueError("A trajectory cannot depend on future development data")
            db.execute("INSERT INTO nodes(scope,node_id,canonical,payload,development_end,recorded_at) VALUES(?,?,?,?,?,?)",
                       (scope, node["id"], node["canonical"], payload, node["development_end"], datetime.now(timezone.utc).isoformat()))
            for parent in node["parents"]:
                db.execute("INSERT INTO edges VALUES(?,?,?)", (scope, parent, node["id"]))
                db.execute("UPDATE nodes SET visits=visits+1,reward_sum=reward_sum+? WHERE scope=? AND node_id=?",
                           (_reward(node), scope, parent))
        return node["id"]

    def list_nodes(self, scope, asof_date=None, limit=200):
        """Project safe records only; cutoff refers to data used by the trajectory."""
        scope = _text(scope, "scope", 128, empty=False)
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise ValueError("Memory limit must be between 1 and 10000")
        cutoff = _day(asof_date) if asof_date is not None else "9999-12-31"
        with self._connect() as db:
            rows = db.execute("SELECT * FROM nodes WHERE scope=? AND development_end<=? ORDER BY rowid DESC LIMIT ?",
                              (scope, cutoff, limit)).fetchall()
            feedback = db.execute("""SELECT e.parent_id, child.payload FROM edges e
                JOIN nodes child ON child.scope=e.scope AND child.node_id=e.child_id
                WHERE e.scope=? AND child.development_end<=?""", (scope, cutoff)).fetchall()
        counts, rewards = {}, {}
        for child in feedback:
            parent = child["parent_id"]
            counts[parent] = counts.get(parent, 0) + 1
            rewards[parent] = rewards.get(parent, 0.0) + _reward(_trial(json.loads(child["payload"])))
        records = []
        for row in rows:
            node = _trial(json.loads(row["payload"]))
            # Post-cutoff children must not leak through visits or UCB rewards.
            node.update(visits=counts.get(node["id"], 0), reward_sum=rewards.get(node["id"], 0.0))
            records.append(node)
        return records

    def retrieve(self, scope, query, limit=4, asof_date=None):
        """Lexical relevance + development score + greedy feature diversity.

        Exact strategies appear once even if separate runs produced multiple
        hypotheses. Rejected trajectories remain useful failure memory.
        """
        query = _text(query, "memory query", 4000)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("Retrieval limit must be between 1 and 20")
        candidates = self.list_nodes(scope, asof_date, limit=10000)
        query_tokens = _tokens(query)
        chosen, seen = [], set()
        while candidates and len(chosen) < limit:
            def priority(node):
                words = _tokens(node["hypothesis"] + " " + " ".join(node["features"]))
                relevance = len(query_tokens & words) / max(1, len(query_tokens))
                feature_words = _tokens(" ".join(node["features"]))
                prior_words = [_tokens(" ".join(value["features"])) for value in chosen]
                novelty = 1 - max((len(feature_words & words) / max(1, len(feature_words | words))
                                   for words in prior_words), default=0)
                return (2 * relevance + .35 * novelty + .2 * _reward(node), node["id"])
            node = max(candidates, key=priority)
            candidates.remove(node)
            if node["canonical"] not in seen:
                chosen.append(node)
                seen.add(node["canonical"])
        return chosen


def choose_parents(trials, exploration=1.4):
    """Choose one unexplored branch, or two explored distinct strategies.

    UCB rewards use bounded development scores and child review outcomes. Failed
    children consume visits with zero reward. This is a one-step DAG scheduler,
    not a reproduction of full MCTS or Bayesian graph retrieval.
    """
    exploration = _number(exploration, "exploration")
    if exploration < 0:
        raise ValueError("exploration must be nonnegative")
    if not isinstance(trials, list):
        raise ValueError("trials must be a list")
    if any(not isinstance(trial, dict) for trial in trials):
        raise ValueError("Each trial must be an object")
    ids = [_text(trial.get("id"), "node id", 160, empty=False) for trial in trials]
    if len(ids) != len(set(ids)):
        raise ValueError("Parent scheduling requires unique node ids")
    eligible = [trial for trial in trials if _approved(trial)]
    if not eligible:
        return []
    visits, rewards = {}, {}
    for trial in eligible:
        node_id = trial["id"]
        children = [child for child in trials if node_id in child.get("parents", [])]
        count = trial.get("visits", len(children))
        if type(count) is not int or count < 0:
            raise ValueError("visits must be a nonnegative integer")
        visits[node_id] = max(count, len(children))
        rewards[node_id] = (_number(trial["reward_sum"], "reward_sum") if "reward_sum" in trial
                            else sum(_reward(child) for child in children))
        if not 0 <= rewards[node_id] <= visits[node_id] + 1e-10:
            raise ValueError("reward_sum must fit the visit count")
    total = sum(visits.values()) + 2
    def priority(trial):
        n = visits[trial["id"]]
        mean = (_reward(trial) + rewards[trial["id"]]) / (n + 1)
        return (mean + exploration * math.sqrt(math.log(total) / (n + 1)), trial["id"])
    ordered = sorted(eligible, key=priority, reverse=True)
    first = ordered[0]
    result = [first["id"]]
    if visits[first["id"]] > 0:
        for other in ordered[1:]:
            if visits[other["id"]] > 0 and canonical_strategy(other) != canonical_strategy(first):
                result.append(other["id"])
                break
    return result
