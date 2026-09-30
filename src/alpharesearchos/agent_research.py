"""Model-driven trajectory research, causal prediction and independent review.

Every candidate is proposed by a model. Budget exhaustion terminates research;
it never falls back to heuristic expression mutation. Selection sees only
development folds. Review is a separate model request, not the proposer's text.
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import research_client
from .backtest import BacktestConfig, backtest, bootstrap_sharpe_interval, evaluate_development, summarize
from .config import ResearchConfig
from .data import file_hash, load_csv
from .factorminer_adapter import differential_check
from .factors import causality_check, evaluate_expression
from .model_settings import validate_jev_settings
from .predictive import strategy_scores, validate_strategy
from .proposals import llm_settings, provider_context
from .research_memory import GraphMemory, choose_parents
from .store import read_json, run_lock

# Conservative admission reservation for the client's bounded 24 KB payload.
# Reported usage above this reservation is still charged; it is not a price.
JEV_TOKEN_RESERVATION = 32768


def jev_identity(settings):
    config = validate_jev_settings(settings, require_key=settings.get("jev_enabled", False))
    if not config["jev_enabled"]:
        return {"enabled": False}
    return {"enabled": True, "base_url": config["jev_base_url"], "model": config["jev_model"],
            "min_confidence": config["jev_min_confidence"]}


def enforce_constraints(strategy, constraints, metrics=None):
    for expression in strategy["features"]:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in constraints.get("forbidden_fields", []):
                raise ValueError(f"Forbidden field: {node.id}")
            required = constraints.get("required_window")
            if required is not None and isinstance(node, ast.Call) and len(node.args) == 2:
                if not isinstance(node.args[1], ast.Constant) or node.args[1].value != required:
                    raise ValueError(f"Every factor window must equal {required}")
    ceiling = constraints.get("max_turnover")
    if metrics is not None and ceiling is not None and metrics["avg_turnover"] > ceiling:
        raise ValueError(f"Development turnover exceeds the {ceiling:g} constraint")


def development_scores(strategy, dev, folds, signal_lag=2):
    """Freeze each fold's fitted estimator before the first tradable signal."""
    scores = pd.DataFrame(np.nan, index=dev["close"].index, columns=dev["close"].columns)
    audits = []
    for start, end in folds:
        # No validation label may train its own fold's estimator.
        cutoff = start - signal_lag
        predicted, audit = strategy_scores(strategy, dev, fit_cutoff=cutoff)
        scores.iloc[start - signal_lag:end - signal_lag] = predicted.iloc[start - signal_lag:end - signal_lag]
        audits.append({"fold_start": start, "fold_end": end, "fit_cutoff": cutoff, **audit})
    return scores, {"policy": "fit_frozen_before_each_development_fold", "folds": audits,
                    "model": strategy["model"], "model_params": strategy["model_params"]}


def _memory_record(report, trial):
    value = dict(trial)
    value["id"] = report["id"] + ":" + trial["id"]
    value["development_end"] = report["split"]["development_end"]
    value["development_metrics"] = trial.get("metrics", {})
    return value


def _graph(report):
    nodes = [{key: trial.get(key) for key in ["id", "parents", "action", "model", "status", "score"]} for trial in report["trials"]]
    for node in nodes:
        node["trial_id"], node["id"] = node["id"], report["id"] + ":" + node["id"]
    return {"nodes": nodes, "edges": [{"from": parent, "to": node["id"]} for node in nodes for parent in node.get("parents") or []]}


def _planner(report, memory):
    trials = report["trials"]
    index = len(trials)
    # Explicit model-family coverage is a research action, not a factor template.
    required_model = ["rank", "ridge", "hist_gbdt"][index] if index < 3 else None
    local = [_memory_record(report, trial) for trial in trials if trial.get("canonical") and not trial.get("memory_warning")]
    pool = [*memory, *local]
    parents = choose_parents(pool)
    if index == 0:
        action = "explore"
    elif required_model is not None:
        action = "model_switch"
    else:
        action = "crossover" if len(parents) > 1 else "mutate" if parents else "explore"
    return action, parents, required_model


def run_agent_research(directory: Path, *, stop_after=None, should_pause=None, memory_path=None):
    from .engine import _checkpoint, _export, code_fingerprint, event, now

    with run_lock(directory):
        report = read_json(directory / "report.json")
        if file_hash(directory / "snapshot.csv") != report["source"]["sha256"]:
            raise ValueError("Snapshot checksum changed")
        if report["status"] == "completed":
            if not (directory / "manifest.json").exists():
                _export(directory, report)
            return report
        if code_fingerprint() != report["provenance"]["source_code_sha256"]:
            raise ValueError("Research code changed since creation; create a new experiment")
        if report["provenance"]["numpy"] != np.__version__ or report["provenance"]["pandas"] != pd.__version__:
            raise ValueError("Numerical dependencies changed")
        import sklearn
        if report["provenance"].get("sklearn", sklearn.__version__) != sklearn.__version__:
            raise ValueError("Predictive model dependency changed")
        report["provenance"]["sklearn"] = sklearn.__version__
        config = ResearchConfig(**report["config"])
        state = report["_state"]
        provider = llm_settings()
        provider.update(validate_jev_settings(provider, require_key=provider.get("jev_enabled", False)))
        if provider.get("provider") == "codex_cli":
            from .codex_provider import codex_status
            detected = codex_status()
            provider["codex_model"] = provider.get("codex_model") or detected["default_model"] or ""
            identity = {"provider": "codex_cli", "model": provider["codex_model"], "cli_version": detected["version"]}
        else:
            identity = {"provider": "openai_compatible", "model": provider.get("model", ""), "base_url": provider.get("base_url", "")}
        if state.get("agent_provider", identity) != identity:
            raise ValueError("Model provider changed since experiment creation; restore it or create a new experiment")
        state["agent_provider"] = identity
        gate_identity = jev_identity(provider)
        if state.get("jev_gate_config", gate_identity) != gate_identity:
            raise ValueError("Jev configuration changed since experiment creation; restore it or create a new experiment")
        state["jev_gate_config"] = gate_identity
        state.setdefault("jev_calls", 0)
        panel = load_csv(directory / "snapshot.csv")
        dev = {field: values.iloc[:state["development_end"]].copy() for field, values in panel.items()}
        bt = BacktestConfig(config.cost_bps, config.top_k, config.rebalance_every)
        clock_start, prior = time.monotonic(), report["progress"]["elapsed_seconds"]
        memory_path = memory_path or state.get("memory_path") or directory.parent.parent / ".alphaos" / "research-memory.sqlite3"
        state["memory_path"] = str(Path(memory_path).resolve())
        memory = GraphMemory(memory_path)

        def checkpoint():
            report["research"]["graph"] = _graph(report)
            report["research"]["model_counts"] = {model: sum(trial.get("model") == model for trial in report["trials"]) for model in ["rank", "ridge", "hist_gbdt"]}
            report["research"]["budget"] = {key: state[key] for key in ["llm_calls", "llm_reserved_tokens", "jev_calls"]}
            _checkpoint(directory, report, clock_start, prior)

        def elapsed():
            return prior + time.monotonic() - clock_start

        def paid_call(role, context):
            if elapsed() >= config.max_seconds:
                raise ValueError("Time budget exhausted before model request")
            reserve = JEV_TOKEN_RESERVATION if role == "jev_gate" else research_client.reservation()
            if state["llm_calls"] >= config.max_llm_calls or state["llm_reserved_tokens"] + reserve > config.max_llm_tokens:
                raise ValueError("Model budget exhausted before review")
            state["llm_calls"] += 1
            state["llm_reserved_tokens"] += reserve
            if role == "jev_gate":
                state["jev_calls"] += 1
                state["pending"]["trial"]["jev_usage"] = {
                    "provider": "typesafe_jev", "model": gate_identity["model"], "status": "attempted"}
            state["pending"]["phase"] = role
            report["research"]["stage"] = role
            checkpoint()
            if role == "jev_gate":
                from .jev_client import review_gate
                value, usage = review_gate(context, settings=provider, timeout=config.max_seconds - elapsed())
            else:
                function = research_client.propose if role == "propose" else research_client.review
                with provider_context(provider):
                    value, usage = function(context, timeout=config.max_seconds - elapsed())
            used = usage.get("reported_total_tokens")
            if type(used) is int and used > reserve:
                state["llm_reserved_tokens"] += used - reserve
            if role == "jev_gate":
                state["pending"]["trial"]["jev_usage"] = usage
                # Aliases may change remotely between calls or across a resume.
                # Do not combine decisions from different model versions.
                actual_model = usage["model"]
                if state.get("jev_actual_model", actual_model) != actual_model:
                    raise ValueError("Jev response model changed during the experiment; create a new experiment")
                state["jev_actual_model"] = actual_model
                report["research"]["jev"]["actual_model"] = actual_model
            return value, usage

        if "research" not in report:
            scope_config = {key: report["config"][key] for key in ["cost_bps", "top_k", "rebalance_every", "folds", "warmup", "holdout_fraction", "constraints"]}
            scope = hashlib.sha256(json.dumps({"data": report["source"]["sha256"], "protocol": scope_config, "jev": gate_identity,
                                              "implementation": report["provenance"]["source_code_sha256"]}, sort_keys=True).encode()).hexdigest()
            state["memory_scope"] = scope
            state["memory_prior"] = memory.retrieve(scope, config.direction, limit=4, asof_date=report["split"]["development_end"])
            report["research"] = {"policy": "model_only", "stage": "propose", "graph": {"nodes": [], "edges": []},
                                   "memory": {"scope": scope, "retrieved": len(state["memory_prior"]), "stored": 0},
                                   "constraints": config.constraints, "model_counts": {}, "provider": identity,
                                   "jev": dict(gate_identity)}
            baseline = evaluate_expression("rank(ret(close, 20))", dev)
            report["research"]["development_baselines"] = {
                "momentum20": evaluate_development(baseline, dev["close"], bt, state["folds"]),
                "momentum60": evaluate_development(evaluate_expression("rank(ret(close, 60))", dev), dev["close"], bt, state["folds"]),
                "equal_weight": evaluate_development(baseline, dev["close"], replace(bt, top_k=dev["close"].shape[1]), state["folds"]),
            }
            event(report, "自主研究：轨迹检索、因子与模型联合提案、独立复核；预算耗尽即停止")
        report["status"] = "running"
        if report.get("stop_reason") in {"user_pause", "checkpoint"}:
            report["stop_reason"] = None
        if state.get("pending"):
            pending = state["pending"]
            interrupted = pending.get("trial") or {"id": pending["id"], "name": "中断候选"}
            interrupted.update(status="failed", score=None, reason="Interrupted model attempt; charged calls preserved, no retry")
            report["trials"].append(interrupted)
            state["pending"] = None
        checkpoint()
        seen = {trial["canonical"] for trial in report["trials"] if trial.get("canonical")}
        signal_cache = {}
        for trial in report["trials"]:
            if trial.get("metrics") and trial.get("features"):
                signal_cache[trial["id"]] = development_scores(trial, dev, state["folds"])[0]
        try:
            while len(report["trials"]) < config.trials and not report.get("selected"):
                if (should_pause and should_pause()) or (stop_after is not None and len(report["trials"]) >= stop_after):
                    report.update(status="paused", stop_reason="user_pause" if should_pause and should_pause() else "checkpoint")
                    checkpoint()
                    return report
                if elapsed() >= config.max_seconds:
                    report["stop_reason"] = "time_budget"
                    break
                reserve = research_client.reservation()
                calls_needed = 3 if gate_identity["enabled"] else 2
                tokens_needed = 2 * reserve + (JEV_TOKEN_RESERVATION if gate_identity["enabled"] else 0)
                if state["llm_calls"] + calls_needed > config.max_llm_calls or state["llm_reserved_tokens"] + tokens_needed > config.max_llm_tokens:
                    report["stop_reason"] = "model_budget"
                    break
                if state["memory_prior"]:
                    # Refresh feedback for the originally retrieved trajectories.
                    # Database counters already include this run's stored children.
                    updated = {node["id"]: node for node in memory.list_nodes(
                        state["memory_scope"], asof_date=report["split"]["development_end"], limit=10000)}
                    if any(node["id"] not in updated for node in state["memory_prior"]):
                        raise ValueError("Retrieved trajectory missing from the 10000-node memory window; create a new experiment")
                    state["memory_prior"] = [updated[node["id"]] for node in state["memory_prior"]]
                action, parents, required_model = _planner(report, state["memory_prior"])
                trial = {"id": f"T{len(report['trials']) + 1:04d}", "status": "failed", "score": None,
                         "action": action, "parents": parents, "origin": "agent"}
                state["pending"] = {"id": trial["id"], "phase": "propose", "trial": trial}
                started = time.monotonic()
                try:
                    history = [research_client.compact_trial(item) for item in report["trials"][-5:]]
                    parent_pool = {item["id"]: item for item in state["memory_prior"]}
                    parent_pool.update({report["id"] + ":" + item["id"]: research_client.compact_trial(item) for item in report["trials"]})
                    context = {"direction": config.direction, "action": action, "parent_ids": parents, "required_model": required_model,
                               "parent_trajectories": [parent_pool[identifier] for identifier in parents if identifier in parent_pool],
                               "constraints": config.constraints, "development_history": history,
                               "retrieved_trajectories": state["memory_prior"],
                               "imported_features": [item["expression"] for item in state["imported_candidates"]],
                               "development_baselines": {name: {"score": value["score"], "metrics": value["metrics"]} for name, value in report["research"]["development_baselines"].items()},
                               "asset_count": len(dev["close"].columns), "complexity_limit": config.max_complexity,
                               "warmup_limit": config.warmup, "execution": {"cost_bps": config.cost_bps, "top_k": config.top_k, "rebalance_every": config.rebalance_every, "signal_lag": 2}}
                    proposed, usage = paid_call("propose", context)
                    trial.update(proposed, llm_usage=usage, proposer_usage=usage)
                    normalized = validate_strategy(proposed)
                    trial.update(normalized)
                    trial["expression"] = trial["features"][0]  # legacy display only; exported strategy contains all features/model.
                    if required_model and trial["model"] != required_model:
                        raise ValueError(f"Research action requires model={required_model}")
                    if trial["complexity"] > config.max_complexity or trial["lookback"] > config.warmup:
                        raise ValueError("Strategy exceeds complexity or warmup budget")
                    enforce_constraints(trial, config.constraints)
                    if trial["canonical"] in seen:
                        raise ValueError("Duplicate strategy (features, model and parameters)")
                    seen.add(trial["canonical"])
                    report["research"]["stage"] = "evaluate"
                    checkpoint()
                    checks = []
                    for expression in trial["features"]:
                        causal = causality_check(expression, dev)
                        values = evaluate_expression(expression, dev)
                        parity = differential_check(expression, dev, values)
                        if not causal["passed"] or parity.get("status") == "failed":
                            raise ValueError("Feature causality or differential check failed")
                        checks.append({"expression": expression, "causality": causal, "engine_parity": parity})
                    trial["feature_audits"] = checks
                    trial["causality"] = {"passed": True, "detail": "Causal features; matured training labels; fit frozen before every evaluation fold"}
                    scores, audit = development_scores(trial, dev, state["folds"])
                    signal_slice = scores.iloc[config.warmup - 2:state["development_end"] - 2].to_numpy()
                    if float(np.isfinite(signal_slice).mean()) < .95:
                        raise ValueError("Less than 95% causal development prediction coverage")
                    if float(scores.iloc[config.warmup - 2:state["development_end"] - 2].std(axis=1).fillna(0).max()) < 1e-12:
                        raise ValueError("No cross-sectional prediction variation")
                    trial["training_audit"] = audit
                    from .engine import _correlation
                    for other_id, other in signal_cache.items():
                        if _correlation(scores, other, config.warmup) > .998:
                            raise ValueError(f"Prediction redundant with {other_id}")
                    result = evaluate_development(scores, dev["close"], bt, state["folds"])
                    trial.update(result)
                    trial["score"] = float(result["score"]) - .002 * trial["complexity"]
                    enforce_constraints(trial, config.constraints, trial["metrics"])
                    signal_cache[trial["id"]] = scores
                    trial["status"] = "evaluated"
                    review_context = {"direction": config.direction, "constraints": config.constraints,
                                                                "candidate": research_client.compact_trial(trial),
                                                                "technical_audit": {"causal_features": True, "training": "Frozen before each development fold; only matured labels", "coverage": float(np.isfinite(signal_slice).mean())},
                                                                "execution": context["execution"],
                                                                "development_baselines": context["development_baselines"]}
                    verdict, review_usage = paid_call("review", review_context)
                    trial.update(review=verdict, reviewer_usage=review_usage)
                    trial["status"] = "ok" if verdict["decision"] == "approve" else "rejected"
                    if trial["status"] != "ok":
                        trial["reason"] = "Reviewer " + verdict["decision"] + ": " + verdict["critique"]
                    elif gate_identity["enabled"]:
                        gate, gate_usage = paid_call("jev_gate", {**review_context, "review": verdict})
                        trial.update(jev_gate=gate, jev_usage=gate_usage)
                        if gate["decision"] != "approve":
                            trial.update(status="rejected", reason="Jev " + gate["decision"] + ": " + gate["summary"])
                except (ValueError, SyntaxError) as exc:
                    trial.update(status="rejected", reason=str(exc)[:1500])
                except Exception as exc:
                    trial.update(status="failed", reason=f"{type(exc).__name__}: {str(exc)[:700]}")
                trial["seconds"] = round(time.monotonic() - started, 3)
                if trial.get("features"):
                    try:
                        memory.record_trial(state["memory_scope"], _memory_record(report, trial))
                        report["research"]["memory"]["stored"] += 1
                    except (ValueError, OSError) as exc:
                        trial["memory_warning"] = str(exc)[:200]
                report["trials"].append(trial)
                state["pending"] = None
                event(report, f"{trial['id']} {trial.get('model', '')} {trial.get('name', 'proposal')}: {trial['status']}")
                checkpoint()
            if should_pause and should_pause() and not report.get("selected"):
                report.update(status="paused", stop_reason="user_pause")
                checkpoint()
                return report
            accepted = [trial for trial in report["trials"] if trial["status"] == "ok"
                        and trial.get("review", {}).get("decision") == "approve"
                        and (not gate_identity["enabled"] or trial.get("jev_gate", {}).get("decision") == "approve")]
            if not accepted:
                report.update(status="failed", stop_reason=report.get("stop_reason") or "no_reviewed_candidates")
                event(report, "没有通过独立复核的有效候选；没有调用本地搜索兜底", "warning")
            else:
                if not report.get("selected"):
                    report["selected"] = max(accepted, key=lambda trial: trial["score"]).copy()
                    report["selection_frozen_at"] = now()
                    report["research"]["stage"] = "select"
                    event(report, f"冻结候选 {report['selected']['id']} 及模型参数，开始最终评价")
                    checkpoint()
                scores, audit = strategy_scores(report["selected"], panel, fit_cutoff=state["development_end"])
                start, end = state["holdout_start"], len(panel["close"])
                curve = backtest(scores, panel["close"], bt, start, end)
                baseline = curve.copy()
                baseline["return"], baseline["turnover"] = curve["benchmark_return"], curve["benchmark_turnover"]
                metrics = summarize(curve)
                fixed = {str(window): summarize(backtest(evaluate_expression(f"rank(ret(close, {window}))", panel), panel["close"], bt, start, end)) for window in [20, 60]}
                display = curve.reset_index(names="date")
                display["date"] = display["date"].dt.strftime("%Y-%m-%d")
                report["holdout"] = {"metrics": metrics, "baseline_metrics": summarize(baseline), "momentum_metrics": fixed["60"],
                                     "fixed_baselines": fixed, "training_audit": audit, "curve": display.to_dict("records"),
                                     "sharpe_interval": bootstrap_sharpe_interval(curve["return"].to_numpy(), config.seed),
                                     "stress": [{"cost_bps": cost, "metrics": summarize(backtest(scores, panel["close"], replace(bt, cost_bps=cost), start, end))} for cost in sorted({0.0, config.cost_bps, config.cost_bps * 2, config.cost_bps * 4})]}
                curve.to_csv(directory / "holdout_curve.csv", index_label="date")
                report.update(status="completed", stop_reason=report.get("stop_reason") or "trial_budget")
                report["research"]["stage"] = "complete"
                report["conclusion"] = "最终评价超过等权基线" if metrics["excess_return"] > 0 else "最终评价未超过等权基线"
                event(report, report["conclusion"])
            checkpoint()
            _export(directory, report)
            return report
        except BaseException as exc:
            report["status"] = "paused" if isinstance(exc, KeyboardInterrupt) else "failed"
            event(report, f"自主研究中断：{type(exc).__name__}", "error")
            checkpoint()
            raise
