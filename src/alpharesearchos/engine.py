"""Bounded research loop with frozen data and one terminal holdout evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import __version__
from .backtest import BacktestConfig, backtest, evaluate_development, summarize
from .config import ResearchConfig
from .data import describe, file_hash, load_csv, make_demo, save_panel
from .factorminer_adapter import differential_check, provenance
from .factors import causality_check, evaluate_expression, validate_expression
from .proposals import llm_configured, llm_proposal, llm_token_reservation, local_proposal
from .store import atomic_json, read_json, run_lock


def now():
    return datetime.now(timezone.utc).isoformat()


def event(report, message, level="info"):
    report["events"].append({"time": now(), "level": level, "message": message})


def code_fingerprint():
    h = hashlib.sha256()
    for file in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(str(file.relative_to(Path(__file__).parent)).encode())
        h.update(file.read_bytes())
    return h.hexdigest()


def create_run(root: Path, config: ResearchConfig, csv_path: Path | None = None, candidates=None):
    if config.mode in {"llm", "agent"} and not llm_configured():
        raise ValueError("LLM mode requires a configured API provider or an authenticated local Codex CLI")
    if config.mode == "agent":
        from .agent_research import jev_identity
        from .proposals import llm_settings
        gate_identity = jev_identity(llm_settings())
    panel = load_csv(csv_path) if csv_path else make_demo(seed=config.seed)
    if config.top_k > panel["close"].shape[1]:
        raise ValueError("top_k exceeds the number of assets")
    total = len(panel["close"])
    development_end = int(total * (1 - config.holdout_fraction))
    # Two bars excluded before final test; development feedback cannot see them.
    development_end -= 2
    if development_end - config.warmup < config.folds * 40:
        raise ValueError("Insufficient data for warmup and at least 40 sessions per fold")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    directory = root / run_id
    directory.mkdir(parents=True)
    snapshot = directory / "snapshot.csv"
    save_panel(panel, snapshot)
    # Evaluate reloaded serialized values so reruns use exactly the same prices.
    panel = load_csv(snapshot)
    kind, label = ("csv", Path(csv_path).name) if csv_path else ("synthetic", "Synthetic regime / noise demo")
    source = describe(panel, snapshot, kind, label)
    if csv_path and csv_path.with_suffix(".metadata.json").exists():
        supplied = read_json(csv_path.with_suffix(".metadata.json"))
        source["provider_metadata"] = supplied
        # Keep snapshot hash as the actual proof; metadata never replaces it.
    boundaries = np.linspace(config.warmup, development_end, config.folds + 1, dtype=int).tolist()
    fold_ranges = list(zip(boundaries[:-1], boundaries[1:], strict=True))
    index = panel["close"].index
    report = {"id": run_id, "status": "queued", "created_at": now(), "source": source,
              "config": config.to_dict(), "progress": {"completed": 0, "total": config.trials, "elapsed_seconds": 0},
              "split": {"development_start": str(index[config.warmup].date()),
                        "development_end": str(index[development_end - 1].date()),
                        "holdout_start": str(index[development_end + 2].date()), "holdout_end": str(index[-1].date()),
                        "embargo_sessions": 2,
                        "folds": [{"start": str(index[a].date()), "end": str(index[b - 1].date())} for a, b in fold_ranges]},
              "trials": [], "selected": None, "holdout": None, "events": [], "warnings": [],
              "provenance": {"alphaos_version": __version__, "python": platform.python_version(),
                             "numpy": np.__version__, "pandas": pd.__version__, "source_code_sha256": code_fingerprint(),
                             "factorminer": provenance()},
              "stop_reason": None,
              "_state": {"folds": fold_ranges, "development_end": development_end, "holdout_start": development_end + 2,
                         "llm_calls": 0, "llm_reserved_tokens": 0, "pending": None,
                         "imported_candidates": candidates or []}}
    if config.mode == "agent":
        import sklearn
        report["provenance"]["sklearn"] = sklearn.__version__
        report["_state"]["jev_gate_config"] = gate_identity
    if source["kind"] == "synthetic":
        report["warnings"].append("合成数据用于验证工程流程，收益指标不代表真实市场表现。")
    else:
        report["warnings"].append("数据按所选资产集合评估；未重建历史成分股，需自行确认复权与数据授权。")
    report["warnings"].append("开发集反复用于搜索；最终测试仅在选定因子后评估。重复创建实验仍会产生跨运行选择偏差。")
    event(report, "已冻结数据快照、时间切分和搜索预算")
    atomic_json(directory / "report.json", report)
    return directory


def _checkpoint(directory, report, clock_start, prior_elapsed):
    report["progress"]["completed"] = len(report["trials"])
    report["progress"]["elapsed_seconds"] = round(prior_elapsed + time.monotonic() - clock_start, 3)
    atomic_json(directory / "report.json", report)


def _correlation(left, right, start):
    a = left.iloc[start:].rank(axis=1, pct=True).to_numpy().ravel()
    b = right.iloc[start:].rank(axis=1, pct=True).to_numpy().ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 100 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def _export(directory, report):
    public_trials = []
    for trial in report["trials"]:
        row = {k: trial.get(k) for k in ["id", "name", "expression", "hypothesis", "origin", "status", "score", "reason", "complexity", "seconds"]}
        if report["config"]["mode"] == "agent":
            row.update({"model": trial.get("model"), "action": trial.get("action"),
                        "review_decision": trial.get("review", {}).get("decision")})
            row.update({key: json.dumps(trial.get(key), ensure_ascii=False) for key in ["features", "model_params", "parents"]})
            if trial.get("jev_gate") or trial.get("jev_usage"):
                row.update({"jev_decision": trial.get("jev_gate", {}).get("decision"),
                            "jev_gate": json.dumps(trial.get("jev_gate"), ensure_ascii=False),
                            "jev_usage": json.dumps(trial.get("jev_usage"), ensure_ascii=False)})
        row.update({"dev_" + k: v for k, v in trial.get("metrics", {}).items() if not isinstance(v, (list, dict))})
        public_trials.append(row)
    pd.DataFrame(public_trials).to_csv(directory / "trials.csv", index=False)
    atomic_json(directory / "config.json", report["config"])
    if report.get("selected"):
        if report["config"]["mode"] == "agent":
            strategy = {key: report["selected"][key] for key in ["features", "model", "model_params"]}
            code = ('"""Frozen strategy specification; retraining uses only pre-holdout matured labels."""\n'
                    'from alpharesearchos.predictive import strategy_scores\n\n'
                    f'STRATEGY = {strategy!r}\nFIT_CUTOFF = {report["_state"]["development_end"]}\n\n'
                    'def compute(panel):\n    return strategy_scores(STRATEGY, panel, fit_cutoff=FIT_CUTOFF)[0]\n')
            atomic_json(directory / "selected_candidate.json", report["selected"])
            atomic_json(directory / "training_audit.json", (report.get("holdout") or {}).get("training_audit", {}))
        else:
            expression = report["selected"]["expression"]
            code = ('"""Generated causal factor. Inspectable; executed by the bounded DSL interpreter."""\n'
                'from alpharesearchos.factors import evaluate_expression\n\n'
                f'EXPRESSION = {expression!r}\n\n'
                'def compute(panel):\n    return evaluate_expression(EXPRESSION, panel)\n')
        (directory / "selected_factor.py").write_text(code, encoding="utf-8")
    if report.get("research"):
        atomic_json(directory / "research_graph.json", report["research"]["graph"])
    from .reports import write_report
    write_report(directory, report)
    files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".") and p.name != "manifest.json"]
    atomic_json(directory / "manifest.json", {"run_id": report["id"], "files": {p.name: file_hash(p) for p in files}})


def run_research(directory: Path, *, stop_after: int | None = None, should_pause: Callable[[], bool] | None = None):
    """Resume the same frozen experiment; stop_after is for controlled checkpoints."""
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative integer")
    if should_pause is not None and not callable(should_pause):
        raise ValueError("should_pause must be callable")
    if read_json(directory / "report.json")["config"]["mode"] == "agent":
        from .agent_research import run_agent_research
        return run_agent_research(directory, stop_after=stop_after, should_pause=should_pause)
    with run_lock(directory):
        report = read_json(directory / "report.json")
        if file_hash(directory / "snapshot.csv") != report["source"]["sha256"]:
            raise ValueError("Snapshot checksum changed; create a new experiment")
        if report["status"] == "completed":
            if not (directory / "manifest.json").exists():
                _export(directory, report)
            return report
        config = ResearchConfig(**report["config"])
        if code_fingerprint() != report["provenance"]["source_code_sha256"]:
            raise ValueError("Research code changed since creation; create a new experiment")
        if report["provenance"]["numpy"] != np.__version__ or report["provenance"]["pandas"] != pd.__version__:
            raise ValueError("Numerical dependencies changed; restore uv.lock environment or create a new experiment")
        panel = load_csv(directory / "snapshot.csv")
        state = report["_state"]
        dev = {key: value.iloc[:state["development_end"]].copy() for key, value in panel.items()}
        clock_start, prior = time.monotonic(), report["progress"]["elapsed_seconds"]
        bt = BacktestConfig(cost_bps=config.cost_bps, top_k=config.top_k, rebalance_every=config.rebalance_every)
        report["status"] = "running"
        if state["pending"] is not None:
            pending = state["pending"]
            report["trials"].append({"id": pending["id"], "name": "中断实验", "status": "failed", "score": None,
                                     "reason": "Previous process ended during this attempt; budget consumed, no paid retry"})
            state["pending"] = None
            event(report, "恢复中断实验：保留已消耗预算", "warning")
        _checkpoint(directory, report, clock_start, prior)
        seen = {t["canonical"] for t in report["trials"] if t.get("canonical")}
        scores_cache = {}
        for t in report["trials"]:
            if t["status"] == "ok":
                scores_cache[t["id"]] = evaluate_expression(t["expression"], dev)
        try:
            while len(report["trials"]) < config.trials and not report.get("selected"):
                if should_pause is not None and should_pause():
                    report["status"], report["stop_reason"] = "paused", "user_pause"
                    event(report, "已暂停，候选和预算已保存")
                    _checkpoint(directory, report, clock_start, prior)
                    return report
                elapsed = prior + time.monotonic() - clock_start
                if elapsed >= config.max_seconds:
                    report["stop_reason"] = "time_budget"
                    break
                if stop_after is not None and len(report["trials"]) >= stop_after:
                    report["status"] = "paused"
                    report["stop_reason"] = "checkpoint"
                    _checkpoint(directory, report, clock_start, prior)
                    return report
                index = len(report["trials"])
                trial_id = f"T{index + 1:04d}"
                state["pending"] = {"id": trial_id, "started_at": now()}
                # CLI reservation admits a call; its output has no hard token cap.
                reserve = llm_token_reservation() if config.mode == "llm" else 0
                use_llm = (config.mode == "llm" and index >= max(2, len(state["imported_candidates"])) and state["llm_calls"] < config.max_llm_calls
                           and state["llm_reserved_tokens"] + reserve <= config.max_llm_tokens)
                if use_llm:
                    state["llm_calls"] += 1
                    state["llm_reserved_tokens"] += reserve
                _checkpoint(directory, report, clock_start, prior)
                trial = {"id": trial_id, "status": "failed", "score": None}
                started = time.monotonic()
                try:
                    imported = state["imported_candidates"]
                    if index < len(imported):
                        trial.update(imported[index])
                    elif use_llm:
                        candidate, usage = llm_proposal(config.direction, report["trials"], timeout=config.max_seconds - elapsed)
                        trial.update(candidate)
                        trial["llm_usage"] = usage
                        reported = usage.get("reported_total_tokens")
                        if type(reported) is int and reported > reserve:
                            state["llm_reserved_tokens"] += reported - reserve
                    else:
                        trial.update(local_proposal(index, config.seed, report["trials"], config.direction))
                        if config.mode == "llm" and index >= 2:
                            trial["origin"] = "local_budget_fallback"
                    info = validate_expression(trial["expression"])
                    trial.update({"canonical": info["canonical"], "complexity": info["complexity"]})
                    if info["complexity"] > config.max_complexity or info["lookback"] > config.warmup:
                        raise ValueError("Expression exceeds complexity or warmup budget")
                    if info["canonical"] in seen:
                        raise ValueError("Duplicate expression")
                    seen.add(info["canonical"])
                    check = causality_check(trial["expression"], dev)
                    trial["causality"] = check
                    if not check["passed"]:
                        raise ValueError("Causality audit failed: " + str(check))
                    scores = evaluate_expression(trial["expression"], dev)
                    parity = differential_check(trial["expression"], dev, scores)
                    trial["engine_parity"] = parity
                    if parity.get("status") == "failed":
                        raise ValueError("FactorMiner differential audit failed: " + str(parity))
                    finite = np.isfinite(scores.iloc[config.warmup:].to_numpy())
                    if finite.mean() < 0.95:
                        raise ValueError("Less than 95% finite development signal coverage")
                    if float(scores.iloc[config.warmup:].std(axis=1).fillna(0).max()) < 1e-12:
                        raise ValueError("No cross-sectional signal variation")
                    for other_id, other in scores_cache.items():
                        if _correlation(scores, other, config.warmup) > 0.998:
                            raise ValueError(f"Numerically redundant with {other_id} (rank correlation > 0.998)")
                    result = evaluate_development(scores, dev["close"], bt, state["folds"])
                    trial.update(result)
                    trial["score"] = float(result["score"]) - 0.002 * info["complexity"]
                    if not np.isfinite(trial["score"]):
                        raise ValueError("Non-finite development score")
                    trial["status"] = "ok"
                    scores_cache[trial_id] = scores
                except (ValueError, SyntaxError) as exc:
                    trial["status"], trial["reason"] = "rejected", str(exc)[:1500]
                except Exception as exc:
                    trial["status"], trial["reason"] = "failed", f"{type(exc).__name__}: {str(exc)[:700]}"
                trial["seconds"] = round(time.monotonic() - started, 4)
                report["trials"].append(trial)
                state["pending"] = None
                event(report, f"{trial_id} {trial.get('name', 'proposal')}: {trial['status']}", "info" if trial["status"] == "ok" else "warning")
                _checkpoint(directory, report, clock_start, prior)
            if should_pause is not None and should_pause() and not report.get("selected"):
                report["status"], report["stop_reason"] = "paused", "user_pause"
                event(report, "已暂停，候选和预算已保存")
                _checkpoint(directory, report, clock_start, prior)
                return report
            accepted = [t for t in report["trials"] if t["status"] == "ok"]
            if not accepted:
                report["status"] = "failed"
                report["stop_reason"] = "no_valid_candidates"
                event(report, "没有候选通过开发集检查；请查看各实验失败原因", "error")
            else:
                # Selection is checkpointed BEFORE final test access, including after a crash.
                if not report.get("selected"):
                    report["selected"] = max(accepted, key=lambda t: t["score"]).copy()
                    report["selection_frozen_at"] = now()
                    event(report, f"冻结开发集优胜因子 {report['selected']['id']}，开始最终测试")
                    _checkpoint(directory, report, clock_start, prior)
                expression = report["selected"]["expression"]
                scores = evaluate_expression(expression, panel)
                a, b = state["holdout_start"], len(panel["close"])
                curve = backtest(scores, panel["close"], bt, a, b)
                metrics = summarize(curve)
                baseline_curve = curve.copy()
                baseline_curve["return"] = curve["benchmark_return"]
                baseline_curve["equity"] = curve["benchmark_equity"]
                baseline_curve["turnover"] = curve["benchmark_turnover"]
                baseline_metrics = summarize(baseline_curve)
                momentum = evaluate_expression("rank(ret(close, 60))", panel)
                momentum_metrics = summarize(backtest(momentum, panel["close"], bt, a, b))
                stress = [{"cost_bps": cost, "metrics": summarize(backtest(scores, panel["close"], replace(bt, cost_bps=cost), a, b))}
                          for cost in sorted({0.0, config.cost_bps, config.cost_bps * 2, config.cost_bps * 4})]
                display_curve = curve.reset_index()
                display_curve = display_curve.rename(columns={display_curve.columns[0]: "date"})
                display_curve["date"] = display_curve["date"].dt.strftime("%Y-%m-%d")
                report["holdout"] = {"metrics": metrics, "baseline_metrics": baseline_metrics,
                                     "momentum_metrics": momentum_metrics, "stress": stress,
                                     "curve": display_curve.to_dict("records")}
                try:
                    from .backtest import bootstrap_sharpe_interval
                    report["holdout"]["sharpe_interval"] = bootstrap_sharpe_interval(curve["return"].to_numpy(), seed=config.seed)
                except ImportError:
                    pass
                curve.to_csv(directory / "holdout_curve.csv", index_label="date")
                report["status"] = "completed"
                report["stop_reason"] = report["stop_reason"] if report["stop_reason"] == "time_budget" else "trial_budget"
                report["conclusion"] = ("最终测试净收益高于等权基线" if metrics["excess_return"] > 0 else "最终测试未超过等权基线")
                if metrics["excess_return"] <= 0:
                    report["warnings"].append("此次搜索未在最终测试中优于等权基线，不能据此采用为交易策略。")
                event(report, report["conclusion"])
            _checkpoint(directory, report, clock_start, prior)
            _export(directory, report)
            return report
        except BaseException as exc:
            report["status"] = "paused" if isinstance(exc, KeyboardInterrupt) else "failed"
            event(report, f"研究中断：{type(exc).__name__}: {str(exc)[:300]}", "error")
            _checkpoint(directory, report, clock_start, prior)
            raise


def verify_run(directory: Path):
    manifest = read_json(directory / "manifest.json")
    failures = [name for name, digest in manifest["files"].items()
                if not (directory / name).is_file() or file_hash(directory / name) != digest]
    return {"ok": not failures, "run_id": manifest["run_id"], "checked": len(manifest["files"]), "failures": failures}
