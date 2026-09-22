"""Freeze and evaluate explicitly selected factors, without another search.

These are retrospective fixed-factor experiments, not untouched holdout tests.
Factor source periods and any date overlap are retained in the report. The
existing causal interpreter and self-financing portfolio engine are reused.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .backtest import BacktestConfig, backtest, summarize
from .data import describe, file_hash, load_csv, make_demo, save_panel
from .factors import causality_check, evaluate_expression, validate_expression
from .library import factor_id
from .store import atomic_json, read_json, run_lock

BACKTEST_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[a-f0-9]{8}$")
ARTIFACTS = frozenset({"report.json", "curve.csv", "factors.json", "config.json", "snapshot.csv"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date(value: Any, name: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must use YYYY-MM-DD.")
    try:
        return pd.Timestamp(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}.") from exc


def _overlap(factors: list[dict], source: dict, start: str, end: str) -> list[dict]:
    records = []

    def intersects(first, last):
        try:
            left, right = _date(first, "source start"), _date(last, "source end")
            if left is None or right is None:
                return None
            return bool(left <= pd.Timestamp(end) and right >= pd.Timestamp(start))
        except (ValueError, TypeError):
            return None

    for factor in factors:
        for provenance in factor.get("provenance", []):
            split = provenance.get("split", {})
            prior_source = provenance.get("source", {})
            records.append({
                "factor_id": factor["id"], "run_id": provenance.get("run_id"),
                "trial_id": provenance.get("trial_id"),
                "development_overlap": intersects(split.get("development_start"), split.get("development_end")),
                "holdout_overlap": intersects(split.get("holdout_start"), split.get("holdout_end")),
                "same_snapshot": bool(prior_source.get("sha256") and prior_source["sha256"] == source["sha256"]),
                "source": deepcopy(prior_source), "split": deepcopy(split),
            })
    return records


def create_backtest(
    backtests_root: Path,
    dataset_root: Path,
    body: dict[str, Any],
    factors: list[dict[str, Any]] | dict[str, dict[str, Any]],
) -> Path:
    """Validate, freeze input artifacts, and save a queued report.

    Requested start/end dates are inclusive. Non-session dates inside the
    available span map to the first/last included session, which is explicitly
    recorded. An explicit start with insufficient past-only warmup is rejected.
    """
    allowed = {"factor_ids", "dataset", "cost_bps", "top_k", "rebalance_every", "start_date", "end_date", "combination"}
    if not isinstance(body, dict) or set(body) - allowed:
        raise ValueError("Backtest request contains unknown fields or is not a JSON object.")
    selected_ids = body.get("factor_ids")
    if not isinstance(selected_ids, list) or not 1 <= len(selected_ids) <= 10:
        raise ValueError("Select 1–10 factor_ids.")
    if any(not isinstance(value, str) for value in selected_ids) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("factor_ids must be distinct strings.")
    available = factors if isinstance(factors, dict) else {record["id"]: record for record in factors}
    if any(identifier not in available for identifier in selected_ids):
        raise ValueError("One or more selected factors are no longer available in the library.")
    selected = [deepcopy(available[identifier]) for identifier in selected_ids]
    lookbacks = []
    for record in selected:
        info = validate_expression(record.get("expression", ""))
        if record.get("id") != factor_id(info["canonical"]):
            raise ValueError("A selected factor ID does not match its frozen expression.")
        record["expression"] = info["canonical"]
        record["lookback"] = info["lookback"]
        lookbacks.append(info["lookback"])

    dataset = body.get("dataset", "demo")
    if not isinstance(dataset, str):
        raise ValueError("dataset must be demo or the name of a registered CSV dataset.")
    if dataset == "demo":
        panel = make_demo(seed=42)
        kind, label = "synthetic", "Synthetic regime / noise demo"
    else:
        root = Path(dataset_root).resolve()
        if Path(dataset).name != dataset or not dataset.endswith(".csv") or "\\" in dataset:
            raise ValueError("Choose a registered CSV dataset by filename.")
        path = root / dataset
        if not path.is_file() or path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Dataset not found or is not a registered local CSV.")
        panel = load_csv(path)
        kind, label = "csv", dataset

    cost = body.get("cost_bps", 10.0)
    if type(cost) not in {int, float} or not math.isfinite(cost) or not 0 <= cost <= 200:
        raise ValueError("cost_bps must be a finite number between 0 and 200.")
    top_k, every = body.get("top_k", 3), body.get("rebalance_every", 5)
    if type(top_k) is not int or not 1 <= top_k <= panel["close"].shape[1]:
        raise ValueError("top_k must be an integer between 1 and the number of assets.")
    if type(every) is not int or not 1 <= every <= 63:
        raise ValueError("rebalance_every must be an integer from 1 to 63.")
    if body.get("combination", "equal_rank") != "equal_rank":
        raise ValueError("Only equal_rank combination is supported.")
    requested_start = _date(body.get("start_date"), "start_date")
    requested_end = _date(body.get("end_date"), "end_date")
    index = panel["close"].index
    warmup = max(lookbacks) + 2
    if warmup >= len(index):
        raise ValueError("The dataset is too short for the selected factors and two-bar execution lag.")
    for value, name in [(requested_start, "start_date"), (requested_end, "end_date")]:
        if value is not None and not index[0] <= value <= index[-1]:
            raise ValueError(f"{name} is outside the dataset's available date range.")
    if requested_start is not None and requested_end is not None and requested_start > requested_end:
        raise ValueError("start_date must be on or before end_date.")
    start = int(index.searchsorted(requested_start, side="left")) if requested_start is not None else warmup
    end = int(index.searchsorted(requested_end, side="right")) if requested_end is not None else len(index)
    if start < warmup:
        raise ValueError(f"Insufficient warmup: earliest eligible start is {index[warmup].date()} ({warmup} prior sessions including execution lag).")
    if start >= end:
        raise ValueError("The requested period has no eligible trading sessions after factor warmup.")

    identifier = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    directory = Path(backtests_root) / identifier
    directory.mkdir(parents=True)
    save_panel(panel, directory / "snapshot.csv")
    # The serialized data is the canonical source for both queued and replayed jobs.
    panel = load_csv(directory / "snapshot.csv")
    source = describe(panel, directory / "snapshot.csv", kind, label)
    config = {
        "factor_ids": selected_ids, "dataset": dataset, "cost_bps": float(cost), "top_k": top_k,
        "rebalance_every": every, "signal_lag": 2, "annualization": 252,
        "combination": "equal_rank", "start_date": str(requested_start.date()) if requested_start is not None else None,
        "end_date": str(requested_end.date()) if requested_end is not None else None,
        "demo_seed": 42 if dataset == "demo" else None,
    }
    period = {
        "requested_start": config["start_date"], "requested_end": config["end_date"],
        "start": str(index[start].date()), "end": str(index[end - 1].date()),
        "sessions": end - start, "warmup_sessions": warmup,
    }
    warnings = ["固定因子历史回测：不重新搜索或择优，也不构成独立未见样本检验。"]
    if kind == "synthetic":
        warnings.append("合成数据用于验证流程，收益不代表真实市场表现。")
    overlaps = _overlap(selected, source, period["start"], period["end"])
    if any(item["development_overlap"] or item["holdout_overlap"] for item in overlaps):
        warnings.append("回测日期与来源研究的开发或最终测试日期重叠；请结合来源数据理解结果。")
    report = {
        "id": identifier, "status": "queued", "created_at": _now(), "kind": "fixed_factor_retrospective",
        "source": source, "config": config, "factors": selected, "period": period,
        "metrics": None, "baseline_metrics": None, "curve": [], "overlap": overlaps,
        "causality": [], "warnings": warnings, "error": None,
        "provenance": {"alphaos_version": __version__, "protocol": "fixed selection; equal-rank combination; lag 2; cold start and liquidation"},
        "_state": {"start": start, "end": end},
    }
    atomic_json(directory / "factors.json", {"factors": selected})
    atomic_json(directory / "config.json", config)
    report["input_sha256"] = {name: file_hash(directory / name) for name in ["snapshot.csv", "factors.json", "config.json"]}
    atomic_json(directory / "report.json", report)
    return directory


def run_backtest(directory: Path) -> dict[str, Any]:
    """Run the frozen selection once; persist an honest failed status on errors."""
    directory = Path(directory)
    with run_lock(directory):
        report = read_json(directory / "report.json")
        if report.get("status") == "completed":
            return report
        started = time.monotonic()
        report.update({"status": "running", "error": None, "metrics": None, "baseline_metrics": None, "curve": []})
        atomic_json(directory / "report.json", report)
        try:
            for name in ("snapshot.csv", "factors.json", "config.json"):
                if file_hash(directory / name) != report["input_sha256"].get(name):
                    raise ValueError(f"Frozen input checksum changed: {name}. Create a new backtest.")
            config = read_json(directory / "config.json")
            selected = read_json(directory / "factors.json")["factors"]
            full_panel = load_csv(directory / "snapshot.csv")
            start, end = report["_state"]["start"], report["_state"]["end"]
            # Later snapshot rows cannot participate in either factor evaluation
            # or the causality audit for an earlier requested end date.
            panel = {field: frame.iloc[:end] for field, frame in full_panel.items()}
            scored = []
            report["causality"] = []
            for record in selected:
                audit = causality_check(record["expression"], panel)
                report["causality"].append({"factor_id": record["id"], **audit})
                if not audit["passed"]:
                    raise ValueError(f"Factor {record['name']} failed its causality audit: {audit['detail']}")
                values = evaluate_expression(record["expression"], panel)
                decisions = values.iloc[start - 2:end - 2]
                if not np.isfinite(decisions.to_numpy()).any():
                    raise ValueError(f"Factor {record['name']} has no valid signals in the requested period.")
                scored.append(values.rank(axis=1, method="average", pct=True, na_option="keep"))
            # NaN in any constituent keeps the combined asset ineligible. No
            # varying hidden factor weights or score imputation are introduced.
            combined = np.mean(np.stack([value.to_numpy() for value in scored]), axis=0)
            scores = pd.DataFrame(combined, index=panel["close"].index, columns=panel["close"].columns)
            if not np.isfinite(scores.iloc[start - 2:end - 2].to_numpy()).any():
                raise ValueError("Selected factors have no jointly valid signals in the requested period.")
            bt = BacktestConfig(**{key: config[key] for key in ["cost_bps", "top_k", "rebalance_every", "signal_lag", "annualization"]})
            curve = backtest(scores, panel["close"], bt, start, end)
            baseline = curve.copy()
            baseline["return"] = curve["benchmark_return"]
            baseline["equity"] = curve["benchmark_equity"]
            baseline["turnover"] = curve["benchmark_turnover"]
            # Every allowed period begins after warmup, hence at row >= 2.
            # The benchmark invests at that first row and stays fully invested
            # throughout each return interval before terminal liquidation.
            baseline["exposure"] = 1.0
            report["metrics"] = summarize(curve, bt.annualization)
            report["baseline_metrics"] = summarize(baseline, bt.annualization)
            for equity, name in [("equity", "drawdown"), ("benchmark_equity", "benchmark_drawdown")]:
                values = curve[equity].to_numpy()
                peaks = np.maximum.accumulate(np.concatenate(([1.0], values)))[1:]
                curve[name] = values / peaks - 1.0
            if not (curve["exposure"] > 0).any():
                report["warnings"].append("实际调仓日没有可用组合信号，组合全程持有现金。")
            curve.to_csv(directory / "curve.csv", index_label="date")
            display = curve.reset_index()
            display = display.rename(columns={display.columns[0]: "date"})
            display["date"] = display["date"].dt.strftime("%Y-%m-%d")
            report["curve"] = display.to_dict("records")
            report.update({"status": "completed", "completed_at": _now()})
        except Exception as exc:
            report.update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:1200]}"})
        report["elapsed_seconds"] = round(time.monotonic() - started, 4)
        atomic_json(directory / "report.json", report)
        return report


def list_backtests(root: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(Path(root).glob("*/report.json"), reverse=True):
        if not BACKTEST_ID.fullmatch(path.parent.name) or path.parent.is_symlink() or path.is_symlink():
            continue
        try:
            report = read_json(path)
            results.append({key: report.get(key) for key in ["id", "status", "created_at", "kind", "source", "config", "factors", "period", "metrics", "error"]})
        except (ValueError, OSError, AttributeError):
            continue
    return results
