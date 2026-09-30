"""Strict rectangular OHLCV snapshots: no filling future/missing prices."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume")


def validate_panel(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(FIELDS) - set(panel):
        raise ValueError(f"Missing fields: {set(FIELDS) - set(panel)}")
    close = panel["close"]
    if len(close) < 400 or len(close) > 20000 or not 3 <= close.shape[1] <= 100:
        raise ValueError("Data must contain 400–20000 sessions and 3–100 assets")
    if not isinstance(close.index, pd.DatetimeIndex) or close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and ascending")
    if close.columns.has_duplicates:
        raise ValueError("Duplicate asset names")
    for field in FIELDS:
        frame = panel[field]
        if not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError(f"{field}: inconsistent dates/assets")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{field}: missing or non-finite observations; use a complete common window")
        if (frame < 0).any().any() or (field != "volume" and (frame <= 0).any().any()):
            raise ValueError(f"{field}: invalid nonpositive prices or negative volume")
    if (panel["high"] < np.maximum(panel["open"], panel["close"]) - 1e-7).any().any():
        raise ValueError("High must cover open and close")
    if (panel["low"] > np.minimum(panel["open"], panel["close"]) + 1e-7).any().any():
        raise ValueError("Low must cover open and close")
    return panel


def make_demo(seed: int = 42, sessions: int = 1260, assets: int = 8):
    """Synthetic regime/noise data for engineering validation, not alpha evidence."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.00012, 0.008, sessions)
    noise = rng.normal(0, 0.009, (sessions, assets))
    # Random asset/regime drifts are independent of any proposed factor.
    drifts = rng.normal(0, 0.00045, (1 + sessions // 126, assets))
    log_ret = market[:, None] * rng.uniform(0.4, 1.2, assets) + noise
    log_ret += drifts[np.arange(sessions) // 126]
    close = 100 * np.exp(np.cumsum(log_ret, axis=0))
    open_ = np.vstack([close[0], close[:-1]]) * np.exp(rng.normal(0, 0.002, close.shape))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, close.shape))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, close.shape))
    volume = rng.lognormal(14, 0.4, close.shape)
    index = pd.bdate_range("2020-01-02", periods=sessions)
    columns = [f"SIM{i + 1:02d}" for i in range(assets)]
    return {f: pd.DataFrame(a, index=index, columns=columns) for f, a in zip(FIELDS, [open_, high, low, close, volume], strict=True)}


def to_long(panel):
    frames = []
    for symbol in panel["close"].columns:
        frame = pd.DataFrame({f: panel[f][symbol] for f in FIELDS})
        frame.insert(0, "symbol", symbol)
        frame.index.name = "date"
        frames.append(frame.reset_index())
    return pd.concat(frames).sort_values(["date", "symbol"]).reset_index(drop=True)


def save_panel(panel, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_long(panel).to_csv(path, index=False, float_format="%.12g", date_format="%Y-%m-%d")


def load_csv(path: str | Path):
    path = Path(path)
    if path.stat().st_size > 200_000_000:
        raise ValueError("CSV exceeds 200 MB limit")
    return _read_panel(path)


def _read_panel(source):
    frame = pd.read_csv(source, dtype={"symbol": str})
    required = {"date", "symbol", *FIELDS}
    if required - set(frame.columns):
        raise ValueError(f"CSV needs columns {sorted(required)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.tz_convert(None)
    if frame["symbol"].isna().any() or frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Missing symbol or duplicate date/symbol rows")
    if (frame["date"] != frame["date"].dt.normalize()).any():
        raise ValueError("Daily observations must use date-only timestamps")
    panel = {f: frame.pivot(index="date", columns="symbol", values=f).sort_index().astype(float) for f in FIELDS}
    return validate_panel(panel)


def inspect_csv(content: str) -> dict:
    """Describe an upload without writing, filling, or silently discarding rows.

    Coverage uses dates actually present in the upload, not an assumed exchange
    calendar. Eligibility refers to the workbench's default research protocol.
    """
    result = {
        "rows": 0, "assets": 0, "sessions": 0, "start": None, "end": None,
        "missing_values": 0, "duplicate_rows": 0, "missing_records": 0,
        "asset_coverage": [], "common_range": {"start": None, "end": None, "sessions": 0},
        "importable": False, "eligible_for_research": False, "eligible_for_backtest": False,
        "issues": [], "adjustment": {
            "status": "unknown",
            "message": "CSV 本身无法确认复权口径，请核对数据源；系统不会自动复权或补齐数据。",
        },
    }

    def issue(code, message, count=1, severity="error"):
        result["issues"].append({"code": code, "severity": severity, "message": message, "count": count})

    required = ["date", "symbol", *FIELDS]
    coverage, dates, pairs = {}, set(), set()
    invalid_dates = invalid_numbers = invalid_prices = invalid_ranges = 0
    try:
        reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")), strict=True)
        header = reader.fieldnames or []
        missing = [field for field in required if field not in header]
        if missing:
            issue("missing_columns", "缺少必需列：" + ", ".join(missing) + "。", len(missing))
            return result
        if any(header.count(field) != 1 for field in required):
            issue("duplicate_columns", "date、symbol 和 OHLCV 列名不能重复。")
            return result
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(field) is None for field in required):
                raise ValueError(f"第 {row_number} 行的列数与表头不一致。")
            result["rows"] += 1
            result["missing_values"] += sum(not row[field].strip() for field in required)
            symbol, day = row["symbol"], row["date"]
            valid_day = False
            try:
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                    date.fromisoformat(day)
                    valid_day = True
            except ValueError:
                pass
            if not valid_day:
                invalid_dates += 1
            if symbol.strip():
                asset = coverage.setdefault(symbol, {"rows": 0, "dates": set()})
                asset["rows"] += 1
                if valid_day:
                    asset["dates"].add(day)
                    dates.add(day)
                    pair = (day, symbol)
                    result["duplicate_rows"] += pair in pairs
                    pairs.add(pair)
            if len(coverage) > 100 or len(dates) > 20000:
                raise ValueError("最多支持 100 个资产和 20,000 个交易日。")
            values = {}
            for field in FIELDS:
                try:
                    value = float(row[field])
                except ValueError:
                    invalid_numbers += 1
                    continue
                if not math.isfinite(value):
                    invalid_numbers += 1
                    continue
                values[field] = value
                if value < 0 or (field != "volume" and value == 0):
                    invalid_prices += 1
            if all(field in values for field in ("open", "high", "low", "close")):
                invalid_ranges += (values["high"] < max(values["open"], values["close"]) - 1e-7
                                   or values["low"] > min(values["open"], values["close"]) + 1e-7)
    except csv.Error:
        raise ValueError("CSV 格式有误，请检查引号配对、逗号分隔和单元格长度。") from None
    result.update(assets=len(coverage), sessions=len(dates), start=min(dates) if dates else None,
                  end=max(dates) if dates else None)
    common = set(dates)
    for symbol, asset in sorted(coverage.items()):
        observed = asset["dates"]
        common.intersection_update(observed)
        missing_sessions = len(dates - observed)
        result["missing_records"] += missing_sessions
        result["asset_coverage"].append({
            "symbol": symbol, "rows": asset["rows"], "sessions": len(observed),
            "start": min(observed) if observed else None, "end": max(observed) if observed else None,
            "missing_sessions": missing_sessions,
        })
    result["common_range"] = {"start": min(common) if common else None,
                              "end": max(common) if common else None, "sessions": len(common)}
    for code, count, message in [
        ("missing_values", result["missing_values"], "必需字段存在空值，请核对源数据。"),
        ("duplicate_rows", result["duplicate_rows"], "同一 date、symbol 只能有一行，请检查重复记录。"),
        ("missing_records", result["missing_records"], "资产日期覆盖不一致；请从源数据补齐记录，或自行选择完整的共同区间。"),
        ("invalid_dates", invalid_dates, "date 必须是有效的 YYYY-MM-DD 日期，不含时分秒。"),
        ("invalid_numbers", invalid_numbers, "OHLCV 必须是有限数字，不能包含空值、NaN 或无穷值。"),
        ("invalid_prices", invalid_prices, "OHLC 价格必须大于 0，volume 必须大于等于 0。"),
        ("invalid_ranges", invalid_ranges, "high 和 low 必须覆盖同一行的 open 和 close。"),
    ]:
        if count:
            issue(code, message, count)
    if not 3 <= len(coverage) <= 100:
        issue("asset_count", "数据需要 3–100 个资产。", len(coverage))
    if not 400 <= len(dates) <= 20000:
        issue("session_count", "数据需要 400–20,000 个共同交易日。", len(dates))
    if not result["issues"]:
        # Match the importer exactly, including pandas' missing-value parsing.
        try:
            _read_panel(io.StringIO(content))
        except (ValueError, TypeError, KeyError):
            issue("invalid_panel", "CSV 无法按完整 OHLCV 面板读取，请检查日期、资产名称和数值字段。")
    result["importable"] = not result["issues"]
    result["eligible_for_backtest"] = result["importable"]
    result["eligible_for_research"] = result["importable"] and int(len(dates) * .8) - 2 - 252 >= 3 * 40
    if result["importable"] and not result["eligible_for_research"]:
        issue("research_history", "可用于固定回测；默认自主研究需要至少 468 个共同交易日（252 日预热、20% 留出和三折验证）。",
              len(dates), "warning")
    issue("calendar_scope", "日期覆盖仅比较文件内各资产的日期，不能确认所有资产同时缺失的交易日。", 1, "warning")
    return result


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(panel, path: Path, kind: str, label: str):
    close = panel["close"]
    return {"kind": kind, "label": label, "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()), "assets": list(close.columns), "rows": len(close),
            "sha256": file_hash(path)}


def download_yahoo(symbols: list[str], start: str, end: str, destination: Path):
    """Fetch adjusted daily bars; fail explicitly when any session is missing."""
    try:
        if any(not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value)
               for value in (start, end)) or date.fromisoformat(start) >= date.fromisoformat(end):
            raise ValueError
    except ValueError:
        raise ValueError("Choose valid YYYY-MM-DD dates with start before the exclusive end") from None
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install market support: pip install '.[market]'") from exc
    if not 3 <= len(set(symbols)) <= 100:
        raise ValueError("Choose 3–100 unique symbols")
    series = {}
    for symbol in sorted(set(symbols)):
        bars = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=True, actions=False, timeout=20)
        if bars.empty:
            raise ValueError(f"No data for {symbol}; check network/symbol/date range")
        bars.index = pd.to_datetime(bars.index.date)
        series[symbol] = bars.rename(columns=str.lower)[list(FIELDS)]
    panel = {f: pd.DataFrame({s: bars[f] for s, bars in series.items()}).sort_index() for f in FIELDS}
    validate_panel(panel)
    save_panel(panel, destination)
    metadata = describe(panel, destination, "yahoo", "Yahoo Finance / adjusted daily OHLCV")
    metadata.update({"symbols": sorted(series), "requested_start": start, "requested_end_exclusive": end,
                     "auto_adjust": True, "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                     "survivorship_note": "User-chosen current universe; no historical constituent reconstruction"})
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata
