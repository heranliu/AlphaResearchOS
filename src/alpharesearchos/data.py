"""Strict rectangular OHLCV snapshots: no filling future/missing prices."""

from __future__ import annotations

import hashlib
import json
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
    frame = pd.read_csv(path, dtype={"symbol": str})
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
    metadata.update({"symbols": symbols, "requested_start": start, "requested_end_exclusive": end,
                     "auto_adjust": True, "downloaded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                     "survivorship_note": "User-chosen current universe; no historical constituent reconstruction"})
    destination.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata
