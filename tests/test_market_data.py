"""Yahoo adapter contracts use deterministic frames and never fetch market data."""

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from alpharesearchos.data import FIELDS, download_yahoo, file_hash, load_csv, make_demo


@pytest.fixture
def market(monkeypatch):
    panel = make_demo(seed=19, sessions=400, assets=3)
    frames = {}
    for symbol in panel["close"].columns:
        frames[symbol] = pd.DataFrame({field.title(): panel[field][symbol] for field in FIELDS})
        frames[symbol].index = frames[symbol].index.tz_localize("America/New_York")
    calls = []

    def ticker(symbol):
        def history(**kwargs):
            calls.append((symbol, kwargs))
            return frames[symbol].copy()
        return SimpleNamespace(history=history)

    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(Ticker=ticker))
    return frames, calls


def test_adjusted_download_preserves_sessions_and_records_actual_unique_universe(market, tmp_path):
    frames, calls = market
    destination = tmp_path / "datasets" / "market.csv"
    metadata = download_yahoo(["SIM03", "SIM01", "SIM02", "SIM01"], "2020-01-01", "2022-01-01", destination)
    assert calls == [(symbol, {"start": "2020-01-01", "end": "2022-01-01", "auto_adjust": True,
                               "actions": False, "timeout": 20}) for symbol in sorted(frames)]
    saved = load_csv(destination)
    assert list(saved["close"].columns) == metadata["symbols"] == ["SIM01", "SIM02", "SIM03"]
    for field in FIELDS:
        for symbol in frames:
            pd.testing.assert_series_equal(saved[field][symbol], frames[symbol][field.title()].tz_localize(None),
                                           check_names=False, check_freq=False, rtol=1e-10)
    assert metadata["auto_adjust"] is True
    assert metadata["requested_end_exclusive"] == "2022-01-01"
    assert metadata["sha256"] == file_hash(destination)
    assert metadata["rows"] == 400
    assert json.loads(destination.with_suffix(".metadata.json").read_text()) == metadata


@pytest.mark.parametrize("problem", ["empty", "missing_session", "nan", "invalid_high", "missing_column"])
def test_bad_provider_data_never_replaces_a_previous_snapshot(market, tmp_path, problem):
    frames, _ = market
    if problem == "empty":
        frames["SIM02"] = frames["SIM02"].iloc[:0]
    elif problem == "missing_session":
        frames["SIM02"] = frames["SIM02"].iloc[1:]
    elif problem == "nan":
        frames["SIM02"].iloc[10, 3] = float("nan")
    elif problem == "invalid_high":
        frames["SIM02"].iloc[10, 1] = 1
    else:
        frames["SIM02"] = frames["SIM02"].drop(columns="Volume")
    destination = tmp_path / "market.csv"
    destination.write_text("previous snapshot")
    destination.with_suffix(".metadata.json").write_text("previous metadata")
    with pytest.raises((ValueError, KeyError)):
        download_yahoo(list(frames), "2020-01-01", "2022-01-01", destination)
    assert destination.read_text() == "previous snapshot"
    assert destination.with_suffix(".metadata.json").read_text() == "previous metadata"


@pytest.mark.parametrize(("start", "end"), [
    ("2022-01-01", "2020-01-01"), ("2020-01-01", "2020-01-01"),
    ("2020-02-31", "2022-01-01"), ("2020-1-1", "2022-01-01"),
    ("2020-01-01T00:00:00", "2022-01-01"), (None, "2022-01-01"),
])
def test_invalid_dates_fail_before_any_market_request(market, tmp_path, start, end):
    _, calls = market
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        download_yahoo(["SIM01", "SIM02", "SIM03"], start, end, tmp_path / "market.csv")
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_invalid_universe_and_missing_optional_dependency_are_explicit(market, tmp_path, monkeypatch):
    _, calls = market
    with pytest.raises(ValueError, match="3–100"):
        download_yahoo(["SIM01", "SIM01", "SIM02"], "2020-01-01", "2022-01-01", tmp_path / "market.csv")
    assert calls == []
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(RuntimeError, match="market"):
        download_yahoo(["SIM01", "SIM02", "SIM03"], "2020-01-01", "2022-01-01", tmp_path / "market.csv")
    assert list(tmp_path.iterdir()) == []
