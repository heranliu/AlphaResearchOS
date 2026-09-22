"""Persisted factor reuse and fixed-selection retrospective experiment checks."""

from copy import deepcopy
from datetime import timedelta

import pytest

from alpharesearchos.backtest import BacktestConfig, backtest, summarize
from alpharesearchos.data import file_hash, load_csv, make_demo, save_panel
from alpharesearchos.independent_backtest import create_backtest, list_backtests, run_backtest
from alpharesearchos.library import FactorLibrary, factor_id
from alpharesearchos.store import atomic_json, read_json


@pytest.fixture
def library(tmp_path):
    return FactorLibrary(tmp_path / "runs", tmp_path / "state")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "datasets"
    path = root / "prices.csv"
    save_panel(make_demo(seed=23, sessions=420, assets=4), path)
    return root, load_csv(path)


def _import(library, expression="rank(ret(close, 20))", name="momentum"):
    return library.import_factors({"factors": [{"name": name, "expression": expression, "hypothesis": "test hypothesis"}]})["factors"][0]


def _create(tmp_path, registered, records, **body):
    request = {"factor_ids": [record["id"] for record in records], "dataset": "prices.csv", "top_k": 2}
    request.update(body)
    return create_backtest(tmp_path / "backtests", registered[0], request, records)


def test_library_aggregates_only_valid_trials_with_real_development_provenance(library, tmp_path):
    expression = "rank(ret(close, 20))"
    for index in range(2):
        report = {
            "id": f"run{index}", "created_at": f"2026-09-1{index}T00:00:00Z",
            "source": {"kind": "csv", "label": f"dataset{index}", "sha256": str(index)},
            "split": {"development_start": "2020-01-02", "development_end": "2021-01-01"},
            "trials": [
                {"id": "T1", "name": f"tested{index}", "status": "ok", "expression": expression,
                 "score": index + 0.2, "metrics": {"total_return": index / 10}, "causality": {"passed": True}},
                {"id": "T2", "status": "rejected", "expression": "close"},
                {"id": "T3", "status": "ok", "expression": "delay(close, -1)"},
                {"id": "T4", "name": f"same{index}", "status": "ok", "expression": expression,
                 "score": index + 0.2, "metrics": {"total_return": index / 10}},
            ],
        }
        atomic_json(tmp_path / "runs" / f"run{index}" / "report.json", report)
    factors = library.list_factors()
    assert len(factors) == 1
    factor = factors[0]
    assert factor["id"] == factor_id(" rank( ret(close,20) ) ")
    assert factor["score"] == 1.2 and factor["metrics"] == {"total_return": 0.1}
    assert factor["metrics_scope"] == "development"
    assert {p["run_id"] for p in factor["provenance"]} == {"run0", "run1"}
    assert len(factor["provenance"]) == 4
    result = library.import_factors({"factors": [{"expression": expression}]})
    assert result["imported"] == 0 and result["duplicates"] == 1


def test_import_is_persistent_atomic_and_does_not_trust_external_returns(library):
    record = _import(library)
    again = FactorLibrary(library.runs_root, library.directory.parent)
    assert again.list_factors() == library.list_factors()
    before = file_hash(library.path)
    with pytest.raises(ValueError):
        library.import_factors({"factors": [{"expression": "close"}, {"expression": "delay(close, -1)"}]})
    assert file_hash(library.path) == before
    duplicate = library.import_factors({"factors": [{"expression": "rank(ret(close,20))", "score": 999, "metrics": {"sharpe": 999}}]})
    assert duplicate["duplicates"] == 1
    assert duplicate["factors"][0]["id"] == record["id"]
    assert duplicate["factors"][0]["score"] is None
    assert duplicate["factors"][0]["metrics"] == {}


def test_common_quantaalpha_export_translates_without_running_embedded_code(library, tmp_path):
    marker = tmp_path / "must-not-exist"
    payload = {"metadata": {"version": "1"}, "factors": {"foreign-id": {
        "factor_name": "Quanta momentum", "factor_expression": "RANK($close / DELAY($close, 5))",
        "factor_description": "Price ratio", "factor_implementation_code": f"open({str(marker)!r}, 'w').write('unsafe')",
        "cache_location": "/etc/passwd", "backtest_results": {"sharpe": 999},
    }}}
    factor = library.import_factors(payload)["factors"][0]
    assert factor["name"] == "Quanta momentum"
    assert factor["expression"] == "rank(close / delay(close, 5))"
    assert factor["format"] == "quantaalpha"
    assert factor["metrics"] == {} and not marker.exists()
    assert "factor_implementation_code" not in factor and "cache_location" not in factor


def test_explicit_qlib_and_factorminer_subset_conversion(library):
    result = library.import_factors({"factors": [
        {"name": "Qlib", "expression": "$close / Ref($close, 5) - 1", "format": "qlib"},
        {"name": "FactorMiner", "expression": "CsRank(Delta($close, 5))", "format": "factorminer"},
    ]})
    assert result["imported"] == 2
    assert result["factors"][0]["expression"] == "close / delay(close, 5) - 1"
    assert result["factors"][1]["expression"] == "rank(delta(close, 5))"


@pytest.mark.parametrize("dialect, expression", [
    ("qlib", "Ref($close, -1)"), ("qlib", "Ref($close, 0)"),
    ("qlib", "Rank($close, 5)"), ("qlib", "Mean($close, 5)"),
    ("quantaalpha", "TS_MEAN($close, 5)"), ("quantaalpha", "TS_STD($close, 5)"),
    ("quantaalpha", "LOG($close)"), ("quantaalpha", "DELAY($close, -1)"),
    ("alphaos", "__import__('os').system('id')"),
    ("qlib", "$close.iloc[-1]"), ("qlib", "Ref($vwap, 2)"),
])
def test_unsupported_or_unsafe_imports_are_explicitly_rejected(library, dialect, expression):
    with pytest.raises(ValueError):
        library.import_factors({"factors": [{"expression": expression, "format": dialect}]})
    assert library.list_factors() == []


def test_import_limits_and_batch_duplicates(library):
    with pytest.raises(ValueError, match="1–50"):
        library.import_factors({"factors": [{"expression": "close"}] * 51})
    with pytest.raises(ValueError):
        library.import_factors({"factors": [{"expression": "close", "format": []}]})
    result = library.import_factors({"factors": [{"expression": "close"}, {"expression": " (close) "}]})
    assert result["imported"] == 1 and result["duplicates"] == 1


def test_fixed_backtest_combo_matches_explicit_rank_average_and_is_idempotent(library, dataset, tmp_path):
    first = _import(library, "ret(close, 20)", "momentum")
    second = _import(library, "-ret(close, 5)", "reversal")
    index = dataset[1]["close"].index
    directory = _create(tmp_path, dataset, [first, second], start_date=str(index[100].date()), end_date=str(index[179].date()))
    queued = read_json(directory / "report.json")
    assert queued["status"] == "queued" and queued["metrics"] is None and queued["curve"] == []
    report = run_backtest(directory)
    assert report["status"] == "completed", report.get("error")
    prices = load_csv(directory / "snapshot.csv")["close"]
    explicit = ((prices / prices.shift(20) - 1).rank(axis=1, pct=True)
                + (-(prices / prices.shift(5) - 1)).rank(axis=1, pct=True)) / 2
    expected = backtest(explicit, prices, BacktestConfig(top_k=2), 100, 180)
    assert report["metrics"] == summarize(expected)
    assert report["period"]["sessions"] == 80
    assert report["curve"][0]["date"] == str(index[100].date())
    assert report["curve"][-1]["date"] == str(index[179].date())
    assert report["kind"] == "fixed_factor_retrospective"
    assert report["curve"][-1]["drawdown"] <= 0
    assert run_backtest(directory) == report
    assert list_backtests(tmp_path / "backtests")[0]["id"] == report["id"]
    assert all((directory / name).exists() for name in ["report.json", "curve.csv", "config.json", "factors.json", "snapshot.csv"])


def test_all_asset_portfolio_equals_costed_baseline(library, dataset, tmp_path):
    record = _import(library)
    free = run_backtest(_create(tmp_path, dataset, [record], cost_bps=0, top_k=4))
    costly = run_backtest(_create(tmp_path, dataset, [record], cost_bps=100, top_k=4))
    assert free["status"] == costly["status"] == "completed"
    assert costly["metrics"]["total_return"] < free["metrics"]["total_return"]
    for row in costly["curve"]:
        assert row["equity"] == row["benchmark_equity"]
    assert costly["metrics"]["total_return"] == costly["baseline_metrics"]["total_return"]
    assert costly["metrics"]["avg_turnover"] == costly["baseline_metrics"]["avg_turnover"]


@pytest.mark.parametrize("invalid", [
    {"dataset": "../prices.csv"}, {"dataset": "/tmp/prices.csv"}, {"dataset": "missing.csv"},
    {"cost_bps": True}, {"cost_bps": float("nan")}, {"cost_bps": -1}, {"top_k": 5},
    {"rebalance_every": 0}, {"signal_lag": 0}, {"combination": "best_factor"},
    {"start_date": "2020/06/01"}, {"end_date": "2100-01-01"},
    {"start_date": "2020-01-02"}, {"start_date": "2021-01-01", "end_date": "2020-06-01"},
    {"factor_ids": []}, {"factor_ids": ["unknown"]},
])
def test_fixed_backtest_validates_registered_data_config_dates_and_factor_ids(library, dataset, tmp_path, invalid):
    record = _import(library)
    with pytest.raises(ValueError):
        _create(tmp_path, dataset, [record], **invalid)
    assert not (tmp_path / "backtests").exists()


def test_symlink_dataset_rejected_and_explicit_end_is_inclusive(library, dataset, tmp_path):
    record = _import(library)
    (dataset[0] / "alias.csv").symlink_to(dataset[0] / "prices.csv")
    with pytest.raises(ValueError, match="registered"):
        _create(tmp_path, dataset, [record], dataset="alias.csv")
    index = dataset[1]["close"].index
    friday = next(date for date in index[60:90] if date.weekday() == 4)
    directory = _create(tmp_path, dataset, [record], end_date=str(friday.date() + timedelta(days=2)))
    report = run_backtest(directory)
    assert report["status"] == "completed"
    assert report["period"]["end"] == str(friday.date())
    assert report["curve"][-1]["date"] == str(friday.date())


def test_future_prices_and_original_dataset_changes_do_not_change_frozen_result(library, dataset, tmp_path):
    record = _import(library)
    index = dataset[1]["close"].index
    body = {"start_date": str(index[80].date()), "end_date": str(index[180].date())}
    directory = _create(tmp_path, dataset, [record], **body)
    changed = deepcopy(dataset[1])
    for field in ("open", "high", "low", "close"):
        changed[field].iloc[181:] *= 100
    save_panel(changed, dataset[0] / "changed.csv")
    altered = _create(tmp_path, dataset, [record], dataset="changed.csv", **body)
    (dataset[0] / "prices.csv").unlink()
    first, second = run_backtest(directory), run_backtest(altered)
    assert first["status"] == second["status"] == "completed"
    assert first["metrics"] == second["metrics"] and first["curve"] == second["curve"]


def test_tampered_frozen_config_and_undefined_factors_persist_failed(library, dataset, tmp_path):
    record = _import(library)
    directory = _create(tmp_path, dataset, [record])
    config = read_json(directory / "config.json")
    config["cost_bps"] = 0
    atomic_json(directory / "config.json", config)
    report = run_backtest(directory)
    assert report["status"] == "failed" and "checksum changed" in report["error"]
    assert report["metrics"] is None and report["curve"] == []
    undefined = _import(library, "close / 0", "undefined")
    second = run_backtest(_create(tmp_path, dataset, [undefined]))
    assert second["status"] == "failed" and "no valid signals" in second["error"]


def test_source_research_date_overlap_is_explicit(library, dataset, tmp_path):
    record = _import(library)
    index = dataset[1]["close"].index
    record["provenance"] = [{
        "run_id": "research-source", "trial_id": "T1", "source": {"sha256": file_hash(dataset[0] / "prices.csv")},
        "split": {"development_start": str(index[30].date()), "development_end": str(index[300].date()),
                  "holdout_start": str(index[301].date()), "holdout_end": str(index[-1].date())},
    }]
    report = run_backtest(_create(tmp_path, dataset, [record], start_date=str(index[150].date()), end_date=str(index[350].date())))
    assert report["status"] == "completed"
    assert report["overlap"][0]["development_overlap"] is True
    assert report["overlap"][0]["holdout_overlap"] is True
    assert report["overlap"][0]["same_snapshot"] is True
    assert any("重叠" in warning for warning in report["warnings"])
