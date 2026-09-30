"""Upload diagnostics are bounded, descriptive, and do not publish datasets."""

import json

import pytest
from test_dataset_import import market_csv
from test_dataset_import import web as web

from alpharesearchos import server as workspace


def inspect(request, content, name="preview.csv"):
    return request({"name": name, "content": content}, path="/api/datasets/inspect")


def test_valid_inspection_has_stable_coverage_without_disk_side_effects(web):
    request, directory = web
    before = sorted(directory.parent.rglob("*"))
    status, response = inspect(request, market_csv(), "行业 ETF.CSV")
    assert status == 200
    result = response["inspection"]
    assert result["name"] == "行业 ETF.csv"
    assert (result["rows"], result["sessions"], result["assets"]) == (1500, 500, 3)
    assert result["start"] == "2020-01-01" and result["end"] == "2021-05-14"
    assert result["common_range"] == {"start": "2020-01-01", "end": "2021-05-14", "sessions": 500}
    assert result["importable"] and result["eligible_for_research"] and result["eligible_for_backtest"]
    assert result["missing_values"] == result["duplicate_rows"] == result["missing_records"] == 0
    assert result["asset_coverage"][0] == {
        "symbol": "ASSET0", "rows": 500, "sessions": 500, "start": "2020-01-01",
        "end": "2021-05-14", "missing_sessions": 0,
    }
    assert result["adjustment"]["status"] == "unknown"
    assert "不会自动复权或补齐" in result["adjustment"]["message"]
    assert {item["code"] for item in result["issues"]} == {"calendar_scope"}
    assert all(item["severity"] == "warning" for item in result["issues"])
    assert sorted(directory.parent.rglob("*")) == before
    assert request() == (200, {"datasets": []})
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("sessions", "importable", "research"), [
    (399, False, False), (400, True, False), (467, True, False), (468, True, True),
])
def test_research_eligibility_matches_default_split_and_backtest_minimum(web, sessions, importable, research):
    request, directory = web
    status, response = inspect(request, market_csv(sessions=sessions))
    assert status == 200
    result = response["inspection"]
    assert result["importable"] is importable
    assert result["eligible_for_backtest"] is importable
    assert result["eligible_for_research"] is research
    if importable and not research:
        assert any(issue["code"] == "research_history" and issue["severity"] == "warning"
                   for issue in result["issues"])
    assert list(directory.iterdir()) == []


def test_inspection_reports_missing_values_duplicates_and_date_coverage_together(web):
    request, directory = web

    def mutate(rows):
        rows[0][5] = ""
        rows.append(rows[1].copy())
        rows.pop(2)  # ASSET2 has no first date; do not invent one.

    content = market_csv(mutate=mutate)
    status, response = inspect(request, content)
    assert status == 200
    result = response["inspection"]
    assert not result["importable"] and not result["eligible_for_research"]
    assert result["missing_values"] == result["duplicate_rows"] == result["missing_records"] == 1
    assert result["common_range"] == {"start": "2020-01-02", "end": "2021-05-14", "sessions": 499}
    coverage = {item["symbol"]: item for item in result["asset_coverage"]}
    assert coverage["ASSET1"]["rows"] == 501 and coverage["ASSET1"]["sessions"] == 500
    assert coverage["ASSET2"]["missing_sessions"] == 1
    assert {"missing_values", "duplicate_rows", "missing_records"} <= {item["code"] for item in result["issues"]}
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(("content", "issue"), [
    ("date,symbol,close\n2020-01-01,A,1\n", "missing_columns"),
    ("date,symbol,open,high,low,close,volume,close\n", "duplicate_columns"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(0, "2020-02-31")), "invalid_dates"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(0, "2020-01-01T12:00:00")), "invalid_dates"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(6, "NaN")), "invalid_numbers"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(6, "inf")), "invalid_numbers"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(5, 0)), "invalid_prices"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(3, 1)), "invalid_ranges"),
    (market_csv(mutate=lambda rows: [row.__setitem__(1, "NA") for row in rows if row[1] == "ASSET0"]), "invalid_panel"),
])
def test_data_errors_return_diagnostics_instead_of_successful_eligibility(web, content, issue):
    request, directory = web
    status, response = inspect(request, content)
    assert status == 200
    result = response["inspection"]
    assert not result["importable"] and not result["eligible_for_backtest"]
    assert issue in {item["code"] for item in result["issues"]}
    assert list(directory.iterdir()) == []


def test_inspection_uses_upload_security_and_size_limits(web, monkeypatch):
    request, directory = web
    assert inspect(request, market_csv(), "../escape.csv")[0] == 400
    assert inspect(request, "")[0] == 400
    assert inspect(request, 'date,symbol,open,high,low,close,volume\n"unterminated')[0] == 400
    assert inspect(request, market_csv(sessions=1, assets=101))[0] == 400
    assert request({"name": "preview.csv", "content": "x"}, path="/api/datasets/inspect",
                   headers={"Origin": "https://other.example"})[0] == 403
    assert request(raw=b'{"name":"bad.csv","content":"\xff"}', path="/api/datasets/inspect")[0] == 400
    monkeypatch.setattr(workspace, "DATASET_MAX_BYTES", 100)
    assert inspect(request, "x" * 101)[0] == 413
    assert request(raw=b"x" * 5000, path="/api/datasets/inspect")[0] == 413
    assert list(directory.iterdir()) == []


def test_import_alias_preserves_bytes_and_does_not_overwrite_after_inspection(web):
    request, directory = web
    content = "\ufeff" + market_csv()
    assert inspect(request, content)[1]["inspection"]["importable"]
    status, response = request({"name": "preview.csv", "content": content}, path="/api/datasets/import")
    assert status == 201 and response["dataset"]["rows"] == 500
    assert (directory / "preview.csv").read_bytes() == content.encode()
    assert inspect(request, content)[0] == 200
    assert request({"name": "preview.csv", "content": content})[0] == 409
    assert sorted(path.name for path in directory.iterdir()) == ["preview.csv"]
