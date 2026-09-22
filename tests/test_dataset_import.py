"""CSV uploads become selectable datasets only after strict, bounded validation."""

import csv
import http.client
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pytest

from alpharesearchos import server as workspace


def market_csv(sessions=500, assets=3, mutate=None):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["date", "symbol", "open", "high", "low", "close", "volume"])
    rows = [[str(date(2020, 1, 1) + timedelta(days=day)), f"ASSET{asset}", 100, 102, 99, 101, 1000]
            for day in range(sessions) for asset in range(assets)]
    if mutate:
        mutate(rows)
    writer.writerows(rows)
    return output.getvalue()


@pytest.fixture
def web(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Importing data must not call models or start research")
    monkeypatch.setattr(workspace, "run_research", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    service = workspace.make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()

    def request(body=None, *, path="/api/datasets", raw=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", service.server_port, timeout=10)
        try:
            content = raw if raw is not None else None if body is None else json.dumps(body).encode()
            connection.request("GET" if content is None else "POST", path, body=content,
                               headers={"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    yield request, tmp_path / "datasets"
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    thread.join(timeout=5)


def test_valid_upload_is_selected_metadata_and_preserves_file_bytes(web):
    request, directory = web
    assert request() == (200, {"datasets": []})
    content = market_csv()
    code, response = request({"name": "行业 ETF.csv", "content": content})
    assert code == 201
    dataset = response["dataset"]
    assert dataset == {"id": "行业 ETF.csv", "label": "行业 ETF", "source": "csv", "rows": 500,
                       "assets": 3, "start": "2020-01-01", "end": "2021-05-14"}
    assert (directory / dataset["id"]).read_bytes() == content.encode()
    assert request() == (200, {"datasets": [dataset]})
    assert [path.name for path in directory.iterdir()] == [dataset["id"]]


def test_minimum_400_session_csv_is_available_for_fixed_backtest(web):
    request, _ = web
    code, response = request({"name": "short.csv", "content": market_csv(sessions=400)})
    assert code == 201 and response["dataset"]["rows"] == 400


@pytest.mark.parametrize("name", ["../escape.csv", "/tmp/escape.csv", "sub/file.csv", "sub\\file.csv",
                                  ".hidden.csv", "file.csv\x00", "bad.txt", "../.env", "x" * 181 + ".csv"])
def test_unsafe_filename_is_rejected_without_files(web, name):
    request, directory = web
    assert request({"name": name, "content": market_csv()})[0] == 400
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize(("content", "message"), [
    ("", "为空"),
    ("date,symbol,close\n2020-01-01,A,1\n", "缺少必需列"),
    ('date,symbol,open,high,low,close,volume\n"unterminated', "CSV 格式"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(5, "wrong")), "第 2 行 close"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(6, "NaN")), "第 2 行 volume"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(0, "2020-02-31")), "有效日期"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(0, "2020-01-01T12:00:00")), "YYYY-MM-DD"),
    (market_csv(mutate=lambda rows: rows.append(rows[0])), "只能有一行"),
    (market_csv(mutate=lambda rows: rows.pop()), "相同交易日期"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(3, 1)), "high 必须"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(4, 1000)), "low 必须"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(5, 0)), "必须大于 0"),
    (market_csv(mutate=lambda rows: rows[0].__setitem__(6, -1)), "volume 必须"),
    (market_csv(sessions=399), "400–20,000"),
    (market_csv(assets=2), "3–100"),
    (market_csv(sessions=1, assets=101), "最多支持 100"),
], ids=["empty", "missing-columns", "broken-quotes", "nonnumeric", "nan", "invalid-date", "timestamp",
        "duplicate", "incomplete-panel", "invalid-high", "invalid-low", "zero-price", "negative-volume",
        "too-short", "too-few-assets", "too-many-assets"])
def test_invalid_csv_has_actionable_error_and_no_partial_file(web, content, message):
    request, directory = web
    code, response = request({"name": "invalid.csv", "content": content})
    assert code == 400 and message in response["error"]
    assert list(directory.iterdir()) == []


def test_same_name_and_symlink_are_never_overwritten(web):
    request, directory = web
    content = market_csv()
    assert request({"name": "market.csv", "content": content})[0] == 201
    assert request({"name": "market.csv", "content": market_csv(sessions=501)})[0] == 409
    assert (directory / "market.csv").read_text() == content
    outside = directory.parent / "outside.csv"
    outside.write_text("untouched")
    (directory / "linked.csv").symlink_to(outside)
    assert request({"name": "linked.csv", "content": content})[0] == 409
    assert outside.read_text() == "untouched"
    assert not list(directory.glob(".upload-*"))


def test_concurrent_same_name_uploads_commit_exactly_one_file(web):
    request, directory = web
    content = market_csv()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: request({"name": "same.csv", "content": content}), range(2)))
    assert sorted(code for code, _ in results) == [201, 409]
    assert (directory / "same.csv").read_text() == content
    assert len(list(directory.iterdir())) == 1


def test_upload_size_origin_and_utf8_bounds(web, monkeypatch):
    request, directory = web
    monkeypatch.setattr(workspace, "DATASET_MAX_BYTES", 100)
    assert request({"name": "big.csv", "content": "x" * 101})[0] == 413
    assert request(raw=b"x" * 5000)[0] == 413
    assert request(raw=b'{"name":"bad.csv","content":"\xff"}')[0] == 400
    assert request({"name": "bad.csv", "content": "x"}, headers={"Origin": "https://other.example"})[0] == 403
    assert list(directory.iterdir()) == []


@pytest.mark.parametrize("path", ["/api/runs", "/api/backtests"])
def test_new_jobs_require_registered_csv_instead_of_synthetic_fallback(web, path):
    request, _ = web
    for body in [{}, {"dataset": "demo"}, {"dataset": "../private.csv"}]:
        code, response = request(body, path=path)
        assert code == 400 and "CSV" in response["error"]
