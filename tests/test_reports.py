"""Portable reports must preserve evidence and safely render model-written text."""

from alpharesearchos.reports import write_report


def test_report_escapes_untrusted_text_and_is_offline(tmp_path):
    report = {
        "id": "sample-run",
        "status": "completed",
        "config": {"direction": '<script>alert("x")</script>', "dataset": "market.csv"},
        "source": {"kind": "csv", "label": "market.csv", "sha256": "internal-data-marker"},
        "provenance": {"source_code_sha256": "internal-code-marker"},
        "selected": {"name": "candidate", "expression": "rank(ret(close,20))", "id": "t1"},
        "trials": [{"id": "t1", "name": "candidate", "status": "ok", "expression": "rank(ret(close,20))", "score": 1.25}],
        "holdout": {
            "metrics": {"total_return": 0.1234, "sharpe": 1.75, "max_drawdown": -.04},
            "curve": [{"date": "2025-01-01", "equity": 1, "benchmark_equity": 1}, {"date": "2025-02-01", "equity": 1.1234, "benchmark_equity": 1.04}],
        },
    }
    write_report(tmp_path, report)
    html = (tmp_path / "report.html").read_text()
    markdown = (tmp_path / "report.md").read_text()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "12.34%" in html and "12.34%" in markdown
    assert "<svg " in html and "<polyline " in html
    assert 'src="http' not in html and 'href="http' not in html
    assert "合成" not in html and "合成" not in markdown
    assert "market.csv" in html
    for marker in ("SHA-256", "sha256", "internal-data-marker", "internal-code-marker"):
        assert marker not in html and marker not in markdown
    assert "rank(ret(close,20))" in html
    assert "开发期评分" in html and "留出期" in html


def test_partial_report_does_not_fabricate_metrics(tmp_path):
    write_report(tmp_path, {"id": "empty", "status": "failed", "holdout": None, "selected": None})
    html = (tmp_path / "report.html").read_text()
    assert "尚无留出期净值数据" in html
    assert "0.00%" not in html
    assert "最终因子代码" not in html
    assert "执行失败" in html


def test_nonfinite_values_and_malformed_optional_rows_are_tolerated(tmp_path):
    write_report(tmp_path, {
        "source": None,
        "trials": [None],
        "events": [None],
        "holdout": {"metrics": {"total_return": float("nan")}, "curve": [None, {"equity": float("inf"), "benchmark_equity": 1}], "stress": [None]},
    })
    html = (tmp_path / "report.html").read_text()
    assert "nan%" not in html and "inf%" not in html
    assert "尚无留出期净值数据" in html
