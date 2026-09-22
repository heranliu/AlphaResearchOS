"""The web default is model-only research; library features have no portfolio returns."""

import http.client
import json
import threading

import pytest

from alpharesearchos import codex_provider
from alpharesearchos import server as workspace
from alpharesearchos.data import make_demo, save_panel
from alpharesearchos.factors import validate_expression
from alpharesearchos.library import FactorLibrary
from alpharesearchos.store import atomic_json, read_json


@pytest.fixture
def isolated_web(tmp_path, monkeypatch):
    save_panel(make_demo(sessions=500, assets=3), tmp_path / "datasets" / "fixture.csv")
    for name in ["ALPHAOS_LLM_PROVIDER", "ALPHAOS_CODEX_MODEL", "ALPHAOS_LLM_API_KEY", "OPENAI_API_KEY",
                 "ALPHAOS_LLM_MODEL", "ALPHAOS_LLM_BASE_URL", "ALPHAOS_LLM_TEMPERATURE", "ALPHAOS_LLM_TOKEN_FIELD"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(codex_provider, "codex_status", lambda **kwargs: {
        "available": False, "authenticated": False, "version": None, "login_method": None, "default_model": None,
    })

    def forbidden(*args, **kwargs):
        pytest.fail("Contract tests must not call a real model or subprocess")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(codex_provider.subprocess, "Popen", forbidden)
    monkeypatch.setattr(codex_provider.subprocess, "run", forbidden)
    launched, entered = [], threading.Event()

    def record_worker(directory, **kwargs):
        launched.append(read_json(directory / "config.json") if (directory / "config.json").exists() else read_json(directory / "report.json")["config"])
        entered.set()
        return {}

    monkeypatch.setattr(workspace, "run_research", record_worker)
    service = workspace.make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", service.server_port, timeout=10)
        try:
            connection.request("GET" if body is None else "POST", path,
                               body=None if body is None else json.dumps(body),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    yield request, tmp_path, launched, entered
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    thread.join(timeout=5)


def test_new_run_without_mode_requires_model_and_never_falls_back_to_local(isolated_web):
    request, root, launched, entered = isolated_web
    status, response = request("/api/runs", {"direction": "测试自主研究入口", "dataset": "fixture.csv"})
    assert status == 400
    assert "configured" in response["error"] or "模型" in response["error"]
    assert not launched and not entered.is_set()
    assert request("/api/runs") == (200, {"runs": []})
    assert not list((root / "runs").glob("*/report.json"))


def test_new_run_without_mode_uses_agent_defaults_with_configured_model(isolated_web):
    request, root, launched, entered = isolated_web
    status, settings = request("/api/settings", {"provider": "openai_compatible", "base_url": "https://provider.example/v1", "model": "contract-model", "api_key": "not-a-real-key"})
    assert status == 200 and settings["configured"]
    status, response = request("/api/runs", {"direction": "测试自主研究入口", "dataset": "fixture.csv"})
    assert status == 201
    assert entered.wait(timeout=3)
    config = read_json(root / "runs" / response["id"] / "report.json")["config"]
    assert config["mode"] == launched[0]["mode"] == "agent"
    assert config["trials"] == 6 and config["max_llm_calls"] == 12
    assert config["max_llm_tokens"] == 786432 and config["max_seconds"] == 900
    assert config["warmup"] == 252 and config["max_complexity"] == 150


@pytest.mark.parametrize("model", ["rank", "ridge", "hist_gbdt"])
def test_agent_multifeature_library_expansion_does_not_assign_portfolio_results(tmp_path, model):
    runs = tmp_path / "runs"
    expressions = ["rank(ret(close,20))", "-rank(std(ret(close,1),20))"]
    metrics = {"sharpe": 7.25, "total_return": 3.5, "max_drawdown": -0.01}
    report = {
        "id": "agent-research", "created_at": "2026-09-18T13:00:00+00:00", "config": {"mode": "agent"},
        "source": {"kind": "csv", "label": "fixture.csv"},
        "split": {"development_start": "2020-01-01", "development_end": "2024-12-31"},
        "trials": [{"id": "T0001", "name": "双特征模型", "hypothesis": "测试模型组合", "status": "ok",
                    "model": model, "features": expressions, "expression": expressions[0],
                    "model_params": {}, "score": 9.75, "metrics": metrics,
                    "review": {"decision": "approve"}, "causality": {"passed": True}}],
    }
    atomic_json(runs / "agent-research" / "report.json", report)
    factors = FactorLibrary(runs, tmp_path / "state").list_factors()
    assert len(factors) == 2
    assert {factor["expression"] for factor in factors} == {validate_expression(expression)["canonical"] for expression in expressions}
    assert {factor["trial_id"] for factor in factors} == {"T0001:F1", "T0001:F2"}
    for factor in factors:
        assert factor["origin"] == "agent_feature"
        assert factor["score"] is None and factor["metrics"] == {} and factor["metrics_scope"] is None
        assert factor["run_id"] == "agent-research"
        assert factor["provenance"][0]["source"] == report["source"]
        assert factor["provenance"][0]["trial_id"] == factor["trial_id"]
        assert "model" not in factor and "model_params" not in factor
    assert read_json(runs / "agent-research" / "report.json")["trials"][0]["metrics"] == metrics
