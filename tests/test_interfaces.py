import io
import json
import threading
import urllib.error
import urllib.request

import pytest

from alpharesearchos.proposals import llm_proposal
from alpharesearchos.server import make_server


@pytest.fixture(autouse=True)
def isolate_generation_options(monkeypatch):
    monkeypatch.delenv("ALPHAOS_LLM_TOKEN_FIELD", raising=False)
    monkeypatch.delenv("ALPHAOS_LLM_TEMPERATURE", raising=False)


def test_llm_json_protocol_and_no_secret_in_failure(monkeypatch):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "mock-model")
    monkeypatch.setenv("ALPHAOS_LLM_API_KEY", "test-key-do-not-log")
    body = {"choices": [{"message": {"content": '```json\n{"name":"test","hypothesis":"momentum","expression":"rank(ret(close, 20))"}\n```'}}],
            "usage": {"total_tokens": 100}}
    captured = []

    def response(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr(urllib.request, "urlopen", response)
    candidate, usage = llm_proposal("研究趋势", [])
    assert candidate["expression"] == "rank(ret(close, 20))"
    assert usage["reported_total_tokens"] == 100
    assert "holdout" not in captured[0]["messages"][1]["content"]
    assert captured[0]["max_completion_tokens"] == 700
    assert "max_tokens" not in captured[0]
    assert "temperature" not in captured[0]

    def failed(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, "test-key-do-not-log", {}, io.BytesIO(b"secret"))
    monkeypatch.setattr(urllib.request, "urlopen", failed)
    with pytest.raises(RuntimeError) as caught:
        llm_proposal("trend", [])
    assert "401" in str(caught.value)
    assert "test-key" not in str(caught.value)


def test_legacy_provider_parameters_are_explicit(monkeypatch):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "arbitrary-model-without-name-heuristics")
    monkeypatch.setenv("ALPHAOS_LLM_API_KEY", "test")
    monkeypatch.setenv("ALPHAOS_LLM_TOKEN_FIELD", "max_tokens")
    monkeypatch.setenv("ALPHAOS_LLM_TEMPERATURE", "0.4")
    captured = []

    def response(request, **kwargs):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"choices": [{"message": {"content":
            '{"name":"test","hypothesis":"trend","expression":"rank(ret(close,20))"}'}}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", response)
    _, usage = llm_proposal("trend", [])
    assert captured[0]["max_tokens"] == 700
    assert captured[0]["temperature"] == 0.4
    assert "max_completion_tokens" not in captured[0]
    assert usage["generation_options"] == {"max_tokens": 700, "temperature": 0.4}


@pytest.mark.parametrize("key,value", [
    ("ALPHAOS_LLM_TOKEN_FIELD", "max_output_tokens"),
    ("ALPHAOS_LLM_TEMPERATURE", "nan"), ("ALPHAOS_LLM_TEMPERATURE", "inf"),
    ("ALPHAOS_LLM_TEMPERATURE", "-0.1"), ("ALPHAOS_LLM_TEMPERATURE", "2.1"),
    ("ALPHAOS_LLM_TEMPERATURE", "invalid"),
])
def test_invalid_generation_options_fail_before_network(monkeypatch, key, value):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "mock")
    monkeypatch.setenv("ALPHAOS_LLM_API_KEY", "test")
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not call provider"))
    with pytest.raises(ValueError, match=key):
        llm_proposal("trend", [])


@pytest.mark.parametrize("content,finish_reason", [(None, "stop"), ("", "stop"), ("{}", "length")])
def test_empty_or_token_limited_completion_has_clear_failure(monkeypatch, content, finish_reason):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "mock")
    monkeypatch.setenv("ALPHAOS_LLM_API_KEY", "test")
    response = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError, match="700-token|no text JSON"):
        llm_proposal("trend", [])


@pytest.mark.parametrize("content", ["not JSON", "[]", '{"name":"x","hypothesis":"y","expression":4}'])
def test_malformed_llm_response_is_rejected(monkeypatch, content):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "mock")
    monkeypatch.setenv("ALPHAOS_LLM_API_KEY", "test")
    response = {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(response).encode()))
    with pytest.raises(ValueError):
        llm_proposal("trend", [])


@pytest.fixture
def dashboard_server(tmp_path):
    server = make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    server.research_executor.shutdown(wait=True)
    thread.join(timeout=5)


def test_dashboard_reads_real_empty_state_and_assets(dashboard_server):
    for route in ["/", "/static/app.js", "/static/style.css", "/api/health", "/api/runs", "/api/datasets"]:
        with urllib.request.urlopen(dashboard_server + route, timeout=5) as response:
            assert response.status == 200
    with urllib.request.urlopen(dashboard_server + "/api/runs") as response:
        assert json.load(response) == {"runs": []}


def test_dashboard_denies_cross_origin_and_path_access(dashboard_server):
    for headers in [{"Origin": "https://example.com"}, {"Host": "evil.example"}]:
        request = urllib.request.Request(dashboard_server + "/api/runs", headers=headers)
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 403
    request = urllib.request.Request(dashboard_server + "/api/runs", data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 415
    request = urllib.request.Request(dashboard_server + "/api/runs", data=b'{"dataset":"../secret.csv"}', headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request)
    assert caught.value.code == 400
