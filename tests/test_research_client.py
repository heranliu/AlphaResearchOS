"""Exercise provider contracts with recorded requests and zero live inference."""
from __future__ import annotations

import copy
import http.client
import io
import json
import urllib.error

import pytest

from alpharesearchos import codex_provider, research_client
from alpharesearchos.model_settings import SettingsStore
from alpharesearchos.proposals import provider_context
from alpharesearchos.provider_transport import ModelRequestError

PROPOSAL = {"name": "动量", "hypothesis": "检验价格趋势", "features": ["rank(ret(close,20))"],
            "model": "rank", "model_params": {"alpha": 1, "train_window": 504, "retrain_every": 63,
                                              "horizon": 5, "smoothing": 3}, "rationale": "连接契约检查"}
REVIEW = {"decision": "approve", "critique": "技术契约有效", "risks": ["需要更多证据"],
          "suggested_action": "进行下一项预先指定的检验"}
SECRET = "private-provider-canary"


def settings(**changes):
    return {"provider": "openai_compatible", "base_url": "https://provider.example/v1", "model": "chosen-model",
            "api_key": SECRET, "token_field": "max_completion_tokens", "temperature": None, **changes}


def response(value=PROPOSAL, **changes):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}], **changes}


@pytest.fixture(autouse=True)
def no_external_requests(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Provider tests must never make an unmocked network or CLI call")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr("alpharesearchos.provider_transport.bounded_read", forbidden)
    monkeypatch.setattr(codex_provider.subprocess, "run", forbidden)
    monkeypatch.setattr(codex_provider.subprocess, "Popen", forbidden)
    monkeypatch.setattr(codex_provider, "codex_status", lambda **kwargs: {
        "available": False, "authenticated": False, "version": None, "default_model": None,
    })
    with provider_context(settings()):
        yield


def transport(monkeypatch, body=None, *, error=None):
    calls = []

    def read(request, *, timeout, max_bytes):
        calls.append((request, {"timeout": timeout}))
        assert max_bytes == 100000
        if error:
            raise error
        raw = body if isinstance(body, bytes) else json.dumps(response() if body is None else body).encode()
        return raw[:max_bytes + 1]

    monkeypatch.setattr("alpharesearchos.provider_transport.bounded_read", read)
    return calls


@pytest.mark.parametrize("base_url,token_field", [
    ("https://api.deepseek.com", "max_tokens"),
    ("https://generativelanguage.googleapis.com/v1beta/openai/", "max_completion_tokens"),
    ("http://localhost:11434/v1", "max_tokens"),
])
def test_compatible_endpoints_use_explicit_capabilities_and_full_research_contract(monkeypatch, base_url, token_field):
    calls = transport(monkeypatch, response(usage={"prompt_tokens": 40, "completion_tokens": 12, "total_tokens": 52}))
    with provider_context(settings(base_url=base_url, token_field=token_field, temperature=0)):
        result, usage = research_client.propose({"direction": "趋势研究"}, timeout=7)
    assert result == PROPOSAL and len(calls) == 1
    request, options = calls[0]
    payload = json.loads(request.data)
    assert request.full_url == base_url.rstrip("/") + "/chat/completions"
    assert request.get_header("Authorization") == "Bearer " + SECRET
    assert options == {"timeout": 7}
    assert payload[token_field] == 1800 and payload["temperature"] == 0
    assert payload["model"] == "chosen-model" and "response_format" not in payload
    assert json.loads(payload["messages"][0]["content"].split(": ", 1)[1]) == research_client.PROPOSAL_SCHEMA
    assert usage["reported_total_tokens"] == 52
    assert usage["request_bytes"] == len(request.data)
    assert SECRET not in json.dumps(usage)


@pytest.mark.parametrize("raw,expected", [
    (None, None), ({}, None), ([], None),
    ({"total_tokens": True}, None), ({"total_tokens": -2}, None), ({"total_tokens": "70"}, None),
    ({"prompt_tokens": 9}, None), ({"total_tokens": 17}, 17),
    ({"prompt_tokens": 10, "completion_tokens": 5}, 15),
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 2}, 15),
    ({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 20}, 20),
    ({"prompt_tokens": 10, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 8},
      "completion_tokens_details": {"reasoning_tokens": 4}, "secret": SECRET}, 15),
])
def test_usage_is_numeric_or_unknown_and_subsets_are_not_double_counted(monkeypatch, raw, expected):
    transport(monkeypatch, response(usage=raw))
    _, usage = research_client.propose({}, timeout=1)
    assert usage["reported_total_tokens"] == expected
    assert SECRET not in json.dumps(usage)


@pytest.mark.parametrize("body", [
    b"private-provider-canary", b"\xff", b"[]", b'{"choices":[],"choices":[]}',
    b'{"choices": [], "usage": {"total_tokens": NaN}}',
    {}, {"choices": None}, {"choices": []}, {"choices": [None]},
    {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": None}}]},
    {"choices": [{"message": {"content": []}}]},
    {"choices": [{"message": {"content": "   "}}]},
    {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]},
    {"choices": [{"message": {"content": "{}"}, "finish_reason": "content_filter"}]},
    {"choices": [{"message": {"content": "{}"}, "finish_reason": []}]},
    {"choices": [{"message": {"content": "{}", "refusal": SECRET}}]},
    {"choices": [{"message": {"content": "{}", "tool_calls": [{"id": SECRET}]}}]},
    {"choices": [{"message": {"content": '{"x": 1, "x": 2}'}}]},
    {"choices": [{"message": {"content": '{"x": Infinity}'}}]},
    {"choices": [{"message": {"content": '{"x": 1e9999}'}}]},
    b"x" * 100001,
])
def test_bad_envelopes_truncation_and_unsafe_json_fail_once_without_body_leaks(monkeypatch, body):
    calls = transport(monkeypatch, body)
    with pytest.raises(ValueError) as error:
        research_client.propose({}, timeout=2)
    assert SECRET not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("code", [301, 302, 307, 308, 401, 429, 500])
def test_http_failures_are_not_retried_and_never_expose_body_or_headers(monkeypatch, code):
    fp = io.BytesIO(SECRET.encode())
    error = urllib.error.HTTPError("https://provider.example/" + SECRET, code, SECRET,
                                   {"Location": "https://another.example/" + SECRET, "Retry-After": SECRET}, fp)
    calls = transport(monkeypatch, error=error)
    with pytest.raises(ModelRequestError) as caught:
        research_client.propose({}, timeout=2)
    assert caught.value.http_status == code and str(code) in str(caught.value)
    assert SECRET not in str(caught.value)
    assert len(calls) == 1 and fp.closed
    if code == 429:
        assert "rate limited" in str(caught.value)
    if code < 400:
        assert "redirect refused" in str(caught.value)


@pytest.mark.parametrize("error", [urllib.error.URLError(SECRET), TimeoutError(SECRET), OSError(SECRET),
                                   http.client.IncompleteRead(SECRET.encode(), 200)])
def test_connection_and_incomplete_read_failures_remain_private(monkeypatch, error):
    calls = transport(monkeypatch, error=error)
    with pytest.raises(ModelRequestError) as caught:
        research_client.review({}, timeout=2)
    assert SECRET not in str(caught.value) and len(calls) == 1


def test_context_trimming_preserves_candidate_parents_and_does_not_mutate_caller(monkeypatch):
    calls = transport(monkeypatch, response(REVIEW))
    context = {"candidate": PROPOSAL, "parent_trajectories": [{"id": "chosen-parent"}],
               "development_history": [{"id": i, "text": "历" * 3000} for i in range(3)],
               "retrieved_trajectories": [{"id": i, "text": "检" * 2000} for i in range(4)]}
    original = copy.deepcopy(context)
    research_client.review(context, timeout=3)
    request = calls[0][0]
    supplied = json.loads(json.loads(request.data)["messages"][1]["content"].rsplit("\n", 1)[1])
    assert supplied["candidate"] == PROPOSAL
    assert supplied["parent_trajectories"] == [{"id": "chosen-parent"}]
    assert supplied["development_history"] == [] and supplied["context_trimmed"]
    remaining_ids = [item["id"] for item in supplied["retrieved_trajectories"]]
    assert remaining_ids and remaining_ids == list(range(remaining_ids[0], 4))
    assert len(request.data) <= 22200 and context == original and len(calls) == 1


def test_untrimmable_context_fails_before_a_request(monkeypatch):
    calls = transport(monkeypatch)
    with pytest.raises(ValueError, match="22 KB"):
        research_client.propose({"candidate": "中" * 8000}, timeout=3)
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_time_budget_never_starts_request(monkeypatch, timeout):
    calls = transport(monkeypatch)
    with pytest.raises(ValueError, match="timeout"):
        research_client.propose({}, timeout=timeout)
    assert calls == []


def test_review_contract_and_fenced_json_are_supported(monkeypatch):
    body = response(REVIEW)
    body["choices"][0]["message"]["content"] = "```json\n" + json.dumps(REVIEW) + "\n```"
    calls = transport(monkeypatch, body)
    result, _ = research_client.review({"candidate": PROPOSAL}, timeout=1000)
    assert result == REVIEW and calls[0][1]["timeout"] == 90


@pytest.mark.parametrize("patch", [{"decision": []}, {"decision": "maybe"}, {"critique": ""},
                                  {"risks": [7]}, {"risks": "none"}, {"unexpected": True}])
def test_review_schema_rejects_invalid_decisions_and_evidence(monkeypatch, patch):
    calls = transport(monkeypatch, response({**REVIEW, **patch}))
    with pytest.raises(ValueError):
        research_client.review({}, timeout=2)
    assert len(calls) == 1


def test_codex_route_receives_the_same_research_schema_and_no_http_call(monkeypatch):
    calls = []

    def codex(prompt, schema, *, model, timeout):
        calls.append((prompt, schema, model, timeout))
        return copy.deepcopy(PROPOSAL), {"provider": "codex_cli", "model": model}

    monkeypatch.setattr(codex_provider, "codex_structured", codex)
    with provider_context(settings(provider="codex_cli", codex_model="selected-cli-model")):
        value, _ = research_client.connection_probe(timeout=9)
    assert value == PROPOSAL and len(calls) == 1
    assert calls[0][1] == research_client.PROPOSAL_SCHEMA
    assert calls[0][2:] == ("selected-cli-model", 9)


@pytest.mark.parametrize("patch", [
    {"features": ["close.__class__"]}, {"features": []}, {"features": ["ret(close,-1)"]},
    {"features": ["close", "close"]}, {"model": "unknown"}, {"model_params": {}},
    {"model_params": {**PROPOSAL["model_params"], "horizon": True}}, {"unexpected": "field"},
])
def test_connection_probe_rejects_non_executable_proposals(monkeypatch, tmp_path, patch):
    store = SettingsStore(tmp_path)
    store.update(settings())
    calls = transport(monkeypatch, response({**PROPOSAL, **patch}))
    result = store.test_connection()
    assert not result["ok"] and len(calls) == 1
    assert SECRET not in json.dumps(result)


def test_explicit_probe_returns_no_model_content_and_preserves_scoped_provider(monkeypatch, tmp_path):
    store = SettingsStore(tmp_path)
    store.update(settings(model="saved-model"))
    calls = transport(monkeypatch, response({**PROPOSAL, "hypothesis": SECRET}, model=SECRET))
    result = store.test_connection()
    assert result["ok"] and result["model"] == "saved-model" and SECRET not in json.dumps(result)
    assert calls[0][1]["timeout"] == 15
    research_client.propose({}, timeout=1)
    assert json.loads(calls[1][0].data)["model"] == "chosen-model"


def test_development_summary_excludes_holdout_private_data_and_unbounded_reviews():
    source = {**PROPOSAL, "holdout": {"sharpe": SECRET}, "api_key": SECRET, "curve": [SECRET],
              "metrics": {"sharpe": 1, "private": SECRET}, "review": {**REVIEW, "secret": SECRET}}
    compact = research_client.compact_trial(source)
    assert SECRET not in json.dumps(compact)
    assert compact["development_metrics"]["sharpe"] == 1
