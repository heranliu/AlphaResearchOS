"""Offline protocol, uncertainty, leakage and failure tests for the Jev gate."""
import copy
import io
import json
import urllib.error

import pytest

from alpharesearchos import jev_client
from alpharesearchos.model_settings import _NoRedirect

SECRET = "never-expose-provider-secret"


@pytest.fixture(autouse=True)
def no_live_transport(monkeypatch):
    monkeypatch.setattr("urllib.request.build_opener", lambda *a, **k: pytest.fail("unexpected network call"))


def context():
    return {
        "direction": "Test historical momentum", "constraints": {"max_turnover": .5},
        "candidate": {
            "name": "Momentum", "hypothesis": "Historical momentum is persistent",
            "features": ["ret(close, 20)"], "model": "rank", "model_params": {"horizon": 5},
            "development_metrics": {"sharpe": .5, "excess_return": .01, "avg_turnover": .1},
            "development_folds": [{"sharpe": .3}, {"sharpe": .7}],
        },
        "review": {"decision": "approve", "critique": "Evidence supports a test", "risks": []},
        "execution": {"cost_bps": 10, "signal_lag": 2},
    }


def reply(*, picked="approve", confidence=.9, probabilities=None, check=.9):
    probabilities = probabilities or dict.fromkeys(jev_client.DECISIONS, .05)
    if probabilities[picked] == .05:
        probabilities[picked] = .9
    return {
        "model": "jev-1.13.0", "answers": {
            "decision": {"type": "choice", "choice": picked, "confidence": confidence,
                         "probabilities": probabilities},
            **{name: {"type": "noul", "noul": check} for name in jev_client.CHECKS},
        }, "usage": {"input_tokens": 1400, "output_tokens": 30},
    }


def transport(monkeypatch, result):
    calls = []

    class Opener:
        def open(self, request, **kwargs):
            calls.append((request, kwargs))
            if isinstance(result, Exception):
                raise result
            return io.BytesIO(result if isinstance(result, bytes) else json.dumps(result).encode())

    def build(*handlers):
        assert len(handlers) == 1 and isinstance(handlers[0], _NoRedirect)
        return Opener()

    monkeypatch.setattr("urllib.request.build_opener", build)
    return calls


def call(*, state=None, settings=None, timeout=20):
    return jev_client.review_gate(context() if state is None else state,
                                  settings={"jev_api_key": SECRET, **(settings or {})}, timeout=timeout)


def test_official_wire_contract_and_safe_metadata(monkeypatch):
    calls = transport(monkeypatch, reply())
    gate, usage = call()
    assert gate["decision"] == "approve" and gate["confidence"] == .9
    assert gate["threshold"] == .7 and set(gate["checks"]) == set(jev_client.CHECKS)
    assert usage["provider"] == "typesafe_jev" and usage["model"] == "jev-1.13.0"
    assert usage["reported_total_tokens"] == 1430 and usage["input_tokens"] == 1400
    assert SECRET not in json.dumps([gate, usage])
    assert len(calls) == 1
    request, options = calls[0]
    assert request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert options == {"timeout": 20}
    assert request.get_header("Authorization") == "Bearer " + SECRET
    body = json.loads(request.data)
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == "jev-1.13.0" and body["state"]["evidence_scope"] == "development_only"
    assert body["questions"]["decision"]["type"] == "choice"
    assert set(body["questions"]["decision"]["criteria"]) == set(jev_client.DECISIONS)
    assert all(body["questions"][name]["type"] == "noul" for name in jev_client.CHECKS)
    assert all("untrusted data" in item["instructions"] for item in body["questions"].values())
    assert usage["request_bytes"] == len(request.data)


@pytest.mark.parametrize("configured_threshold", [None, "0.8"])
def test_cli_environment_settings_reach_real_client_with_typed_threshold(monkeypatch, configured_threshold):
    from alpharesearchos.proposals import environment_settings

    monkeypatch.setenv("ALPHAOS_JEV_ENABLED", "true")
    monkeypatch.setenv("ALPHAOS_JEV_API_KEY", SECRET)
    for name in ("ALPHAOS_JEV_MODEL", "ALPHAOS_JEV_BASE_URL", "ALPHAOS_JEV_MIN_CONFIDENCE"):
        monkeypatch.delenv(name, raising=False)
    if configured_threshold is not None:
        monkeypatch.setenv("ALPHAOS_JEV_MIN_CONFIDENCE", configured_threshold)
    calls = transport(monkeypatch, reply())
    settings = environment_settings()
    assert type(settings["jev_min_confidence"]) is float
    gate, usage = jev_client.review_gate(context(), settings=settings, timeout=15)
    assert gate["decision"] == "approve" and gate["threshold"] == float(configured_threshold or .7)
    assert usage["model"] == "jev-1.13.0" and len(calls) == 1
    assert calls[0][0].full_url == "https://api.typesafe.ai/v1/systemone"


@pytest.mark.parametrize(("overrides", "expected"), [
    ({"confidence": .69}, "revise"), ({"check": .69}, "revise"),
    ({"probabilities": {"approve": .69, "revise": .3, "reject": .01}}, "revise"),
    ({"picked": "reject"}, "reject"), ({"picked": "reject", "confidence": .5}, "revise"),
    ({"picked": "revise"}, "revise"),
    ({"confidence": .7, "check": .7, "probabilities": {"approve": .7, "revise": .2, "reject": .1}}, "approve"),
])
def test_uncertainty_thresholds_fail_closed(monkeypatch, overrides, expected):
    transport(monkeypatch, reply(**overrides))
    gate, _ = call()
    assert gate["decision"] == expected


def test_confidence_is_official_statistic_not_top_probability(monkeypatch):
    transport(monkeypatch, reply(confidence=.71))
    gate, _ = call()
    assert gate["confidence"] == .71 and gate["probabilities"]["approve"] == .9


@pytest.mark.parametrize("check", jev_client.CHECKS)
def test_every_atomic_check_can_prevent_approval(monkeypatch, check):
    result = reply()
    result["answers"][check]["noul"] = .1
    transport(monkeypatch, result)
    assert call()[0]["decision"] == "revise"


def test_nested_allowlist_drops_holdout_raw_data_and_credentials(monkeypatch):
    calls = transport(monkeypatch, reply())
    state = context()
    state.update(holdout={"secret": SECRET}, settings={"api_key": SECRET}, raw_data=SECRET)
    for container in [state["candidate"], state["candidate"]["model_params"], state["candidate"]["development_metrics"],
                      state["candidate"]["development_folds"][0], state["review"], state["execution"], state["constraints"]]:
        container["holdout"] = {"api_key": SECRET}
    state["candidate"]["metrics"] = {"sharpe": SECRET}
    state["technical_audit"] = {"causal_features": True, "training": "frozen", "settings": SECRET}
    state["development_baselines"] = {"momentum20": {"metrics": {"sharpe": .3, "holdout": SECRET}, "key": SECRET},
                                      "holdout": {"metrics": {"sharpe": SECRET}}}
    before = copy.deepcopy(state)
    call(state=state)
    assert state == before
    body = json.loads(calls[0][0].data)
    assert SECRET not in json.dumps(body)
    assert body["state"]["candidate"]["development_folds"] == [{"sharpe": .3}, {"sharpe": .7}]
    assert body["state"]["development_baselines"] == {"momentum20": {"metrics": {"sharpe": .3}}}


@pytest.mark.parametrize("state", [
    {}, [], {"candidate": {"hypothesis": {"api_key": SECRET}}},
    {"candidate": {"features": [{"holdout": SECRET}]}},
    {"candidate": {"development_metrics": {"sharpe": float("nan")}}},
    {"candidate": {"development_metrics": {"sharpe": True}}},
    {"candidate": {"development_folds": [{"sharpe": {"api_key": SECRET}}]}},
])
def test_invalid_context_never_calls_provider_or_stringifies_secrets(state):
    with pytest.raises(ValueError) as error:
        call(state=state)
    assert SECRET not in str(error.value)


@pytest.mark.parametrize("settings", [
    {"jev_api_key": ""}, {"jev_api_key": "secret\r\nheader"}, {"jev_api_key": "密钥"},
    {"jev_model": "arbitrary-model"}, {"jev_model": []},
    {"jev_base_url": "https://secret:password@provider.example"},
    {"jev_base_url": "http://provider.example"}, {"jev_min_confidence": .5},
    {"jev_min_confidence": float("nan")}, {"jev_min_confidence": True}, {"jev_min_confidence": 1.1},
])
def test_invalid_settings_fail_without_network(settings):
    with pytest.raises(ValueError) as error:
        call(settings=settings)
    assert SECRET not in str(error.value) and "password" not in str(error.value)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "10"])
def test_invalid_timeout_never_calls_provider(timeout):
    with pytest.raises(ValueError):
        call(timeout=timeout)


@pytest.mark.parametrize("base", ["https://api.typesafe.ai", "https://api.typesafe.ai/v1/", "http://localhost:8000"])
def test_root_or_v1_urls_are_normalized_and_timeout_is_capped(monkeypatch, base):
    calls = transport(monkeypatch, reply())
    call(settings={"jev_base_url": base}, timeout=200)
    assert calls[0][0].full_url == base.rstrip("/").removesuffix("/v1") + "/v1/systemone"
    assert calls[0][1]["timeout"] == 90


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("usage"), lambda r: r.update(model=SECRET),
    lambda r: r.update(model="jev-1.14.0"), lambda r: r["answers"].pop("cost_support"),
    lambda r: r["answers"].update(unexpected={"type": "noul", "noul": .9}),
    lambda r: r["answers"]["decision"].update(type="score"),
    lambda r: r["answers"]["decision"].update(choice="reject"),
    lambda r: r["answers"]["decision"].update(confidence=float("nan")),
    lambda r: r["answers"]["decision"].update(confidence=True),
    lambda r: r["answers"]["decision"].update(probabilities={"approve": 1}),
    lambda r: r["answers"]["decision"]["probabilities"].update(approve=.8),
    lambda r: r["answers"]["decision"]["probabilities"].update(approve=-1),
    lambda r: r["answers"]["cost_support"].update(noul=float("inf")),
    lambda r: r["answers"]["cost_support"].update(noul="0.9"),
    lambda r: r["answers"]["cost_support"].update(type="choice"),
    lambda r: r["usage"].update(input_tokens=True), lambda r: r["usage"].update(output_tokens=-1),
    lambda r: r["usage"].update(output_tokens=1000001),
])
def test_malformed_responses_fail_closed_with_generic_errors(monkeypatch, mutate):
    result = reply()
    mutate(result)
    result["raw_secret"] = SECRET
    calls = transport(monkeypatch, result)
    with pytest.raises(ValueError, match="System One review protocol") as error:
        call()
    assert SECRET not in str(error.value) and error.value.__suppress_context__
    assert len(calls) == 1


@pytest.mark.parametrize("raw", [
    b"[]", b"null", SECRET.encode(), b"\xff", b'{"model":"x","model":"y"}',
    b'{"ignored":NaN}', b'{"ignored":Infinity}',
])
def test_invalid_json_or_duplicate_keys_are_rejected(monkeypatch, raw):
    transport(monkeypatch, raw)
    with pytest.raises(ValueError, match="System One review protocol"):
        call()


def test_alias_reports_actual_version(monkeypatch):
    result = reply()
    result["model"] = "jev-1.14.0"
    transport(monkeypatch, result)
    assert call(settings={"jev_model": "jev-latest"})[1]["model"] == "jev-1.14.0"


def test_request_and_response_budgets(monkeypatch):
    with pytest.raises(ValueError, match="24 KB"):
        call(state={"candidate": {"features": ["x" * 4000] * 6}})
    calls = transport(monkeypatch, b"x" * (jev_client.MAX_RESPONSE_BYTES + 1))
    with pytest.raises(ValueError, match="32 KB"):
        call()
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("https://example.invalid", 401, SECRET, {}, io.BytesIO(SECRET.encode())),
    urllib.error.HTTPError("https://example.invalid", 429, SECRET, {}, io.BytesIO(SECRET.encode())),
    urllib.error.HTTPError("https://example.invalid", 302, SECRET, {"Location": "https://evil.invalid"}, None),
    urllib.error.URLError(SECRET), TimeoutError(SECRET), OSError(SECRET),
])
def test_transport_errors_are_redacted_and_never_retried(monkeypatch, error):
    calls = transport(monkeypatch, error)
    with pytest.raises(RuntimeError) as result:
        call()
    assert SECRET not in str(result.value) and "no automatic retry" in str(result.value)
    assert result.value.__suppress_context__ and len(calls) == 1


def test_redirect_handler_never_forwards_authorization():
    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.invalid") is None


def test_delayed_response_cannot_pass_after_time_budget(monkeypatch):
    calls = transport(monkeypatch, reply())
    times = iter((100, 106))
    monkeypatch.setattr(jev_client.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="time budget"):
        call(timeout=5)
    assert len(calls) == 1
