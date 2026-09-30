import asyncio
import io
import json
import os
import stat
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from alpharesearchos.model_settings import SettingsStore, _NoRedirect, validate_base_url
from alpharesearchos.proposals import llm_configured, llm_proposal, llm_settings, provider_context


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ["ALPHAOS_LLM_BASE_URL", "ALPHAOS_LLM_MODEL", "ALPHAOS_LLM_API_KEY", "OPENAI_API_KEY",
                "ALPHAOS_LLM_TOKEN_FIELD", "ALPHAOS_LLM_TEMPERATURE"]:
        monkeypatch.delenv(key, raising=False)
    # This suite must never reach a real provider, even if a test forgets its mock.
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("unexpected network call"))
    monkeypatch.setattr("urllib.request.build_opener", lambda *a, **k: pytest.fail("unexpected network call"))


def provider(**changes):
    return {"base_url": "https://provider.example/v1", "model": "manual-model", "api_key": "secret-never-return",
            "token_field": "max_completion_tokens", "temperature": None, **changes}


def test_environment_fallback_and_no_read_side_effect(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "state")
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")
    public = store.public_config()
    assert public["source"] == "environment" and public["configured"]
    assert public["api_key_set"] and "api_key" not in public
    assert "env-secret" not in json.dumps(public)
    assert store.resolve()["api_key"] == "env-secret"
    assert not store.root.exists()


def test_private_atomic_save_blank_retention_and_explicit_clear(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "state")
    public = store.update(provider())
    assert public["configured"] and public["source"] == "file"
    assert "secret-never-return" not in json.dumps(public)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert json.loads(store.path.read_text())["api_key"] == "secret-never-return"
    before = dict(os.environ)
    store.update({"api_key": "", "model": "new-model", "temperature": 0})
    assert store.resolve()["api_key"] == "secret-never-return"
    assert store.public_config()["temperature"] == 0
    assert dict(os.environ) == before
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reappear")
    public = store.update({"clear_api_key": True})
    assert not public["api_key_set"] and not public["configured"]
    assert store.resolve()["api_key"] == ""
    assert list(store.root.iterdir()) == [store.path]


def test_partial_file_settings_preserve_environment_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    store = SettingsStore(tmp_path / "state")
    public = store.update({"model": "managed-model", "api_key": ""})
    assert public["configured"]
    assert public["field_sources"]["api_key"] == "environment"
    assert public["field_sources"]["model"] == "file"
    assert "api_key" not in json.loads(store.path.read_text())


@pytest.mark.parametrize("url", [
    "http://provider.example/v1", "ftp://localhost/v1", "https://key:password@provider.example/v1",
    "https://provider.example/v1?key=secret", "https://provider.example/v1#secret", "https://provider.example/?",
    "https://provider.example/#", "https://provider.example:99999/v1", "http://127.0.0.1.evil/v1",
    "https:///missing-host", "https://provider.example/\nsecret", "https://provider.example\\@evil/v1",
])
def test_invalid_provider_urls_are_rejected_without_echo(url):
    with pytest.raises(ValueError) as error:
        validate_base_url(url)
    assert "secret" not in str(error.value) and "password" not in str(error.value)


@pytest.mark.parametrize("url", ["https://provider.example/v1/", "http://localhost:11434/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8080/v1"])
def test_https_and_local_providers_are_allowed(url):
    assert validate_base_url(url) == url.rstrip("/")


@pytest.mark.parametrize("patch", [
    {"token_field": "max_output_tokens"}, {"token_field": []}, {"temperature": "nan"},
    {"temperature": float("inf")}, {"temperature": True}, {"temperature": 2.1},
    {"api_key": "key\nheader"}, {"clear_api_key": "true"}, {"unknown": "value"},
    {"api_key": "new", "clear_api_key": True},
])
def test_invalid_updates_preserve_saved_configuration(tmp_path, patch):
    store = SettingsStore(tmp_path / "state")
    store.update(provider())
    previous = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.update(patch)
    assert store.path.read_bytes() == previous


def test_settings_symlink_is_not_read_or_overwritten(tmp_path):
    store = SettingsStore(tmp_path / "state")
    store.root.mkdir()
    target = tmp_path / "unrelated-secret.json"
    target.write_text('{"api_key":"unrelated-secret"}')
    store.path.symlink_to(target)
    for operation in [store.public_config, store.resolve, lambda: store.update({"model": "x"})]:
        with pytest.raises(ValueError, match="symbolic"):
            operation()
    assert "unrelated-secret" in target.read_text()
    assert store.path.is_symlink()


def test_settings_directory_symlink_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        SettingsStore(link).update(provider())
    assert not list(actual.iterdir())


def test_corrupt_settings_error_does_not_echo_content(tmp_path):
    store = SettingsStore(tmp_path)
    store.path.write_text('{"api_key":"secret-never-echo" BROKEN')
    with pytest.raises(ValueError) as error:
        store.public_config()
    assert "secret" not in str(error.value)


def test_provider_context_restores_and_copies_settings(monkeypatch):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    snapshot = provider()
    with provider_context(snapshot):
        snapshot["model"] = "later-mutation"
        assert llm_settings()["model"] == "manual-model"
        assert llm_configured()
        with provider_context(provider(model="nested")):
            assert llm_settings()["model"] == "nested"
        assert llm_settings()["model"] == "manual-model"
    assert llm_settings()["model"] == "env-model"
    assert llm_configured(provider(api_key="")) is False


def test_provider_context_isolated_between_threads_and_tasks():
    barrier = threading.Barrier(2)

    def worker(model):
        with provider_context(provider(model=model)):
            barrier.wait(timeout=5)
            return llm_settings()["model"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, model) for model in ["thread-a", "thread-b"]]
        assert [f.result(timeout=5) for f in futures] == ["thread-a", "thread-b"]

    async def run():
        async def task(model):
            with provider_context(provider(model=model)):
                await asyncio.sleep(0)
                return llm_settings()["model"]
        return await asyncio.gather(task("task-a"), task("task-b"))
    assert asyncio.run(run()) == ["task-a", "task-b"]
    assert llm_settings()["model"] == ""


def test_llm_proposal_uses_scoped_settings_and_zero_temperature(monkeypatch):
    captured = []

    def response(request, **kwargs):
        captured.append(request)
        return io.BytesIO(json.dumps({"choices": [{"message": {"content":
            '{"name":"x","hypothesis":"y","expression":"rank(close)"}'}}]}).encode())
    fake_opener(monkeypatch, response)
    with provider_context(provider(temperature=0, token_field="max_tokens")):
        _, usage = llm_proposal("trend", [])
    body = json.loads(captured[0].data)
    assert captured[0].full_url == "https://provider.example/v1/chat/completions"
    assert captured[0].get_header("Authorization") == "Bearer secret-never-return"
    assert body["model"] == "manual-model" and body["temperature"] == 0 and body["max_tokens"] == 700
    assert "secret-never-return" not in json.dumps(usage)


def fake_opener(monkeypatch, callback):
    class Opener:
        def open(self, request, **kwargs):
            return callback(request, **kwargs)
    monkeypatch.setattr("urllib.request.build_opener", lambda *a, **k: Opener())


def test_explicit_connection_probe_uses_real_protocol_but_never_returns_body(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "state")
    store.update(provider())
    captured = []

    def respond(request, **kwargs):
        captured.append(request)
        proposal = {"name": "动量", "hypothesis": "secret-never-return", "features": ["ret(close,20)"],
                    "model": "rank", "model_params": {"alpha": 1, "train_window": 504, "retrain_every": 63,
                                                      "horizon": 5, "smoothing": 3}, "rationale": "连接检查"}
        return io.BytesIO(json.dumps({"model": "secret-never-return", "choices": [{"message": {"content": json.dumps(proposal)}}]}).encode())
    fake_opener(monkeypatch, respond)
    result = store.test_connection()
    assert result["ok"] and result["model"] == "manual-model"
    assert result["latency_ms"] >= 0
    assert "secret-never-return" not in json.dumps(result)
    assert len(captured) == 1
    body = json.loads(captured[0].data)
    assert body["max_completion_tokens"] == 1800
    assert body["messages"][0]["role"] == "system"
    assert "model_params" in body["messages"][0]["content"]
    assert "Connection protocol check" in body["messages"][1]["content"]
    assert "temperature" not in body


@pytest.mark.parametrize("kind", ["http", "connection", "json", "empty", "plain_text"])
def test_connection_failures_redact_provider_messages(tmp_path, monkeypatch, kind):
    store = SettingsStore(tmp_path / "state")
    store.update(provider())

    def respond(request, **kwargs):
        if kind == "http":
            raise urllib.error.HTTPError(request.full_url, 401, "secret-never-return", {}, io.BytesIO(b"secret-never-return"))
        if kind == "connection":
            raise urllib.error.URLError("secret-never-return")
        if kind == "empty":
            return io.BytesIO(b'{"choices":[{"message":{"content":null}}]}')
        if kind == "plain_text":
            return io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
        return io.BytesIO(b'secret-never-return')
    fake_opener(monkeypatch, respond)
    result = store.test_connection()
    assert not result["ok"]
    assert "secret-never-return" not in json.dumps(result)


def test_unconfigured_probe_sends_no_request(tmp_path):
    result = SettingsStore(tmp_path).test_connection()
    assert not result["ok"]
    assert "API key" in result["message"]


def test_probe_never_follows_authorization_redirect():
    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example") is None


@pytest.mark.parametrize("patch", [{"base_url": "https://new.example/v1"},
                                  {"base_url": "https://new.example/v1", "api_key": ""}])
def test_endpoint_change_without_new_key_clears_saved_and_environment_credentials(tmp_path, monkeypatch, patch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    store = SettingsStore(tmp_path)
    store.update(provider())
    public = store.update(patch)
    assert not public["configured"] and not public["api_key_set"]
    assert store.resolve()["api_key"] == ""
    assert json.loads(store.path.read_text())["api_key"] == ""


def test_endpoint_change_clears_an_environment_only_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOS_LLM_MODEL", "environment-model")
    monkeypatch.setenv("ALPHAOS_LLM_BASE_URL", "https://previous.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    store = SettingsStore(tmp_path)
    assert store.public_config()["configured"]
    public = store.update({"base_url": "http://localhost:11434/v1"})
    assert not public["configured"] and store.resolve()["api_key"] == ""
    assert json.loads(store.path.read_text())["api_key"] == ""


@pytest.mark.parametrize("url", ["https://provider.example/v1/", "https://PROVIDER.example:443/v1"])
def test_equivalent_endpoint_and_blank_key_preserve_existing_secret(tmp_path, url):
    store = SettingsStore(tmp_path)
    store.update(provider())
    assert store.update({"base_url": url, "api_key": ""})["configured"]
    assert store.resolve()["api_key"] == "secret-never-return"


def test_new_endpoint_accepts_an_explicit_replacement_key(tmp_path):
    store = SettingsStore(tmp_path)
    store.update(provider())
    assert store.update({"base_url": "https://new.example/v1", "api_key": "new-service-key"})["configured"]
    assert store.resolve()["api_key"] == "new-service-key"


def test_jev_is_optional_and_its_secret_is_managed_independently(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHAOS_JEV_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    store = SettingsStore(tmp_path)
    initial = store.update(provider())
    assert initial["configured"] and not initial["jev_enabled"] and not initial["jev_configured"]
    public = store.update({"jev_enabled": True, "jev_api_key": "JEV_PRIVATE_CANARY"})
    assert public["jev_configured"] and public["jev_api_key_set"]
    assert "jev_api_key" not in public and "JEV_PRIVATE_CANARY" not in json.dumps(public)
    store.update({"jev_api_key": ""})
    assert store.resolve()["jev_api_key"] == "JEV_PRIVATE_CANARY"
    monkeypatch.setenv("TYPESAFE_API_KEY", "must-not-reappear")
    public = store.update({"clear_jev_api_key": True})
    assert not public["jev_configured"] and public["configured"]
    assert store.resolve()["api_key"] == "secret-never-return"
    assert store.resolve()["jev_api_key"] == ""
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


@pytest.mark.parametrize("patch", [
    {"jev_enabled": "false"}, {"jev_min_confidence": .5}, {"jev_min_confidence": 1.1},
    {"jev_min_confidence": True}, {"jev_min_confidence": "nan"}, {"jev_min_confidence": []},
    {"jev_api_key": "bad\nheader"}, {"jev_model": "gpt-test"}, {"jev_base_url": "http://evil.example"},
    {"clear_jev_api_key": "true"}, {"clear_jev_api_key": True, "jev_api_key": "new"},
])
def test_invalid_jev_settings_do_not_overwrite_existing_settings(tmp_path, patch):
    store = SettingsStore(tmp_path)
    store.update(provider())
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.update(patch)
    assert store.path.read_bytes() == before


def test_jev_environment_fallback_never_changes_original_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHAOS_JEV_ENABLED", "true")
    monkeypatch.setenv("TYPESAFE_API_KEY", "JEV_ENV_CANARY")
    monkeypatch.setenv("ALPHAOS_JEV_MIN_CONFIDENCE", "0.8")
    store = SettingsStore(tmp_path)
    settings = store.resolve()
    assert settings["jev_enabled"] and settings["jev_min_confidence"] == .8
    assert settings["jev_api_key"] == "JEV_ENV_CANARY" and settings["api_key"] == ""
    assert "JEV_ENV_CANARY" not in json.dumps(store.public_config())
    assert not store.path.exists()


def test_saved_jev_settings_override_invalid_environment_fallback(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path)
    store.update({"jev_enabled": False, "jev_min_confidence": .8})
    monkeypatch.setenv("ALPHAOS_JEV_ENABLED", "not-a-bool")
    monkeypatch.setenv("ALPHAOS_JEV_MIN_CONFIDENCE", "not-a-number")
    result = store.resolve()
    assert result["jev_enabled"] is False and result["jev_min_confidence"] == .8


def test_jev_explicit_probe_can_test_disabled_gate_and_redacts_errors(tmp_path, monkeypatch):
    from alpharesearchos import jev_client
    store = SettingsStore(tmp_path)
    store.update({"jev_api_key": "PRIVATE_JEV_KEY"})
    calls = []
    def respond(context, *, settings, timeout):
        calls.append(context)
        assert not settings["jev_enabled"] and timeout == 15
        return {"decision": "reject"}, {"model": "jev-1.13.0"}
    monkeypatch.setattr(jev_client, "review_gate", respond)
    assert store.test_jev_connection()["ok"] and len(calls) == 1
    def fail(*args, **kwargs):
        raise ValueError("PRIVATE_JEV_KEY")
    monkeypatch.setattr(jev_client, "review_gate", fail)
    result = store.test_jev_connection()
    assert not result["ok"] and "PRIVATE_JEV_KEY" not in json.dumps(result)
