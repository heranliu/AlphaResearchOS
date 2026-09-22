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
    monkeypatch.setattr("urllib.request.urlopen", response)
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
        return io.BytesIO(json.dumps({"model": "secret-never-return", "choices": [{"message": {"content": "secret-never-return"}}]}).encode())
    fake_opener(monkeypatch, respond)
    result = store.test_connection()
    assert result["ok"] and result["model"] == "manual-model"
    assert result["latency_ms"] >= 0
    assert "secret-never-return" not in json.dumps(result)
    assert len(captured) == 1
    body = json.loads(captured[0].data)
    assert body["max_completion_tokens"] == 700
    assert body["messages"] == [{"role": "user", "content": "Reply with OK."}]
    assert "temperature" not in body


@pytest.mark.parametrize("kind", ["http", "connection", "json", "empty"])
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
