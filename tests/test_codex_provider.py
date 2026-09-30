"""Exercise Codex transport and provider isolation without any subprocess/network I/O."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpharesearchos import codex_provider as codex
from alpharesearchos import proposals
from alpharesearchos.model_settings import SettingsStore

REAL_STATUS = codex.codex_status
PROPOSAL = {"name": "风险调整动量", "hypothesis": "检验低波动动量在开发期的稳定性", "expression": "rank(ret(close,20))"}
STATUS = {"available": True, "authenticated": True, "version": "codex-cli test-version", "login_method": "ChatGPT", "default_model": "installed-default"}


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch, tmp_path):
    for name in ["ALPHAOS_LLM_PROVIDER", "ALPHAOS_CODEX_MODEL", "ALPHAOS_CODEX_BIN", "ALPHAOS_LLM_BASE_URL",
                 "ALPHAOS_LLM_MODEL", "ALPHAOS_LLM_API_KEY", "OPENAI_API_KEY", "ALPHAOS_LLM_TOKEN_FIELD",
                 "ALPHAOS_LLM_TEMPERATURE"]:
        monkeypatch.delenv(name, raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("This test must not call a real subprocess, login probe, or network service")

    monkeypatch.setattr(codex.subprocess, "run", forbidden)
    monkeypatch.setattr(codex.subprocess, "Popen", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    monkeypatch.setattr(codex, "executable", lambda: str(tmp_path / "mock-codex"))
    monkeypatch.setattr(codex, "default_model", lambda: "installed-default")
    monkeypatch.setattr(codex, "codex_status", lambda **kwargs: dict(STATUS))
    codex._STATUS_CACHE.clear()
    real_temporary_directory = codex.tempfile.TemporaryDirectory
    monkeypatch.setattr(codex.tempfile, "TemporaryDirectory", lambda **kwargs: real_temporary_directory(dir=tmp_path, **kwargs))


class ProcessHarness:
    """Record process arguments and materialize only the CLI's documented files."""

    def __init__(self, monkeypatch, *, output=None, events=None, returncode=0, timeout=False, missing_output=False):
        self.output = json.dumps(PROPOSAL, ensure_ascii=False) if output is None else output
        self.events = events if events is not None else [{"type": "turn.completed", "usage": {"input_tokens": 240, "cached_input_tokens": 80, "output_tokens": 40}}]
        self.returncode = returncode
        self.timeout = timeout
        self.missing_output = missing_output
        self.processes = []
        self.kills = []
        monkeypatch.setattr(codex.subprocess, "Popen", self.spawn)
        monkeypatch.setattr(codex.os, "killpg", lambda pid, sig: self.kills.append((pid, sig)))

    def spawn(self, args, **kwargs):
        process = SimpleNamespace(args=list(args), options=kwargs, pid=48122, returncode=self.returncode, communications=[])
        process.root = Path(args[args.index("-C") + 1])
        process.schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text())
        self.processes.append(process)

        def communicate(data=None, timeout=None):
            process.communications.append({"data": data, "timeout": timeout})
            if self.timeout and len(process.communications) == 1:
                raise subprocess.TimeoutExpired(args, timeout)
            if self.timeout:
                process.returncode = -signal.SIGKILL
                return None, None
            if not self.missing_output:
                Path(args[args.index("--output-last-message") + 1]).write_text(self.output)
            for event in self.events:
                kwargs["stdout"].write((json.dumps(event) + "\n").encode())
            kwargs["stderr"].write(b"private-provider-diagnostic-must-not-escape")
            return None, None

        process.communicate = communicate
        return process


def test_structured_proposal_uses_stdin_schema_isolated_directory_and_reports_usage(monkeypatch):
    harness = ProcessHarness(monkeypatch)
    candidate, usage = codex.codex_proposal("中文研究方向", [], model="chosen-model", timeout=12)
    process = harness.processes[0]
    assert len(harness.processes) == 1
    assert candidate == {**PROPOSAL, "parents": [], "origin": "codex_cli"}
    assert usage["provider"] == "codex_cli" and usage["model"] == "chosen-model"
    assert usage["reported_total_tokens"] == 280  # Cached input is a subset, never added twice.
    assert usage["token_usage"]["cached_input_tokens"] == 80
    assert usage["tool_events"] == 0
    assert process.schema["required"] == ["name", "hypothesis", "expression"]
    assert process.schema["additionalProperties"] is False
    assert process.options["start_new_session"] is True
    assert process.options["stdin"] == subprocess.PIPE
    assert not process.options.get("shell", False)
    assert process.args[process.args.index("--sandbox") + 1] == "read-only"
    assert process.args[process.args.index("--model") + 1] == "chosen-model"
    assert "--ignore-user-config" in process.args and "--ephemeral" in process.args
    assert process.args[-1] == "-" and "中文研究方向" not in " ".join(process.args)
    assert "中文研究方向" in process.communications[0]["data"].decode()
    assert process.communications[0]["timeout"] == 12
    assert usage["request_bytes"] == len(process.communications[0]["data"])
    assert not process.root.exists()


def test_development_feedback_omits_holdout_curves_private_fields_and_old_history(monkeypatch):
    harness = ProcessHarness(monkeypatch)
    history = [{"id": f"trial-{index}", "expression": f"ret(close,{index + 1})", "score": index / 10,
                "status": "ok", "reason": None, "holdout": {"sharpe": "HOLDOUT_SENTINEL"},
                "metrics": {"future_label": "FUTURE_LABEL_SENTINEL"}, "curve": ["CURVE_SENTINEL"],
                "private_config": "SECRET_SENTINEL"} for index in range(15)]
    codex.codex_proposal("测试开发反馈", history)
    prompt = harness.processes[0].communications[0]["data"].decode()
    context = json.loads(prompt.rsplit("\n", 1)[1])
    assert [row["id"] for row in context["development_trials"]] == [f"trial-{index}" for index in range(3, 15)]
    assert all(set(row) == {"id", "expression", "score", "status", "reason"} for row in context["development_trials"])
    assert not any(secret in prompt for secret in ["HOLDOUT_SENTINEL", "FUTURE_LABEL_SENTINEL", "CURVE_SENTINEL", "SECRET_SENTINEL"])


@pytest.mark.parametrize("event_type", ["item.started", "item.updated", "item.completed"])
@pytest.mark.parametrize("tool_type", ["command_execution", "file_change", "mcp_tool_call", "web_search"])
def test_tool_activity_invalidates_otherwise_valid_proposal(monkeypatch, event_type, tool_type):
    harness = ProcessHarness(monkeypatch, events=[{"type": event_type, "item": {"type": tool_type}}])
    with pytest.raises(ValueError, match="工具"):
        codex.codex_proposal("研究方向", [])
    assert len(harness.processes) == 1
    assert not harness.processes[0].root.exists()


def test_timeout_kills_process_group_reaps_once_and_never_retries(monkeypatch):
    harness = ProcessHarness(monkeypatch, timeout=True)
    with pytest.raises(RuntimeError, match="超时"):
        codex.codex_proposal("研究方向", [], timeout=4)
    assert len(harness.processes) == 1
    process = harness.processes[0]
    assert harness.kills == [(process.pid, signal.SIGKILL)]
    assert process.communications == [{"data": process.communications[0]["data"], "timeout": 4}, {"data": None, "timeout": None}]
    assert process.returncode == -signal.SIGKILL
    assert not process.root.exists()


def test_timeout_reaps_even_when_process_exits_before_group_kill(monkeypatch):
    harness = ProcessHarness(monkeypatch, timeout=True)

    def already_exited(pid, sig):
        raise ProcessLookupError("process already exited")

    monkeypatch.setattr(codex.os, "killpg", already_exited)
    with pytest.raises(RuntimeError, match="超时"):
        codex.codex_proposal("研究方向", [], timeout=4)
    assert len(harness.processes[0].communications) == 2


def test_failed_process_does_not_echo_stderr_or_retry(monkeypatch):
    harness = ProcessHarness(monkeypatch, returncode=7)
    with pytest.raises(RuntimeError) as error:
        codex.codex_proposal("研究方向", [])
    assert "7" in str(error.value)
    assert "private-provider-diagnostic" not in str(error.value)
    assert len(harness.processes) == 1


@pytest.mark.parametrize("output", ["not-json", "[]", '{"name":"x","hypothesis":"h"}', '{"name":"x","hypothesis":"h","expression":123}', json.dumps({**PROPOSAL, "expression": "x" * 1001})])
def test_invalid_structured_outputs_are_rejected(monkeypatch, output):
    ProcessHarness(monkeypatch, output=output)
    with pytest.raises(ValueError):
        codex.codex_proposal("研究方向", [])


def test_missing_output_and_oversized_prompt_reject_without_paid_retry(monkeypatch):
    harness = ProcessHarness(monkeypatch, missing_output=True)
    with pytest.raises(ValueError, match="结构化"):
        codex.codex_proposal("研究方向", [])
    assert len(harness.processes) == 1
    with pytest.raises(ValueError, match="22 KB"):
        codex.codex_proposal("中" * 8000, [])
    assert len(harness.processes) == 1


def test_unknown_usage_stays_unknown_and_default_model_is_explicit(monkeypatch):
    ProcessHarness(monkeypatch, events=[])
    _, usage = codex.codex_proposal("研究方向", [], model="")
    assert usage["model"] == "installed-default"
    assert usage["reported_total_tokens"] is None
    assert usage["token_usage"] == {}


@pytest.mark.parametrize("available,authenticated", [(False, False), (True, False)])
def test_unavailable_or_logged_out_cli_never_starts_generation(monkeypatch, available, authenticated):
    monkeypatch.setattr(codex, "codex_status", lambda **kwargs: {**STATUS, "available": available, "authenticated": authenticated})
    with pytest.raises(ValueError, match="未安装或未登录"):
        codex.codex_proposal("研究方向", [])


def test_status_probe_returns_only_safe_summary_and_uses_cache(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="codex-cli test-version\n" if args[-1] == "--version" else "",
                               stderr="Logged in using ChatGPT; private-token-sentinel")

    monkeypatch.setattr(codex.subprocess, "run", run)
    first = REAL_STATUS(refresh=True)
    second = REAL_STATUS()
    assert first == second == STATUS
    assert len(calls) == 2
    assert all(options["timeout"] == 5 for _, options in calls)
    assert "private-token-sentinel" not in json.dumps(first)
    first["authenticated"] = False
    assert REAL_STATUS()["authenticated"] is True


def api_provider(**patch):
    return {"provider": "openai_compatible", "codex_model": "", "base_url": "https://provider.example/v1", "model": "api-model", "api_key": "private-api-key", "token_field": "max_completion_tokens", "temperature": None, **patch}


def test_provider_context_routes_codex_without_touching_api_and_restores_parent(monkeypatch):
    calls = []

    def proposal(direction, history, *, model="", timeout=90):
        calls.append((direction, history, model, timeout))
        return {**PROPOSAL, "origin": "codex_cli", "parents": []}, {"provider": "codex_cli"}

    monkeypatch.setattr(codex, "codex_proposal", proposal)
    outer = api_provider()
    inner = api_provider(provider="codex_cli", codex_model="codex-choice", api_key="", model="")
    with proposals.provider_context(outer):
        with proposals.provider_context(inner):
            assert proposals.llm_configured()
            candidate, usage = proposals.llm_proposal("请求方向", [{"id": "T1"}], timeout=17)
            assert candidate["origin"] == usage["provider"] == "codex_cli"
        assert proposals.llm_settings() == outer
    assert calls == [("请求方向", [{"id": "T1"}], "codex-choice", 17)]


def test_switching_to_codex_preserves_api_configuration_and_hides_secrets(tmp_path):
    store = SettingsStore(tmp_path / "state")
    store.update(api_provider())
    public = store.update({"provider": "codex_cli", "codex_model": ""})
    assert public["provider"] == "codex_cli" and public["configured"]
    assert public["codex"] == STATUS
    assert public["codex_model"] == "" and public["model"] == "api-model"
    assert store.resolve()["api_key"] == "private-api-key"
    assert "private-api-key" not in json.dumps(public)
    assert "api_key" not in public
    assert store.update({"provider": "openai_compatible"})["configured"]


def test_codex_configuration_depends_on_login_instead_of_existing_api_key(tmp_path, monkeypatch):
    store = SettingsStore(tmp_path / "state")
    store.update(api_provider())
    monkeypatch.setattr(codex, "codex_status", lambda **kwargs: {**STATUS, "authenticated": False, "login_method": None})
    public = store.update({"provider": "codex_cli"})
    assert public["api_key_set"] and not public["configured"]
    assert not proposals.llm_configured(store.resolve())


def test_settings_connection_test_dispatches_to_codex_only_on_explicit_probe(tmp_path, monkeypatch):
    calls = []

    def proposal(prompt, schema, *, model="", timeout=90):
        calls.append((prompt, schema, model, timeout))
        value = {"name": "动量", "hypothesis": "检验趋势", "features": ["ret(close,20)"], "model": "rank",
                 "model_params": {"alpha": 1, "train_window": 504, "retrain_every": 63, "horizon": 5, "smoothing": 3},
                 "rationale": "连接检查"}
        return value, {"provider": "codex_cli", "model": model or STATUS["default_model"]}

    monkeypatch.setattr(codex, "codex_structured", proposal)
    store = SettingsStore(tmp_path / "state")
    store.update({"provider": "codex_cli", "codex_model": "codex-choice"})
    store.public_config()
    store.resolve()
    assert calls == []
    result = store.test_connection()
    assert result["ok"] is True and result["model"] == "codex-choice"
    assert len(calls) == 1 and calls[0][2] == "codex-choice"
    assert "model_params" in calls[0][1]["properties"]
