"""Structured research proposals through the installed, authenticated Codex CLI.

The CLI owns authentication. This module never reads auth.json or exports tokens.
Only the research direction and development feedback are passed on stdin.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tomllib
from pathlib import Path

_STATUS_CACHE = {}
_STATUS_LOCK = threading.Lock()
CODEX_RESERVATION = 65536
SCHEMA = {
    "type": "object", "properties": {key: {"type": "string"} for key in ("name", "hypothesis", "expression")},
    "required": ["name", "hypothesis", "expression"], "additionalProperties": False,
}


def executable():
    configured = os.environ.get("ALPHAOS_CODEX_BIN")
    found = configured or shutil.which("codex")
    if not found:
        bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
        found = str(bundled) if bundled.is_file() else None
    return found if found and Path(found).is_file() and os.access(found, os.X_OK) else None


def default_model():
    # Read only the model preference; credentials remain owned by the CLI.
    config = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    try:
        value = tomllib.loads(config.read_text()).get("model")
        return value if isinstance(value, str) and 0 < len(value) <= 200 and not any(ord(c) < 32 for c in value) else None
    except (OSError, ValueError):
        return None


def codex_status(*, refresh=False):
    binary = executable()
    with _STATUS_LOCK:
        if not refresh and _STATUS_CACHE.get("binary") == binary and time.monotonic() - _STATUS_CACHE.get("time", -100) < 15:
            return dict(_STATUS_CACHE["value"])
        result = {"available": bool(binary), "authenticated": False, "version": None,
                  "login_method": None, "default_model": default_model()}
        if binary:
            try:
                version = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5, check=False)
                login = subprocess.run([binary, "login", "status"], capture_output=True, text=True, timeout=5, check=False)
                result["version"] = version.stdout.strip()[:100] if version.returncode == 0 else None
                result["authenticated"] = login.returncode == 0
                output = login.stdout + login.stderr
                result["login_method"] = "ChatGPT" if result["authenticated"] and "ChatGPT" in output else "CLI" if result["authenticated"] else None
            except (OSError, subprocess.TimeoutExpired):
                pass
        _STATUS_CACHE.update(binary=binary, time=time.monotonic(), value=result)
        return dict(result)


def codex_proposal(direction, history, *, model="", timeout=90):
    from .proposals import SYSTEM
    compact = [{k: trial.get(k) for k in ["id", "expression", "score", "status", "reason"]} for trial in history[-12:]]
    prompt = SYSTEM + "\nDo not use tools, read files, browse, or run commands. Work only with the context below. " \
        "Return the required JSON. Keep the hypothesis concise and written in Chinese.\n" + json.dumps(
            {"direction": direction, "development_trials": compact}, ensure_ascii=False)
    candidate, usage = codex_structured(prompt, SCHEMA, model=model, timeout=timeout)
    for key, limit in [("name", 100), ("hypothesis", 1000), ("expression", 1000)]:
        if not isinstance(candidate.get(key), str) or not 1 <= len(candidate[key]) <= limit:
            raise ValueError(f"Invalid Codex proposal field: {key}")
    result = {key: candidate[key] for key in ("name", "hypothesis", "expression")}
    result.update(parents=[], origin="codex_cli")
    return result, usage


def codex_structured(prompt, response_schema, *, model="", timeout=90):
    """One isolated structured call, shared by proposer and independent reviewer."""
    status = codex_status()
    if not status["available"] or not status["authenticated"]:
        raise ValueError("本机 Codex 未安装或未登录；请先运行 codex login")
    selected_model = model.strip() or status["default_model"]
    if len(prompt.encode()) > 22000:
        raise ValueError("Codex prompt exceeds 22 KB request cap")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="alphaos-codex-") as scratch:
        root = Path(scratch)
        schema, output = root / "schema.json", root / "proposal.json"
        schema.write_text(json.dumps(response_schema))
        args = [executable(), "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--color", "never", "--json", "--output-schema", str(schema),
                "--output-last-message", str(output), "-C", str(root),
                "-c", 'model_reasoning_effort="medium"']
        if selected_model:
            args += ["--model", selected_model]
        args.append("-")
        with (root / "events.jsonl").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                process.communicate(prompt.encode(), timeout=max(.1, min(90, timeout)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    process.communicate()
                raise RuntimeError("Codex 请求超时；本次请求已计数，不自动重试") from None
        if process.returncode:
            raise RuntimeError(f"Codex 请求失败（退出码 {process.returncode}）；检查登录和模型可用性")
        if not output.is_file() or output.stat().st_size > 100000:
            raise ValueError("Codex 未返回有效大小的结构化提案")
        candidate = json.loads(output.read_text())
        if not isinstance(candidate, dict):
            raise ValueError("Codex must return one JSON object")
        events_file = root / "events.jsonl"
        if events_file.stat().st_size > 2_000_000:
            raise ValueError("Codex event stream exceeds 2 MB")
        usage, tool_events = {}, 0
        for line in events_file.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("type") == "turn.completed":
                usage = record.get("usage", {})
            if record.get("type") in {"item.started", "item.updated", "item.completed"} and record.get("item", {}).get("type") in {"command_execution", "file_change", "mcp_tool_call", "web_search"}:
                tool_events += 1
        if tool_events:
            raise ValueError("Codex 使用了提案任务以外的工具；结果未进入回测")
    totals = [usage.get("input_tokens"), usage.get("output_tokens")]
    total = sum(totals) if all(type(value) is int and value >= 0 for value in totals) else None
    return candidate, {"provider": "codex_cli", "model": selected_model or "CLI default", "cli_version": status["version"],
                    "reported_total_tokens": total, "token_usage": usage, "request_bytes": len(prompt.encode()),
                    "seconds": round(time.monotonic() - started, 3), "tool_events": tool_events,
                    "generation_options": {"reasoning_effort": "medium", "output_schema": True, "sandbox": "read-only"},
                    "budget_note": "65536-token admission reservation; CLI has no enforced output token cap"}
