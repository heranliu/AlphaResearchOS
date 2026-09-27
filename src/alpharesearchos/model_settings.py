"""Project-local provider settings and an explicit, small connection probe.

Secrets stay in settings.json (0600), never in public API responses. Updating or
reading settings performs no network operation; only test_connection does.
"""
from __future__ import annotations

import ipaddress
import json
import math
import os
import secrets
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

JEV_DEFAULTS = {"jev_enabled": False, "jev_base_url": "https://api.typesafe.ai/v1",
                "jev_model": "jev-1.13.0", "jev_api_key": "", "jev_min_confidence": 0.7}
_SECRET_FIELDS = frozenset({"api_key", "jev_api_key"})
_FIELDS = frozenset({"provider", "codex_model", "base_url", "model", "api_key", "token_field", "temperature",
                     *JEV_DEFAULTS})


def jev_environment_settings(*, raw=False):
    enabled = os.environ.get("ALPHAOS_JEV_ENABLED", "false").strip().lower()
    result = {"jev_enabled": {"true": True, "1": True, "false": False, "0": False}.get(enabled, enabled),
            "jev_base_url": os.environ.get("ALPHAOS_JEV_BASE_URL", JEV_DEFAULTS["jev_base_url"]),
            "jev_model": os.environ.get("ALPHAOS_JEV_MODEL", JEV_DEFAULTS["jev_model"]),
            "jev_api_key": os.environ.get("ALPHAOS_JEV_API_KEY", os.environ.get("TYPESAFE_API_KEY", "")),
            "jev_min_confidence": os.environ.get("ALPHAOS_JEV_MIN_CONFIDENCE", "0.7")}
    return result if raw else validate_jev_settings(result)


def validate_jev_settings(settings, *, require_key=False):
    """Normalize a frozen snapshot; never expose secrets in validation errors."""
    import re
    result = {**JEV_DEFAULTS, **{key: settings[key] for key in JEV_DEFAULTS if key in settings}}
    if type(result["jev_enabled"]) is not bool:
        raise ValueError("jev_enabled must be a boolean")
    result["jev_base_url"] = validate_base_url(result["jev_base_url"])
    for field, limit in [("jev_model", 100), ("jev_api_key", 4096)]:
        value = result[field]
        if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
            raise ValueError(f"{field} must be a single-line string of at most {limit} characters")
        result[field] = value.strip()
    key = result["jev_api_key"]
    if key and (not key.isascii() or any(ord(char) < 33 or ord(char) > 126 for char in key)):
        raise ValueError("jev_api_key must contain only visible ASCII characters without spaces")
    if not re.fullmatch(r"jev-(?:[0-9]+\.[0-9]+\.[0-9]+|latest|preview)", result["jev_model"]):
        raise ValueError("jev_model must be a Jev version or jev-latest/jev-preview")
    value = result["jev_min_confidence"]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("jev_min_confidence must be a number in (0.5, 1]")
    try:
        value = float(value)
    except ValueError:
        raise ValueError("jev_min_confidence must be a number in (0.5, 1]") from None
    if not math.isfinite(value) or not 0.5 < value <= 1:
        raise ValueError("jev_min_confidence must be a number in (0.5, 1]")
    result["jev_min_confidence"] = value
    if require_key and not result["jev_api_key"]:
        raise ValueError("Jev is enabled; configure its API key before starting research")
    return result


def validate_base_url(value: str) -> str:
    """Permit HTTPS providers and plain HTTP on literal loopback/localhost."""
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError("base_url must be a nonempty URL of at most 2048 characters")
    value = value.strip().rstrip("/")
    if any(ord(char) <= 32 for char in value) or "\\" in value:
        raise ValueError("base_url contains invalid URL characters")
    try:
        parts = urllib.parse.urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ValueError("base_url is not a valid provider URL") from None
    if not host or parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError("base_url must not include credentials, query parameters or a fragment")
    if "?" in value or "#" in value or (port is not None and not 1 <= port <= 65535):
        raise ValueError("base_url must not include query/fragment and must use a valid port")
    if parts.scheme not in {"https", "http"}:
        raise ValueError("base_url must use HTTPS or loopback HTTP")
    if parts.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback:
            raise ValueError("HTTP is allowed only for localhost or a loopback IP address")
    return value


def _temperature(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("temperature must be blank or a finite number between 0 and 2")
    try:
        result = float(value)
    except ValueError:
        raise ValueError("temperature must be blank or a finite number between 0 and 2") from None
    if not math.isfinite(result) or not 0 <= result <= 2:
        raise ValueError("temperature must be blank or a finite number between 0 and 2")
    return result


def _validate_provider(provider):
    provider = dict(provider)
    if provider.get("provider") not in {"openai_compatible", "codex_cli"}:
        raise ValueError("provider must be openai_compatible or codex_cli")
    provider["base_url"] = validate_base_url(provider["base_url"])
    for field, limit in [("model", 200), ("codex_model", 200), ("api_key", 4096)]:
        value = provider[field]
        if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 for c in value):
            raise ValueError(f"{field} must be a single-line string of at most {limit} characters")
        provider[field] = value.strip()
    if not isinstance(provider["token_field"], str) or provider["token_field"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("token_field must be max_tokens or max_completion_tokens")
    provider["temperature"] = _temperature(provider["temperature"])
    provider.update(validate_jev_settings(provider))
    return provider


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a saved Authorization header to a redirect destination.
        return None


class SettingsStore:
    """One server's project-specific settings store; no process environment edits."""

    def __init__(self, state_root: str | Path):
        self.root = Path(os.path.abspath(os.fspath(state_root)))
        self.path = self.root / "settings.json"
        self._lock = threading.RLock()

    def _directory_fd(self, *, create=False):
        # Refuse symlinks in this explicit storage path, including parent paths.
        for item in (self.root, *self.root.parents):
            if item.is_symlink():
                raise ValueError("Model settings path must not contain symbolic links")
        if create:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            return os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            if not create:
                return None
            raise
        except OSError:
            raise ValueError("Cannot access model settings directory safely") from None

    def _read(self):
        directory = self._directory_fd()
        if directory is None:
            return {}
        try:
            try:
                fd = os.open("settings.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            except FileNotFoundError:
                return {}
            except OSError:
                raise ValueError("Cannot read model settings safely; symbolic links are forbidden") from None
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                    raise ValueError("Model settings must be a small regular JSON file")
                try:
                    data = json.load(stream)
                except (ValueError, UnicodeError):
                    raise ValueError("Saved model settings are not valid JSON") from None
            if not isinstance(data, dict) or set(data) - _FIELDS:
                raise ValueError("Saved model settings have an invalid schema")
            return data
        finally:
            os.close(directory)

    def _write(self, settings):
        directory = self._directory_fd(create=True)
        temp = ".settings-" + secrets.token_hex(8) + ".tmp"
        try:
            try:
                target = os.stat("settings.json", dir_fd=directory, follow_symlinks=False)
                if not stat.S_ISREG(target.st_mode):
                    raise ValueError("Model settings destination must be a regular file, not a symbolic link")
            except FileNotFoundError:
                pass
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(settings, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, "settings.json", src_dir_fd=directory, dst_dir_fd=directory)
            os.fsync(directory)
        finally:
            try:
                os.unlink(temp, dir_fd=directory)
            except FileNotFoundError:
                pass
            os.close(directory)

    def _effective(self, saved):
        from .proposals import environment_settings
        return _validate_provider({**environment_settings(raw_jev=True), **saved})

    def resolve(self):
        """Private request snapshot, including the key. Never serialize to clients."""
        with self._lock:
            return self._effective(self._read())

    def public_config(self):
        from .codex_provider import codex_status
        with self._lock:
            saved = self._read()
            effective = self._effective(saved)
            result = {key: effective[key] for key in _FIELDS - _SECRET_FIELDS}
            status = codex_status()
            configured = (status["available"] and status["authenticated"]) if effective["provider"] == "codex_cli" else bool(effective["api_key"] and effective["model"])
            result.update({"api_key_set": bool(effective["api_key"]),
                           "jev_api_key_set": bool(effective["jev_api_key"]),
                           "jev_configured": bool(effective["jev_api_key"] and effective["jev_model"]),
                           "configured": configured, "codex": status,
                           "source": "file" if saved else "environment",
                           "field_sources": {key: "file" if key in saved else "environment" for key in sorted(_FIELDS)}})
            return result

    def update(self, body):
        if not isinstance(body, dict) or set(body) - (_FIELDS | {"clear_api_key", "clear_jev_api_key"}):
            raise ValueError("Unknown model settings field")
        with self._lock:
            saved = self._read()
            patch = {key: value for key, value in body.items() if key in _FIELDS}
            for key in _SECRET_FIELDS:
                clear = body.get("clear_" + key, False)
                if type(clear) is not bool:
                    raise ValueError(f"clear_{key} must be a boolean")
                if key in patch:
                    if not isinstance(patch[key], str):
                        raise ValueError(f"{key} must be a string")
                    if not patch[key].strip():
                        patch.pop(key)
                    elif clear:
                        raise ValueError(f"Cannot set and clear {key} in the same request")
                if clear:
                    patch[key] = ""
            effective = self._effective({**saved, **patch})
            # Persist only explicitly managed fields; unmodified env values stay fallback.
            updated = {key: effective[key] for key in set(saved) | set(patch)}
            self._write(updated)
            return self.public_config()

    def test_jev_connection(self):
        """Explicit protocol probe, even while the research gate is disabled."""
        from .jev_client import review_gate
        started = time.monotonic()
        settings = self.resolve()
        result = {"ok": False, "model": settings["jev_model"], "latency_ms": 0}
        if not settings["jev_api_key"]:
            return {**result, "message": "请先配置 Jev API 密钥"}
        try:
            _, usage = review_gate({"direction": "Connection test only; no market evidence supplied; revise or reject.",
                                    "candidate": {"name": "Connection test"}}, settings=settings, timeout=15)
            result.update(ok=True, model=usage["model"], message="Jev 连接成功，已返回结构化判断")
        except (ValueError, RuntimeError, OSError, TypeError, KeyError):
            result["message"] = "Jev 连接或协议校验失败；请检查地址、版本和密钥"
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        return result

    def test_connection(self):
        """One explicit billable probe; no retries, model listing or startup calls."""
        from .proposals import _generation_options
        started = time.monotonic()
        provider = self.resolve()
        if provider["provider"] == "codex_cli":
            from .codex_provider import codex_proposal
            result = {"ok": False, "model": provider["codex_model"] or "CLI default", "latency_ms": 0}
            try:
                _, usage = codex_proposal("连接测试：给出简单的20日价格动量因子。", [], model=provider["codex_model"], timeout=90)
                result.update(ok=True, model=usage["model"], message="Codex 连接成功，已返回结构化提案")
            except (ValueError, RuntimeError, OSError, TypeError, KeyError):
                result["message"] = "Codex 连接失败或超时；请检查本机登录和模型可用性"
            result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
            return result
        result = {"ok": False, "model": provider["model"], "latency_ms": 0}
        if not provider["model"] or not provider["api_key"]:
            return {**result, "message": "Set a model and API key before testing the connection"}
        payload = {"model": provider["model"], **_generation_options(provider),
                   "messages": [{"role": "user", "content": "Reply with OK."}]}
        request = urllib.request.Request(provider["base_url"] + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + provider["api_key"]})
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=15) as response:
                raw = response.read(100001)
            if len(raw) > 100000:
                raise ValueError("oversized")
            data = json.loads(raw)
            content = data["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                result["message"] = "Provider responded without text; check model and reasoning/completion budget"
            else:
                result.update(ok=True, message="Connection succeeded; model returned text")
        except urllib.error.HTTPError as exc:
            result.update(http_status=exc.code, message=f"Provider HTTP {exc.code}; check endpoint, model and credentials")
        except (urllib.error.URLError, TimeoutError, OSError):
            result["message"] = "Provider connection failed or timed out"
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            result["message"] = "Provider response did not match the Chat Completions text protocol"
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        return result
