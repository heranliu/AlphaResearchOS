"""One bounded Chat Completions request, shared by research and connection checks.

There are no paid retries or redirect destinations. Provider response bodies,
headers and credentials never become exception text or usage metadata.
"""
from __future__ import annotations

import http.client
import json
import math
import re
import time
import urllib.error
import urllib.request


class ModelRequestError(RuntimeError):
    """Safe transport diagnostics, optionally carrying only an HTTP status."""

    def __init__(self, message, *, http_status=None):
        super().__init__(message)
        self.http_status = http_status


def generation_options(settings, completion_tokens):
    """Explicit capabilities work across providers without model-name guesses."""
    token_field = settings.get("token_field", "max_completion_tokens")
    if not isinstance(token_field, str) or token_field not in {"max_completion_tokens", "max_tokens"}:
        raise ValueError("ALPHAOS_LLM_TOKEN_FIELD must be max_completion_tokens or max_tokens")
    options = {token_field: completion_tokens}
    value = settings.get("temperature")
    if value is not None and value != "":
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValueError("ALPHAOS_LLM_TEMPERATURE must be finite between 0 and 2")
        try:
            temperature = float(value)
        except ValueError:
            raise ValueError("ALPHAOS_LLM_TEMPERATURE must be finite between 0 and 2") from None
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("ALPHAOS_LLM_TEMPERATURE must be finite between 0 and 2")
        options["temperature"] = temperature
    return options


def _reject_constant(_value):
    raise ValueError("Nonfinite JSON number")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def json_object(raw):
    """Reject ambiguous/nonfinite JSON without copying model output to errors."""
    try:
        result = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_object)
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("Model returned invalid JSON") from None
    if not isinstance(result, dict):
        raise ValueError("Model must return a JSON object")
    return result


def _usage(value):
    value = value if isinstance(value, dict) else {}

    def count(name):
        item = value.get(name)
        return item if type(item) is int and item >= 0 else None

    prompt, completion, total = (count(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens"))
    # Cached/reasoning tokens are subsets, never extra charges. Missing or
    # malformed counts remain unknown; a valid breakdown bounds the total.
    if prompt is not None and completion is not None:
        total = max(total or 0, prompt + completion)
    return {"reported_total_tokens": total, "input_tokens": prompt, "output_tokens": completion}


def chat_json(settings, messages, *, completion_tokens, timeout, max_request_bytes=22200):
    """Return one JSON object and normalized usage; never call a second endpoint."""
    from .model_settings import _NoRedirect, validate_base_url

    base_url = validate_base_url(settings.get("base_url", ""))
    model, key = settings.get("model"), settings.get("api_key")
    if not isinstance(model, str) or not model.strip() or not isinstance(key, str) or not key.strip():
        raise ValueError("Set a model and API key before calling the provider")
    if any(ord(char) < 32 for char in model) or not key.isascii() or any(not 33 <= ord(char) <= 126 for char in key):
        raise ValueError("Model and API key must use valid single-line values")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Model timeout must be a positive finite number")
    options = generation_options(settings, completion_tokens)
    payload = {"model": model, **options, "messages": messages}
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("Model request must contain finite JSON values") from None
    if len(body) > max_request_bytes:
        raise ValueError("Model HTTP request exceeds budget")
    request = urllib.request.Request(base_url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    request.add_unredirected_header("Authorization", "Bearer " + key)
    started = time.monotonic()
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=min(90, timeout)) as response:
            raw = response.read(100001)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        detail = " (rate limited)" if status == 429 else " (redirect refused)" if 300 <= status < 400 else ""
        raise ModelRequestError(f"Model HTTP {status}{detail}; no automatic retry", http_status=status) from None
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
        raise ModelRequestError("Model connection failed or timed out; no automatic retry") from None
    if len(raw) > 100000:
        raise ValueError("Model response exceeds 100 KB")
    result = json_object(raw)
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Model response must contain one Chat Completions choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise ValueError(f"Model response exceeded the {completion_tokens}-token completion budget; no retry")
    if finish_reason is not None and finish_reason != "stop":
        raise ValueError("Model did not finish a text response; no retry")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Model response is missing a text message")
    if message.get("refusal") or message.get("tool_calls") or message.get("function_call"):
        raise ValueError("Model refused or requested tools instead of returning JSON")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned no text JSON; check model support and completion budget")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content.strip(), flags=re.DOTALL)
    parsed = json_object(fenced.group(1) if fenced else content)
    return parsed, {"provider": "openai_compatible", "model": model, **_usage(result.get("usage")),
                    "request_bytes": len(body), "seconds": round(time.monotonic() - started, 3),
                    "generation_options": options}
