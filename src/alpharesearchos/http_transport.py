"""One HTTP request with a deadline covering DNS, connection, headers and body.

urllib's socket timeout only bounds an idle read. A short-lived worker process
lets the caller stop even DNS or a continuously trickling response, then reap
the worker before returning. Credentials travel through stdin, never argv.
"""
from __future__ import annotations

import base64
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request


def bounded_read(request, *, timeout, max_bytes):
    """Read at most max_bytes + 1 bytes, without retries or surviving workers."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("HTTP timeout must be a positive finite number")
    deadline = time.monotonic() + timeout
    payload = json.dumps({
        "url": request.full_url, "method": request.get_method(),
        "headers": request.headers, "unredirected_headers": request.unredirected_hdrs,
        "data": base64.b64encode(request.data or b"").decode("ascii"),
        "deadline": deadline, "max_bytes": max_bytes,
    }).encode("utf-8")
    # -I ignores user Python paths while retaining the installed package and
    # interpreter's site-packages. Proxy and CA environment settings are inherited.
    with subprocess.Popen(
        [sys.executable, "-I", "-m", "alpharesearchos.http_transport"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ) as process:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP request exceeded its time budget")
            output, _ = process.communicate(payload, timeout=remaining)
            if time.monotonic() >= deadline:
                raise TimeoutError("HTTP request exceeded its time budget")
        except subprocess.TimeoutExpired:
            raise TimeoutError("HTTP request exceeded its time budget") from None
        finally:
            # Also clean up on KeyboardInterrupt and unexpected exceptions.
            # No abandoned thread can send credentials after the caller returns.
            if process.poll() is None:
                process.kill()
            process.communicate()
        if process.returncode:
            raise OSError("HTTP worker failed")
    kind, separator, value = output.partition(b"\n")
    if separator and kind == b"OK" and len(value) <= max_bytes + 1:
        return value
    if separator and kind == b"HTTP" and value.isdigit() and 100 <= int(value) <= 599:
        # Only the status crosses the error boundary, never URLs or headers.
        raise urllib.error.HTTPError("", int(value), "HTTP request failed", {}, None)
    raise OSError("HTTP connection failed")


def _worker():
    from .model_settings import _NoRedirect

    try:
        payload = json.load(sys.stdin)
        remaining = payload["deadline"] - time.monotonic()
        if remaining <= 0:
            return b"ERROR\n"
        request = urllib.request.Request(
            payload["url"], data=base64.b64decode(payload["data"]),
            headers=payload["headers"], method=payload["method"],
        )
        for key, value in payload["unredirected_headers"].items():
            request.add_unredirected_header(key, value)
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=remaining) as response:
            return b"OK\n" + response.read(payload["max_bytes"] + 1)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        return b"HTTP\n" + str(status).encode("ascii")
    except Exception:
        return b"ERROR\n"


if __name__ == "__main__":
    sys.stdout.buffer.write(_worker())
