"""Opt-in smoke test of the actual wheel, isolated from the source checkout.

Run `uv build`, then `ALPHAOS_TEST_WHEEL=1 uv run pytest -q
tests/test_distribution.py`. Dependencies are resolved from uv's existing cache;
the test never contacts package indexes or model providers.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.environ.get("ALPHAOS_TEST_WHEEL") != "1", reason="Run after uv build with ALPHAOS_TEST_WHEEL=1")
def test_wheel_installs_and_serves_outside_the_source_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheels = list((root / "dist").glob("alpharesearchos-*.whl"))
    assert len(wheels) == 1, "Build exactly one release wheel before testing"
    uv = shutil.which("uv")
    assert uv, "uv is required for the isolated installation smoke test"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("ALPHAOS_", "PYTHON", "OPENAI_", "TYPESAFE_"))
                   and key not in {"VIRTUAL_ENV", "CONDA_PREFIX"}}

    def run(args, **overrides):
        result = subprocess.run(args, cwd=tmp_path, env={**environment, **overrides},
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    import sys
    target = tmp_path / "installed"
    run([uv, "venv", "--python", sys.executable, str(target)])
    python = target / "bin" / "python"
    # A locked sync caches wheel URLs, but need not cache registry index pages.
    # Reuse those exact locked dependencies rather than resolving broad wheel
    # constraints against an index which a fresh CI runner has never fetched.
    run([uv, "sync", "--locked", "--offline", "--no-install-project", "--no-default-groups",
         "--project", str(root)], UV_PROJECT_ENVIRONMENT=str(target))
    run([uv, "pip", "install", "--offline", "--no-deps", "--python", str(python), str(wheels[0])])
    run([uv, "pip", "check", "--python", str(python)])
    assert "serve" in run([str(target / "bin" / "alphaos"), "--help"])
    assert "serve" in run([str(python), "-I", "-m", "alpharesearchos", "--help"])
    output = run([str(python), "-I", "-c", '''
import importlib.metadata
import json
import pathlib
import threading
import urllib.request
import alpharesearchos
from alpharesearchos import codex_provider, server
assert "site-packages" in alpharesearchos.__file__
assert alpharesearchos.__version__ == importlib.metadata.version("alpharesearchos")
codex_provider.codex_status = lambda **kwargs: {"available": False, "authenticated": False}
service = server.make_server(pathlib.Path("runs"), pathlib.Path("datasets"), 0)
thread = threading.Thread(target=service.serve_forever, daemon=True)
thread.start()
try:
    base = f"http://127.0.0.1:{service.server_port}"
    for path, fragment in [("/", b"theme-toggle"), ("/static/app.js", b"initialize"),
                           ("/static/style.css", b"light"), ("/static/theme.js", b"localStorage")]:
        with urllib.request.urlopen(base + path, timeout=5) as response:
            assert response.status == 200 and fragment in response.read(), path
    with urllib.request.urlopen(base + "/api/datasets", timeout=5) as response:
        assert json.load(response) == {"datasets": []}
finally:
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    thread.join(timeout=5)
print("installed-wheel-ok")
'''])
    assert "installed-wheel-ok" in output
