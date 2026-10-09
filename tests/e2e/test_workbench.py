"""Real Chromium + workspace + local model HTTP fixture; no live providers.

Install with `uv sync --locked --extra dev --extra e2e`, then
`uv run playwright install chromium` and `uv run pytest -q tests/e2e`.
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from alpharesearchos import codex_provider, engine, server  # noqa: E402
from alpharesearchos.data import make_demo, save_panel  # noqa: E402

expect = playwright.expect


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as runtime:
        # An explicit override supports environments with Chromium already installed.
        instance = runtime.chromium.launch(executable_path=os.environ.get("ALPHAOS_CHROMIUM_EXECUTABLE_PATH"))
        yield instance
        instance.close()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("ALPHAOS_", "OPENAI_", "TYPESAFE_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(codex_provider, "codex_status", lambda **kwargs: {
        "available": False, "authenticated": False, "version": "", "default_model": "",
    })
    # Other parallel checks may edit code. Recovery identity has unit coverage.
    monkeypatch.setattr(engine, "code_fingerprint", lambda: "browser-fixture-build")
    calls = []
    entered, release = threading.Event(), threading.Event()
    control = {"block_next": False}

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body)
            assert self.path == "/v1/chat/completions"
            assert self.headers["Authorization"] == "Bearer browser-fixture-key"
            if control["block_next"]:
                control["block_next"] = False
                entered.set()
                assert release.wait(timeout=30), "Browser did not release the paused request"
            review = '"decision"' in body["messages"][0]["content"]
            value = ({"decision": "approve", "critique": "Development evidence is internally consistent.",
                      "risks": ["Selection uncertainty"], "suggested_action": "Compare a frozen holdout."}
                     if review else {"name": "Browser fixture momentum", "hypothesis": "Test historical momentum",
                                     "features": ["ret(close, 20)"], "model": "rank",
                                     "model_params": {"alpha": 1.0, "train_window": 126, "retrain_every": 63,
                                                      "horizon": 1, "smoothing": 2},
                                     "rationale": "Deterministic browser test only"})
            raw = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                              "usage": {"total_tokens": 37}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    provider = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    provider_thread = threading.Thread(target=provider.serve_forever, daemon=True)
    provider_thread.start()
    service = server.make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    source = tmp_path / "fixture.csv"
    save_panel(make_demo(seed=17, sessions=500, assets=3), source)
    yield SimpleNamespace(url=f"http://127.0.0.1:{service.server_port}", root=tmp_path, source=source,
                          provider=f"http://127.0.0.1:{provider.server_port}/v1", calls=calls,
                          control=control, entered=entered, release=release)
    release.set()
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    provider.shutdown()
    provider.server_close()
    thread.join(timeout=5)
    provider_thread.join(timeout=5)


@pytest.fixture
def page(browser, workspace):
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, color_scheme="dark")
    view = context.new_page()
    view.set_default_timeout(12000)
    errors = []
    view.on("pageerror", lambda error: errors.append(str(error)))
    view.goto(workspace.url)
    expect(view.locator("#connection-state")).not_to_contain_text("连接本地研究引擎…")
    yield view
    context.close()
    assert not errors, errors


def navigate(page, name):
    page.locator(f'.app-navigation [data-page="{name}"]').click()


def inspect_file(page, source):
    page.locator("#dataset-file").set_input_files(source)
    expect(page.locator("#dataset-preview")).to_be_visible()


def connect_model(page, workspace):
    navigate(page, "settings")
    page.locator("#settings-url").fill(workspace.provider)
    page.locator("#settings-model").fill("browser-fixture")
    page.locator("#settings-key").fill("browser-fixture-key")
    page.locator("#settings-save").click()
    expect(page.locator("#settings-key-status")).to_contain_text("已保存")
    page.locator("#settings-test").click()
    expect(page.locator("#settings-message")).to_contain_text("结构化", timeout=20000)
    assert len(workspace.calls) == 1


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("width", [390, 1440])
def test_theme_system_preference_toggle_and_reload(browser, workspace, scheme, width):
    context = browser.new_context(color_scheme=scheme, viewport={"width": width, "height": 900})
    page = context.new_page()
    page.goto(workspace.url)
    expect(page.locator("html")).to_have_attribute("data-theme", scheme)
    other = "dark" if scheme == "light" else "light"
    page.locator("#theme-toggle").focus()
    page.keyboard.press("Enter")
    expect(page.locator("html")).to_have_attribute("data-theme", other)
    page.reload()
    expect(page.locator("html")).to_have_attribute("data-theme", other)
    page.emulate_media(color_scheme=scheme)
    expect(page.locator("html")).to_have_attribute("data-theme", other)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    context.close()


def test_dataset_preview_cancel_and_validation(page, workspace):
    inspect_file(page, workspace.source)
    expect(page.locator("#dataset-confirm-import")).to_be_enabled()
    expect(page.locator("#dataset-preview-summary")).to_contain_text("500")
    assert list((workspace.root / "datasets").iterdir()) == []
    page.locator("#dataset-cancel-import").click()
    expect(page.locator("#dataset-preview")).not_to_be_visible()
    assert list((workspace.root / "datasets").iterdir()) == []
    original = workspace.source.read_text()
    bad = workspace.root / "duplicate.csv"
    bad.write_text(original + original.splitlines()[1] + "\n")
    inspect_file(page, bad)
    expect(page.locator("#dataset-confirm-import")).to_be_disabled()
    expect(page.locator("#dataset-preview-issues")).to_contain_text(re.compile("重复|只能有一行"))
    page.keyboard.press("Escape")
    assert list((workspace.root / "datasets").iterdir()) == []


def test_preset_switch_does_not_reuse_saved_credentials(page, workspace):
    connect_model(page, workspace)
    page.locator("#settings-api-preset").select_option("deepseek")
    expect(page.locator("#settings-url")).to_have_value("https://api.deepseek.com/v1")
    expect(page.locator("#settings-key")).to_have_value("")
    page.locator("#settings-model").fill("user-selected-model")
    page.locator("#settings-save").click()
    expect(page.locator("#settings-key-status")).to_contain_text("未设置")
    saved = page.request.get(workspace.url + "/api/settings").json()
    assert saved["api_key_set"] is False
    assert saved["configured"] is False
    assert len(workspace.calls) == 1
    page.locator("#settings-api-preset").select_option("gemini")
    expect(page.locator("#settings-url")).to_have_value("https://generativelanguage.googleapis.com/v1beta/openai")
    expect(page.locator("#settings-token-field")).to_have_value("max_tokens")
    page.locator("#settings-api-preset").select_option("ollama")
    expect(page.locator("#settings-url")).to_have_value("http://localhost:11434/v1")
    expect(page.locator("#settings-key")).to_have_value("ollama")


@pytest.mark.parametrize("provider", ["model", "jev"])
def test_late_connection_result_does_not_validate_newly_saved_settings(page, workspace, provider):
    connect_model(page, workspace)
    prefix = "settings" if provider == "model" else "settings-jev"
    if provider == "jev":
        page.locator("#settings-advanced-review summary").click()
        page.locator("#settings-jev-enabled").check()
        page.locator("#settings-jev-url").fill(workspace.provider)
        page.locator("#settings-jev-key").fill("browser-fixture-key")
        page.locator("#settings-save").click()
        expect(page.locator("#settings-jev-test")).to_be_enabled()
        page.locator(".jev-connection summary").click()
    pending = []
    endpoint = "/api/settings/test" if provider == "model" else "/api/settings/jev/test"
    page.route("**" + endpoint, lambda route: pending.append(route))
    page.locator(f"#{prefix}-test").click()
    expect(page.locator(f"#{prefix}-test")).to_have_text("正在测试…")
    page.locator(f"#{prefix}-url").fill(workspace.provider + "/changed")
    page.locator(f"#{prefix}-key").fill("browser-fixture-key")
    page.locator("#settings-save").click()
    expect(page.locator("#settings-message")).to_have_text("配置已保存。")
    expect(page.locator(f"#{prefix}-test")).to_be_disabled()
    assert len(pending) == 1
    pending[0].fulfill(json={"ok": True, "message": "old endpoint result"})
    expect(page.locator(f"#{prefix}-test")).to_be_enabled()
    expect(page.locator("#settings-status")).to_have_text("已配置")
    expect(page.locator("#settings-message")).to_have_text("配置已保存。")
    expect(page.locator("#settings-jev-message")).to_be_hidden()
    expect(page.locator(f"#{prefix}-message")).not_to_contain_text("old endpoint result")
    assert len(workspace.calls) == 1


def test_out_of_order_backtest_reads_keep_completed_report_and_configuration(page):
    pending = []
    completed = {"id": "browser-race", "status": "completed", "config": {"cost_bps": 12}}

    def report(route):
        pending.append(route)
        if len(pending) > 1:
            route.fulfill(json=completed)

    page.route("**/api/backtests/browser-race", report)
    navigate(page, "backtest")
    page.evaluate("() => {window.backtestReads = [loadBacktest('browser-race'), loadBacktest('browser-race')];}")
    expect(page.locator("#backtest-status")).to_have_text("已完成")
    expect(page.locator("#backtest-cost")).to_have_value("12")
    pending[0].fulfill(json={"id": "browser-race", "status": "running"})
    page.evaluate("Promise.all(window.backtestReads)")
    expect(page.locator("#backtest-status")).to_have_text("已完成")
    expect(page.locator('#backtest-artifacts a[download="report.json"]')).to_be_visible()


def test_research_pause_resume_export_library_and_backtest(page, workspace):
    connect_model(page, workspace)
    navigate(page, "research")
    inspect_file(page, workspace.source)
    page.locator("#dataset-confirm-import").click()
    expect(page.locator("#dataset-import-message")).to_contain_text("已导入")
    expect(page.locator("#dataset")).to_have_value("fixture.csv")
    page.locator("#trials").fill("1")
    workspace.control["block_next"] = True
    page.locator("#start-run").click()
    assert workspace.entered.wait(timeout=15)
    expect(page.locator("#pause-run")).to_be_visible()
    page.locator("#pause-run").click()
    expect(page.locator("#pause-run")).to_be_disabled()
    workspace.release.set()
    expect(page.locator("#run-status")).to_have_text("已暂停", timeout=30000)
    page.reload()
    page.locator("#run-list button").first.click()
    expect(page.locator("#resume-run")).to_be_visible()
    page.locator("#resume-run").click()
    expect(page.locator("#run-status")).to_have_text("已完成", timeout=30000)
    expect(page.locator("#run-warnings")).not_to_contain_text("已暂停")
    expect(page.locator("#equity-chart svg")).to_be_visible()
    expect(page.locator("#trials-body")).to_contain_text("Browser fixture momentum")
    assert len(workspace.calls) == 3  # probe, proposal, independent review; resume never retries
    with page.expect_download() as download:
        page.locator('#artifact-links a[download="report.json"]').click()
    report = json.loads(Path(download.value.path()).read_text())
    assert report["status"] == "completed" and report["holdout"]["curve"]
    assert report["stop_reason"] not in {"user_pause", "checkpoint"}
    assert "browser-fixture-key" not in json.dumps(report)
    navigate(page, "library")
    expect(page.locator('#library-body input[type="checkbox"]').first).to_be_visible()
    page.locator('#library-body input[type="checkbox"]').first.check()
    page.locator("#factor-to-backtest").click()
    expect(page.locator("#backtest-dataset")).to_have_value("fixture.csv")
    page.locator("#start-backtest").click()
    expect(page.locator("#backtest-status")).to_have_text("已完成", timeout=30000)
    expect(page.locator("#backtest-chart svg")).to_be_visible()
    expect(page.locator("#backtest-drawdown-chart svg")).to_be_visible()
    assert len(workspace.calls) == 3
