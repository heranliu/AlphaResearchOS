"""A live CLI lock must not become an interrupted or overwritten web checkpoint."""

import contextlib
import http.client
import json
import multiprocessing
import threading

import pytest

from alpharesearchos import codex_provider
from alpharesearchos import server as workspace
from alpharesearchos.store import atomic_json, read_json, run_lock

RUN_ID = "20260918-120000-1234abcd"


def _hold_run_lock(directory, ready, release):
    with run_lock(directory):
        (directory / ".lock").write_text("external process owns this inode", encoding="utf-8")
        ready.set()
        if not release.wait(timeout=30):
            raise RuntimeError("Test did not release its lock holder")


@contextlib.contextmanager
def external_run(directory):
    """Use a separate process so the test exercises the actual flock contract."""
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_run_lock, args=(directory, ready, release))
    process.start()
    try:
        assert ready.wait(timeout=10), "The isolated lock holder did not start"
        yield process
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        assert process.exitcode == 0


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_provider, "codex_status", lambda **kwargs: {
        "available": False, "authenticated": False, "version": None,
        "login_method": None, "default_model": None,
    })

    def forbidden(*args, **kwargs):
        pytest.fail("Lock status tests must not make an external model request")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    launched = []
    monkeypatch.setattr(workspace, "run_research", lambda directory, **kwargs: launched.append(directory))
    service = workspace.make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()

    def request(path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", service.server_port, timeout=10)
        try:
            connection.request("GET" if body is None else "POST", path,
                               body=None if body is None else json.dumps(body),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    yield request, tmp_path, launched, service
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    thread.join(timeout=5)


def checkpoint(root, kind="research", status="running"):
    directory = root / ("runs" if kind == "research" else "backtests") / RUN_ID
    atomic_json(directory / "report.json", {
        "id": RUN_ID, "status": status, "created_at": "2026-09-18T12:00:00Z",
        "kind": kind, "config": {"mode": "local", "direction": "CLI lock fixture"},
        "source": {"kind": "synthetic", "label": "isolated fixture"},
        "progress": {"completed": 0, "total": 1}, "trials": [], "selected": None,
        "error": None, "_state": {"pending": {"id": "T0001"}},
    })
    return directory


@pytest.mark.parametrize("kind", ["research", "backtest"])
def test_external_lock_remains_active_in_detail_and_listing_without_writes(web, kind):
    request, root, launched, _ = web
    directory = checkpoint(root, kind)
    collection = "runs" if kind == "research" else "backtests"
    before = (directory / "report.json").read_bytes()
    with external_run(directory) as owner:
        lock_before = (directory / ".lock").read_bytes()
        code, report = request(f"/api/{collection}/{RUN_ID}")
        listed_code, listing = request(f"/api/{collection}")
        assert code == listed_code == 200
        assert (directory / "report.json").read_bytes() == before
        assert (directory / ".lock").read_bytes() == lock_before
        assert owner.is_alive() and not launched
        for record in (report, listing[collection][0]):
            assert record["status"] == "running"
            assert record["active"] is True
            assert record["externally_active"] is True
            assert record["pause_requested"] is False
            assert not record.get("error")


@pytest.mark.parametrize("kind", ["research", "backtest"])
def test_released_lock_file_does_not_hide_a_real_interruption(web, kind):
    request, root, _, _ = web
    directory = checkpoint(root, kind)
    collection = "runs" if kind == "research" else "backtests"
    with external_run(directory):
        pass
    before = (directory / "report.json").read_bytes()
    lock_before = (directory / ".lock").read_bytes()
    code, report = request(f"/api/{collection}/{RUN_ID}")
    assert code == 200 and report["status"] == "interrupted"
    assert report["active"] is False and not report.get("externally_active", False)
    assert (directory / "report.json").read_bytes() == before
    assert (directory / ".lock").read_bytes() == lock_before


def test_status_read_does_not_create_a_missing_lock_file(web):
    request, root, _, _ = web
    directory = checkpoint(root)
    assert request(f"/api/runs/{RUN_ID}")[1]["status"] == "interrupted"
    assert not (directory / ".lock").exists()


@pytest.mark.parametrize("status", ["running", "paused"])
def test_resume_rejects_external_owner_without_dispatch_or_checkpoint_write(web, status):
    request, root, launched, service = web
    directory = checkpoint(root, status=status)
    before = (directory / "report.json").read_bytes()
    with external_run(directory) as owner:
        code, response = request(f"/api/runs/{RUN_ID}/resume", {})
        service.research_executor.submit(lambda: None).result(timeout=5)
        assert (directory / "report.json").read_bytes() == before
        assert owner.is_alive()
        assert code == 409 and response.get("error")
        assert not launched


@pytest.mark.parametrize("failure", ["lock_contention", "preflight"])
def test_lock_race_in_worker_does_not_mark_external_checkpoint_failed(web, monkeypatch, failure):
    request, root, _, service = web
    directory = checkpoint(root)
    before = (directory / "report.json").read_bytes()
    entered, proceed = threading.Event(), threading.Event()

    def contended_worker(run_directory, **kwargs):
        entered.set()
        assert proceed.wait(timeout=15)
        if failure == "preflight":
            raise ValueError("Preflight failed while a CLI resumed this run")
        with run_lock(run_directory):
            pytest.fail("An external owner already acquired this lock")

    monkeypatch.setattr(workspace, "run_research", contended_worker)
    try:
        assert request(f"/api/runs/{RUN_ID}/resume", {})[0] == 200
        assert entered.wait(timeout=5)
        with external_run(directory) as owner:
            proceed.set()
            service.research_executor.submit(lambda: None).result(timeout=5)
            assert owner.is_alive()
            assert (directory / "report.json").read_bytes() == before
            assert read_json(directory / "report.json")["status"] == "running"
    finally:
        proceed.set()


def test_contention_exception_preserves_checkpoint_after_external_owner_exits(web, monkeypatch):
    request, root, _, service = web
    directory = checkpoint(root)
    before = (directory / "report.json").read_bytes()
    entered, proceed = threading.Event(), threading.Event()
    contended, owner_released = threading.Event(), threading.Event()

    def contended_worker(run_directory, **kwargs):
        entered.set()
        assert proceed.wait(timeout=15)
        try:
            with run_lock(run_directory):
                pytest.fail("An external owner already acquired this lock")
        except RuntimeError:
            contended.set()
            assert owner_released.wait(timeout=15)
            raise

    monkeypatch.setattr(workspace, "run_research", contended_worker)
    try:
        assert request(f"/api/runs/{RUN_ID}/resume", {})[0] == 200
        assert entered.wait(timeout=5)
        with external_run(directory):
            proceed.set()
            assert contended.wait(timeout=5)
        owner_released.set()
        service.research_executor.submit(lambda: None).result(timeout=5)
        assert (directory / "report.json").read_bytes() == before
    finally:
        proceed.set()
        owner_released.set()
