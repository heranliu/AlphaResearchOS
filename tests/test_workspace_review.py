"""Independent integration regressions for credential boundaries and pausing."""
import json
import threading
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from alpharesearchos import engine
from alpharesearchos import server as workspace
from alpharesearchos.config import ResearchConfig
from alpharesearchos.data import make_demo, save_panel
from alpharesearchos.proposals import llm_proposal, provider_context
from alpharesearchos.store import read_json


@contextmanager
def local_server(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_research_request_does_not_forward_key_to_redirect_destination():
    received, sent = [], []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            payload = json.dumps({"choices": [{"message": {"content":
                '{"name":"mock","hypothesis":"test","expression":"rank(close)"}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    with local_server(Destination) as destination:
        class Redirector(BaseHTTPRequestHandler):
            def do_POST(self):
                sent.append(self.headers.get("Authorization"))
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(302)
                self.send_header("Location", destination + "/changed-provider")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        with local_server(Redirector) as origin:
            with provider_context({"base_url": origin, "model": "mock", "api_key": "dummy-review-key",
                                   "token_field": "max_completion_tokens", "temperature": None}):
                try:
                    llm_proposal("mock research", [])
                except RuntimeError:
                    pass  # A deliberately rejected redirect also satisfies this boundary.
    assert sent == ["Bearer dummy-review-key"]
    assert all(header is None for header in received), "Authorization was forwarded to the redirect destination"


def test_pause_after_final_candidate_prevents_selection_and_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_export", lambda *args: None)
    directory = engine.create_run(tmp_path, ResearchConfig(trials=1))

    def requested_after_current_candidate():
        return len(read_json(directory / "report.json")["trials"]) == 1

    report = engine.run_research(directory, should_pause=requested_after_current_candidate)
    assert report["status"] == "paused"
    assert report["stop_reason"] == "user_pause"
    assert report["selected"] is None and report["holdout"] is None
    assert len(report["trials"]) == 1


def test_async_preflight_failure_persists_and_releases_active_slot(tmp_path, monkeypatch):
    save_panel(make_demo(sessions=500, assets=3), tmp_path / "datasets" / "fixture.csv")
    def fail_preflight(*args, **kwargs):
        raise ValueError("Snapshot checksum changed; create a new experiment")

    monkeypatch.setattr(workspace, "run_research", fail_preflight)
    server = workspace.make_server(tmp_path / "runs", tmp_path / "datasets", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = urllib.request.Request(base + "/api/runs", data=b'{"trials":1,"mode":"local","dataset":"fixture.csv"}',
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            created = json.load(response)
            assert response.status == 201
        # One-worker queue barrier waits for persistence and active-slot cleanup.
        server.research_executor.submit(lambda: None).result(timeout=5)
        with urllib.request.urlopen(base + "/api/runs/" + created["id"], timeout=5) as response:
            report = json.load(response)
        assert report["status"] == "failed"
        assert report["error"] == "Snapshot checksum changed; create a new experiment"
        assert report["active"] is False
        assert "_state" not in report
        with urllib.request.urlopen(base + "/api/health", timeout=5) as response:
            assert json.load(response)["jobs"] == []
    finally:
        server.shutdown()
        server.server_close()
        server.research_executor.shutdown(wait=True)
        thread.join(timeout=3)
