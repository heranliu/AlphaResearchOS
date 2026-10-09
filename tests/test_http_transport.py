"""Real loopback requests exercise deadlines without calling paid providers."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from alpharesearchos import http_transport, jev_client, provider_transport


@pytest.fixture
def worker_processes(monkeypatch):
    processes = []
    popen = subprocess.Popen

    def record(args, **kwargs):
        assert "LOCAL_MOCK_SECRET" not in " ".join(args)
        assert args[:3] == [sys.executable, "-I", "-m"]
        assert kwargs["stderr"] == subprocess.DEVNULL
        process = popen(args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(http_transport.subprocess, "Popen", record)
    yield processes
    for process in processes:
        assert process.poll() is not None
        assert process.stdin.closed and process.stdout.closed


@pytest.fixture
def local_server():
    servers = []

    def create(mode="normal", body=b'{"ok":true}'):
        state = SimpleNamespace(requests=[], finished=threading.Event(), closed=threading.Event(),
                                stop=threading.Event(), body=body)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                state.requests.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    if mode == "redirect":
                        self.send_response(302)
                        self.send_header("Location", state.url + "/redirected")
                        self.end_headers()
                        return
                    if mode == "headers":
                        self.wfile.write(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                    else:
                        self.send_response(200)
                        if mode == "chunked":
                            self.send_header("Transfer-Encoding", "chunked")
                        else:
                            self.send_header("Content-Length", str(len(body) + (60 if mode == "body" else 0)))
                        self.end_headers()
                    if mode in {"headers", "body", "chunked"}:
                        # Every arrival is far inside the socket timeout, but
                        # the complete response takes three seconds. An idle
                        # timeout therefore cannot enforce a one-second budget.
                        for _ in range(60):
                            self.wfile.write(b"1\r\n \r\n" if mode == "chunked" else b" ")
                            self.wfile.flush()
                            if state.stop.wait(.05):
                                return
                    if mode == "headers":
                        self.wfile.write(b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n")
                    if mode == "chunked":
                        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n")
                    else:
                        self.wfile.write(body)
                    self.wfile.flush()
                    state.finished.set()
                except (BrokenPipeError, ConnectionResetError):
                    state.closed.set()

            do_GET = do_POST

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        state.url = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread, state))
        return state

    yield create
    for server, thread, state in servers:
        state.stop.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(url):
    value = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
    value.add_unredirected_header("Authorization", "Bearer LOCAL_MOCK_SECRET")
    return value


@pytest.mark.parametrize("size", [0, 63, 64, 65, 10000])
def test_success_and_bounded_response(local_server, worker_processes, size):
    server = local_server(body=b"x" * size)
    result = http_transport.bounded_read(request(server.url), timeout=5, max_bytes=64)
    assert result == b"x" * min(size, 65)
    assert server.requests == ["/"] and len(worker_processes) == 1


def test_redirect_never_reaches_second_destination(local_server, worker_processes):
    server = local_server(mode="redirect")
    with pytest.raises(urllib.error.HTTPError) as caught:
        http_transport.bounded_read(request(server.url), timeout=5, max_bytes=64)
    assert caught.value.code == 302
    assert "LOCAL_MOCK_SECRET" not in str(caught.value)
    assert server.requests == ["/"] and len(worker_processes) == 1


def provider_reply(provider):
    if provider == "openai":
        return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}).encode()
    return json.dumps({
        "model": "jev-1.13.0", "answers": {
            "decision": {"type": "choice", "choice": "approve", "confidence": .9,
                         "probabilities": {"approve": .9, "revise": .05, "reject": .05}},
            **{name: {"type": "noul", "noul": .9} for name in jev_client.CHECKS},
        }, "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode()


def call_provider(provider, url, *, timeout):
    if provider == "openai":
        return provider_transport.chat_json(
            {"base_url": url, "model": "mock", "api_key": "LOCAL_MOCK_SECRET"},
            [], completion_tokens=10, timeout=timeout,
        )
    return jev_client.review_gate(
        {"candidate": {}}, settings={"jev_base_url": url, "jev_api_key": "LOCAL_MOCK_SECRET"},
        timeout=timeout,
    )


@pytest.mark.parametrize("provider", ["openai", "jev"])
def test_provider_accepts_complete_response(local_server, worker_processes, provider):
    server = local_server(body=provider_reply(provider))
    value, usage = call_provider(provider, server.url, timeout=5)
    assert value == {"ok": True} if provider == "openai" else value["decision"] == "approve"
    assert len(server.requests) == len(worker_processes) == 1
    assert "LOCAL_MOCK_SECRET" not in json.dumps(usage)


@pytest.mark.parametrize("mode", ["headers", "body", "chunked"])
@pytest.mark.parametrize("provider", ["openai", "jev"])
def test_slow_responses_stop_before_completion_and_close_connection(
    local_server, worker_processes, provider, mode,
):
    server = local_server(mode=mode, body=provider_reply(provider))
    with pytest.raises(RuntimeError, match="timed out|time budget") as caught:
        call_provider(provider, server.url, timeout=1)
    # State assertions distinguish prompt cancellation from rejecting a late
    # result, without relying on a tightly scheduled elapsed-time threshold.
    assert not server.finished.is_set()
    assert server.closed.wait(3), "timed-out worker must close its HTTP connection"
    assert len(server.requests) == len(worker_processes) == 1
    assert "LOCAL_MOCK_SECRET" not in str(caught.value)


@pytest.mark.parametrize("operation", ["getaddrinfo", "create_connection"])
def test_deadline_also_stops_dns_and_connection_setup(monkeypatch, local_server, tmp_path, operation):
    server = local_server()
    marker = tmp_path / "started"
    popen = subprocess.Popen
    processes = []
    # Delay the actual worker's setup operation. Parent thread patches cannot
    # cross this process boundary; no provider or external resolver is used.
    bootstrap = (
        "import socket,sys,time; from pathlib import Path; "
        "from alpharesearchos.http_transport import _worker; "
        f"socket.{operation}=lambda *a,**k: (Path({str(marker)!r}).touch(), time.sleep(30))[1]; "
        "sys.stdout.buffer.write(_worker())"
    )

    def delayed_worker(args, **kwargs):
        process = popen([sys.executable, "-I", "-c", bootstrap], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(http_transport.subprocess, "Popen", delayed_worker)
    with pytest.raises(TimeoutError, match="time budget"):
        http_transport.bounded_read(request(server.url), timeout=1, max_bytes=64)
    assert marker.exists() and not server.requests
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_interrupt_terminates_and_reaps_worker(monkeypatch, local_server):
    server = local_server()
    popen = subprocess.Popen
    processes = []

    def interrupted_worker(args, **kwargs):
        process = popen(args, **kwargs)
        communicate = process.communicate
        processes.append(process)

        def interrupted(data=None, timeout=None):
            if timeout is not None:
                raise KeyboardInterrupt
            return communicate(data)

        process.communicate = interrupted
        return process

    monkeypatch.setattr(http_transport.subprocess, "Popen", interrupted_worker)
    with pytest.raises(KeyboardInterrupt):
        http_transport.bounded_read(request(server.url), timeout=5, max_bytes=64)
    assert not server.requests
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


def test_test_suite_rejects_external_worker_before_spawning(monkeypatch):
    monkeypatch.setattr(http_transport.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(AssertionError, match="only loopback"):
        http_transport.bounded_read(request("https://provider.invalid/v1"), timeout=5, max_bytes=64)
