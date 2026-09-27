"""Integrated web workflows: persistence, job isolation, and safe execution controls."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from alpharesearchos import engine
from alpharesearchos import server as workspace
from alpharesearchos.config import ResearchConfig
from alpharesearchos.data import make_demo, save_panel
from alpharesearchos.store import read_json


@pytest.fixture
def web(tmp_path):
    save_panel(make_demo(sessions=500, assets=3), tmp_path / "datasets" / "fixture.csv")
    service = workspace.make_server(tmp_path / 'runs', tmp_path / 'datasets', 0)
    thread = threading.Thread(target=service.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{service.server_port}', tmp_path
    service.shutdown()
    service.server_close()
    service.research_executor.shutdown(wait=True)
    thread.join(timeout=5)


def call(web, path, body=None, headers=None):
    request = urllib.request.Request(web[0] + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json', **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def test_library_import_roundtrip_and_atomic_rejection(web):
    assert call(web, '/api/factors') == (200, {'factors': []})
    code, result = call(web, '/api/factors/import', {'factors': [
        {'name': '趋势', 'expression': 'rank(ret(close,20))', 'hypothesis': '趋势延续'}]})
    assert code == 201 and result['imported'] == 1
    original = call(web, '/api/factors')[1]['factors']
    assert len(original) == 1
    code, _ = call(web, '/api/factors/import', {'factors': [
        {'name': 'valid', 'expression': 'rank(ret(close,10))'},
        {'name': 'unsafe', 'expression': '__import__("os").system("touch unexpected")'}]})
    assert code == 400
    assert call(web, '/api/factors')[1]['factors'] == original
    assert not (web[1] / 'unexpected').exists()
    assert call(web, '/api/factors/import', {'factors': []}, {'Origin': 'https://evil.example'})[0] == 403


def test_settings_persist_without_leaking_key(web, monkeypatch):
    for name in ['OPENAI_API_KEY', 'ALPHAOS_LLM_API_KEY', 'ALPHAOS_LLM_MODEL']:
        monkeypatch.delenv(name, raising=False)
    config = {'base_url': 'http://127.0.0.1:9999/v1', 'model': 'local-test', 'api_key': 'private-test-key'}
    code, saved = call(web, '/api/settings', config)
    assert code == 200 and saved['configured'] and saved['api_key_set']
    assert 'private-test-key' not in json.dumps(saved)
    assert 'api_key' not in call(web, '/api/settings')[1]
    assert call(web, '/api/health')[1]['llm_configured']
    assert call(web, '/api/settings', {'api_key': ''})[1]['api_key_set']
    assert call(web, '/api/settings', {'clear_api_key': True})[1]['api_key_set'] is False
    settings_path = web[1] / '.alphaos/settings.json'
    assert settings_path.stat().st_mode & 0o777 == 0o600
    assert call(web, '/api/settings', config, {'Origin': 'https://evil.example'})[0] == 403
    assert call(web, '/artifacts/../../.alphaos/settings.json')[0] in (400, 404)


def test_jev_settings_and_explicit_probe_are_isolated(web, monkeypatch):
    from alpharesearchos import jev_client
    calls = []
    def gate(context, *, settings, timeout):
        calls.append(context)
        assert settings['jev_api_key'] == 'PRIVATE_JEV_KEY'
        return {'decision': 'revise'}, {'model': 'jev-1.13.0'}
    monkeypatch.setattr(jev_client, 'review_gate', gate)
    code, saved = call(web, '/api/settings', {'jev_api_key': 'PRIVATE_JEV_KEY'})
    assert code == 200 and saved['jev_configured'] and not saved['jev_enabled']
    assert 'PRIVATE_JEV_KEY' not in json.dumps(saved) and 'jev_api_key' not in saved
    assert calls == []
    assert call(web, '/api/settings/jev/test', {'jev_api_key': 'new'})[0] == 400
    assert call(web, '/api/settings/jev/test', {}, {'Origin': 'https://evil.example'})[0] == 403
    assert calls == []
    code, result = call(web, '/api/settings/jev/test', {})
    assert code == 200 and result['ok'] and len(calls) == 1
    assert 'PRIVATE_JEV_KEY' not in json.dumps(result)


def test_enabled_jev_requires_key_before_creating_agent_run(web):
    call(web, '/api/settings', {'base_url': 'https://provider.example/v1', 'model': 'mock',
                              'api_key': 'NOT_USED', 'jev_enabled': True, 'clear_jev_api_key': True})
    code, result = call(web, '/api/runs', {'mode': 'agent', 'dataset': 'fixture.csv', 'trials': 1})
    assert code == 400 and 'Jev' in result['error']
    assert not list((web[1] / 'runs').glob('*/report.json'))


def test_busy_job_pause_and_configuration_lock(web, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    observed = {}
    def controlled_worker(directory, *, should_pause):
        entered.set()
        release.wait(timeout=10)
        observed['pause'] = should_pause()
        finished.set()
        return {}
    monkeypatch.setattr(workspace, 'run_research', controlled_worker)
    try:
        code, first = call(web, '/api/runs', {'trials': 2, 'mode': 'local', 'dataset': 'fixture.csv'})
        assert code == 201
        assert entered.wait(timeout=3)
        assert call(web, '/api/runs', {'trials': 2})[0] == 409
        assert len(list((web[1] / 'runs').iterdir())) == 1
        assert call(web, '/api/settings', {'model': 'changed'})[0] == 409
        assert call(web, '/api/settings/test', {})[0] == 409
        assert call(web, '/api/settings/jev/test', {})[0] == 409
        code, paused = call(web, f"/api/runs/{first['id']}/pause", {})
        assert code == 200 and paused['status'] == 'pause_requested'
        report = call(web, f"/api/runs/{first['id']}")[1]
        assert report['active'] and report['pause_requested']
        assert '_state' not in report and report['artifacts'] == ['report.json']
    finally:
        release.set()
        assert finished.wait(timeout=4)
    assert observed['pause'] is True


def test_pause_at_trial_boundary_preserves_budget_and_resume(tmp_path):
    directory = engine.create_run(tmp_path / 'runs', ResearchConfig(trials=2))
    def after_one_trial():
        return len(read_json(directory / 'report.json')['trials']) >= 1
    report = engine.run_research(directory, should_pause=after_one_trial)
    assert report['status'] == 'paused' and report['stop_reason'] == 'user_pause'
    assert len(report['trials']) == 1 and report['_state']['pending'] is None
    assert report['selected'] is None and report['holdout'] is None
    first = report['trials'][0].copy()
    resumed = engine.run_research(directory)
    assert resumed['status'] == 'completed' and len(resumed['trials']) == 2
    assert resumed['trials'][0] == first


def test_backtest_bad_dataset_and_artifact_traversal(web):
    assert call(web, '/api/backtests') == (200, {'backtests': []})
    _, imported = call(web, '/api/factors/import', {'factors': [{'name': 'm', 'expression': 'rank(ret(close,20))'}]})
    body = {'factor_ids': [imported['factors'][0]['id']], 'dataset': '../secret.csv'}
    assert call(web, '/api/backtests', body)[0] == 400
    for path in ['/api/runs/not-an-id', '/api/backtests/../settings.json', '/artifacts/backtests/../../settings.json']:
        assert call(web, path)[0] in (400, 404)


def test_orphaned_checkpoint_is_exposed_as_interrupted_without_rewriting_it(web):
    directory = engine.create_run(web[1] / 'runs', ResearchConfig(trials=1))
    record = call(web, f'/api/runs/{directory.name}')[1]
    assert record['status'] == 'interrupted' and record['active'] is False
    assert call(web, '/api/runs')[1]['runs'][0]['status'] == 'interrupted'
    assert read_json(directory / 'report.json')['status'] == 'queued'
