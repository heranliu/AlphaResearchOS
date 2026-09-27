"""Exercise the optional Jev UI without a browser, network, or model calls."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "alpharesearchos" / "static"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required for UI contracts")


def run_ui_contract(body):
    source = (STATIC / "app.js").read_text().rsplit("\ninitialize().catch", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const nodes = new Map();
class Element {
  constructor(tag = 'div') { this.tagName = tag; this.children = []; this.checked = false;
    this.disabled = false; this.hidden = false; this.dataset = {}; this.value = ''; this._text = ''; }
  set value(value) { this._value = String(value); }
  get value() { return this._value; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(node => node.textContent).join(' '); }
  set innerHTML(value) { throw new Error('Untrusted report content must never use innerHTML'); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ''; this.children = children; }
  setCustomValidity(message) { this.validationMessage = message; }
  querySelectorAll() { return this.children; }
}
const document = {getElementById(id) {
  if (!nodes.has(id)) nodes.set(id, new Element());
  return nodes.get(id);
}, createElement: tag => new Element(tag)};
class FormData {
  get(name) { const ids = {provider: 'settings-provider', codex_model: 'settings-codex-model',
    base_url: 'settings-url', model: 'settings-model', api_key: 'settings-key',
    token_field: 'settings-token-field', temperature: 'settings-temperature'};
    return document.getElementById(ids[name]).value; }
}
"""
    setup = r"""
const calls = [];
api = async (path, options) => { calls.push({path, options}); return {ok: true, latency_ms: 2}; };
$('settings-jev-fields').children = ['url', 'model', 'key', 'clear-key', 'confidence'].map(key => $('settings-jev-' + key));
$('settings-api-fields').children = ['url', 'model', 'key', 'token-field', 'temperature'].map(key => $('settings-' + key));
$('dataset').value = 'fixture.csv'; $('max-llm-calls').value = 12; $('trials').value = 6;
state.datasets = [{id: 'fixture.csv', rows: 500}];
const configured = {configured: true, provider: 'openai_compatible', base_url: 'https://llm.example/v1', model: 'proposal-model', api_key_set: true};
"""
    result = subprocess.run(
        [NODE, "-e", harness + source + "\n(async () => {\n" + setup + body
         + "\n})().catch(error => { console.error(error); process.exitCode = 1; });"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


def test_jev_is_optional_and_missing_enabled_credentials_block_only_research():
    run_ui_contract(r"""
applySettings(configured);
assert.equal($('settings-jev-enabled').checked, false);
assert.equal($('settings-jev-fields').hidden, true);
assert.equal($('start-run').disabled, false);
assert.equal($('settings-test').disabled, false);
assert.equal($('settings-jev-test').disabled, true);
assert.equal($('settings-jev-url').value, 'https://api.typesafe.ai/v1');
assert.equal($('settings-jev-model').value, 'jev-1.13.0');
applySettings({...configured, jev_enabled: true, jev_configured: false});
assert.equal($('start-run').disabled, true);
assert.equal($('settings-test').disabled, false);
assert.match($('mode-note').textContent, /Jev.*配置不完整/);
applySettings({...configured, jev_enabled: true, jev_configured: true});
assert.equal($('start-run').disabled, false);
assert.match($('mode-note').textContent, /4 个完整候选/);
$('settings-jev-confidence').value = 0.5; renderJevFields();
assert.match($('settings-jev-confidence').validationMessage, /大于 0.5/);
assert.equal(calls.length, 0);
""")


def test_jev_and_original_model_tests_are_separate_explicit_saved_requests():
    run_ui_contract(r"""
applySettings({...configured, jev_enabled: true, jev_configured: true});
assert.equal(calls.length, 0);
state.settingsDirty = true; renderProviderFields(); renderJevFields();
assert.equal($('settings-test').disabled, true);
assert.equal($('settings-jev-test').disabled, true);
await testSettings(); await testJevSettings();
assert.equal(calls.length, 0);
state.settingsDirty = false;
await testSettings();
assert.deepEqual(calls.map(call => call.path), ['/api/settings/test']);
await testJevSettings();
assert.deepEqual(calls.map(call => call.path), ['/api/settings/test', '/api/settings/jev/test']);
assert.equal(calls[1].options.method, 'POST');
assert.equal(calls[1].options.body, '{}');
api = async () => { state.settingsDirty = true; throw new Error('mock provider unavailable'); };
await testJevSettings();
assert.equal($('settings-jev-test').disabled, true);
assert.match($('settings-jev-message').textContent, /mock provider unavailable/);
""")


def test_unified_save_keeps_jev_independent_of_codex_and_never_refills_secrets():
    run_ui_contract(r"""
applySettings({...configured, provider: 'codex_cli', jev_enabled: true, jev_configured: true, jev_api_key_set: true});
$('settings-jev-key').value = 'mock-new-key';
$('settings-jev-confidence').value = 0.83;
api = async (path, options) => {
  calls.push({path, payload: JSON.parse(options.body)});
  return {...configured, provider: 'codex_cli', jev_enabled: true, jev_configured: true, jev_api_key_set: true, jev_min_confidence: 0.83};
};
await saveSettings({preventDefault() {}});
assert.equal(calls.length, 1); assert.equal(calls[0].path, '/api/settings');
assert.equal(calls[0].payload.jev_api_key, 'mock-new-key');
assert.equal(calls[0].payload.jev_enabled, true);
assert.equal(calls[0].payload.jev_min_confidence, 0.83);
assert.equal(calls[0].payload.provider, 'codex_cli');
assert.equal(calls[0].payload.api_key, undefined);
assert.equal($('settings-jev-key').value, '');
await saveSettings({preventDefault() {}});
assert.equal(calls[1].payload.jev_api_key, '');
assert.equal(calls[1].payload.clear_jev_api_key, false);
$('settings-jev-clear-key').checked = true; renderJevFields();
assert.equal($('settings-jev-key').disabled, true);
await saveSettings({preventDefault() {}});
assert.equal(calls[2].payload.clear_jev_api_key, true);
assert.equal(calls[2].payload.jev_api_key, '');
""")


def test_gate_displays_probabilities_threshold_checks_and_untrusted_summary_as_text():
    summary = '<img src=x onerror="globalThis.compromised=true">'
    run_ui_contract(r"""
const container = new Element();
renderJevGate(container, {decision: 'revise', confidence: 0.82, threshold: 0.7,
  probabilities: {approve: 0.12, revise: 0.82, reject: 0.06},
  checks: {hypothesis_alignment: 0.95, cost_support: 0.68, fold_consistency: 0.79, evidence_sufficiency: 0.9},
  summary: """ + json.dumps(summary) + r"""});
assert.match(container.textContent, /需修改/);
assert.match(container.textContent, /置信度：82.0%/);
assert.match(container.textContent, /通过阈值：70.0%/);
assert.match(container.textContent, /通过概率：12.0%/);
assert.match(container.textContent, /需修改概率：82.0%/);
assert.match(container.textContent, /拒绝概率：6.0%/);
assert.match(container.textContent, /95.0% · 达标/);
assert.match(container.textContent, /68.0% · 低于阈值/);
assert.match(container.textContent, /假设一致性.*成本后支持.*滚动窗口一致性.*证据充分性/);
assert.ok(container.textContent.includes(""" + json.dumps(summary) + r"""));
assert.equal(globalThis.compromised, undefined);
assert.equal(trialUsages({jev_usage: {model: 'jev-1.13.0'}})[0][0], 'Jev 裁决');
""")
