"""Offline browser contracts for themes, credential isolation, and import confirmation."""

import json
import subprocess

import pytest
from test_jev_ui_contract import NODE, STATIC, run_ui_contract

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required for UI contracts")


def test_api_presets_never_carry_saved_credentials_to_another_endpoint():
    run_ui_contract(r"""
applySettings(configured);
$('settings-key').value = 'draft-old-service-key';
$('settings-api-preset').value = 'deepseek'; chooseApiPreset();
assert.equal($('settings-url').value, 'https://api.deepseek.com/v1');
assert.equal($('settings-token-field').value, 'max_tokens');
assert.equal($('settings-model').value, '');
assert.equal($('settings-key').value, '');
assert.equal($('settings-test').disabled, true);
api = async (path, options) => {
  const payload = JSON.parse(options.body); calls.push({path, payload});
  return {...configured, base_url: payload.base_url, model: payload.model, api_key_set: !!payload.api_key};
};
$('settings-model').value = 'user-selected-model';
await saveSettings({preventDefault() {}});
assert.equal(calls[0].payload.clear_api_key, true);
assert.equal(calls[0].payload.api_key, '');
assert.equal(calls[0].payload.provider, 'openai_compatible');
$('settings-api-preset').value = 'gemini'; chooseApiPreset();
$('settings-model').value = 'user-selected-gemini-model';
$('settings-key').value = 'new-gemini-key';
await saveSettings({preventDefault() {}});
assert.equal(calls[1].payload.clear_api_key, false);
assert.equal(calls[1].payload.api_key, 'new-gemini-key');
assert.equal(calls[1].payload.base_url, 'https://generativelanguage.googleapis.com/v1beta/openai');
assert.equal($('settings-key').value, '');
$('settings-api-preset').value = 'ollama'; chooseApiPreset();
assert.equal($('settings-key').value, 'ollama');
assert.equal($('settings-url').value, 'http://localhost:11434/v1');
assert.equal($('settings-model').value, '');
assert.equal(calls.length, 2); // Choosing a preset never calls a provider.
""")


def test_unchanged_endpoint_keeps_existing_secret_when_password_is_blank():
    run_ui_contract(r"""
applySettings(configured);
$('settings-url').value += '/';
api = async (path, options) => { calls.push(JSON.parse(options.body)); return configured; };
await saveSettings({preventDefault() {}});
assert.equal(calls[0].clear_api_key, false);
assert.equal(calls[0].api_key, '');
""")


DATASET_SETUP = r"""
document.querySelectorAll = () => [];
Element.prototype.showModal = function () { this.open = true; };
Element.prototype.close = function () { this.open = false; };
renderModeNote = () => {};
renderBacktestSelection = () => {};
renderDatasets = () => {};
const file = {name: 'preview.csv', size: 100, arrayBuffer: async () => new TextEncoder().encode('fixture-content').buffer};
const inspection = {name: file.name, rows: 1500, assets: 3, common_range: {start: '2020-01-01', end: '2022-01-01', sessions: 500}, importable: true, eligible_for_research: true,
  issues: [], asset_coverage: [{symbol: '<img onerror=bad>', rows: 500, sessions: 500, start: '2020-01-01', end: '2022-01-01', missing_sessions: 0}], adjustment: {message: '核对复权口径'}};
"""


def test_dataset_preview_is_read_only_until_confirmed_and_cancel_discards_content():
    run_ui_contract(DATASET_SETUP + r"""
api = async (path, options) => { calls.push({path, payload: JSON.parse(options.body)}); return {inspection}; };
await importDataset(file);
assert.deepEqual(calls.map(call => call.path), ['/api/datasets/inspect']);
assert.equal($('dataset-preview').open, true);
assert.equal($('dataset-confirm-import').disabled, false);
assert.equal(state.pendingDataset.content, 'fixture-content');
assert.ok($('dataset-preview-coverage').textContent.includes('<img onerror=bad>'));
cancelDatasetImport();
assert.equal(state.pendingDataset, null);
assert.equal($('dataset-preview').open, false);
await confirmDatasetImport();
assert.equal(calls.length, 1);
await importDataset(file);
api = async (path, options) => {
  calls.push({path, payload: JSON.parse(options.body)});
  return {dataset: {id: 'preview.csv', assets: 3, rows: 500}};
};
await confirmDatasetImport();
assert.deepEqual(calls.map(call => call.path), ['/api/datasets/inspect', '/api/datasets/inspect', '/api/datasets']);
assert.equal(calls[2].payload.content, 'fixture-content');
assert.equal(state.pendingDataset, null);
assert.equal($('dataset-preview').open, false);
assert.match($('dataset-import-message').textContent, /已导入.*preview.csv/);
""")


def test_invalid_inspection_cannot_be_confirmed_and_import_failures_preserve_preview():
    run_ui_contract(DATASET_SETUP + r"""
api = async (path, options) => { calls.push(path); return {inspection: {...inspection, importable: false, issues: [{severity: 'error', message: '重复记录：2'}]}}; };
await importDataset(file);
assert.equal($('dataset-confirm-import').disabled, true);
assert.equal(state.pendingDataset, null);
await confirmDatasetImport();
assert.deepEqual(calls, ['/api/datasets/inspect']);
assert.match($('dataset-preview-issues').textContent, /重复记录/);
cancelDatasetImport();
api = async () => ({inspection});
await importDataset(file);
api = async () => { throw new Error('disk unavailable'); };
await confirmDatasetImport();
assert.equal($('dataset-preview').open, true);
assert.match($('dataset-preview-issues').textContent, /disk unavailable/);
assert.equal($('dataset-confirm-import').disabled, false);
assert.equal(state.pendingDataset.content, 'fixture-content');
""")


@pytest.mark.parametrize("saved,system_light,expected", [(None, True, "light"), (None, False, "dark"),
                                                        ("dark", True, "dark"), ("light", False, "light")])
def test_theme_initialization_precedes_dom_and_manual_choice_survives_reload(saved, system_light, expected):
    source = (STATIC / "theme.js").read_text()
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
function browser(saved, light, storageUnavailable = false) {
  const events = {}, nodes = {}, savedValues = {};
  ['theme-toggle', 'theme-label', 'theme-icon'].forEach(id => nodes[id] = {
    attrs: {}, setAttribute(key, value) { this.attrs[key] = value; },
    addEventListener(event, callback) { this[event] = callback; },
  });
  const system = {matches: light, addEventListener(name, cb) {this.changed = cb;}};
  const document = {documentElement: {dataset: {}}, getElementById: id => nodes[id],
    addEventListener(name, callback) {events[name] = callback;}};
  const localStorage = {getItem() {if (storageUnavailable) throw new Error('disabled'); return saved;},
    setItem(key, value) {if (storageUnavailable) throw new Error('disabled'); savedValues[key] = value;}};
  const context = {window: {matchMedia: () => system, localStorage}, document};
  vm.runInNewContext(SOURCE, context);
  return {document, system, events, nodes, savedValues};
}
""".replace("SOURCE", json.dumps(source))
    body = f"""
const page = browser({json.dumps(saved)}, {json.dumps(system_light)});
assert.equal(page.document.documentElement.dataset.theme, {json.dumps(expected)});
page.events.DOMContentLoaded();
page.nodes['theme-toggle'].click();
const changed = {json.dumps(expected)} === 'dark' ? 'light' : 'dark';
assert.equal(page.document.documentElement.dataset.theme, changed);
assert.equal(page.savedValues['alphaos-theme'], changed);
page.system.matches = !page.system.matches; page.system.changed();
assert.equal(page.document.documentElement.dataset.theme, changed);
assert.equal(browser(page.savedValues['alphaos-theme'], {json.dumps(system_light)}).document.documentElement.dataset.theme, changed);
const restricted = browser(null, true, true);
restricted.events.DOMContentLoaded(); restricted.nodes['theme-toggle'].click();
assert.equal(restricted.document.documentElement.dataset.theme, 'dark');
const automatic = browser(null, false); automatic.system.matches = true; automatic.system.changed();
assert.equal(automatic.document.documentElement.dataset.theme, 'light');
"""
    result = subprocess.run([NODE, "-e", harness + body], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_delayed_settings_reads_cannot_erase_edits_or_newer_responses():
    run_ui_contract(r"""
applySettings(configured);
let deliver;
api = () => new Promise(resolve => {deliver = resolve;});
const loading = loadSettings();
$('settings-key').value = 'new-draft-key';
$('settings-model').value = 'draft-model';
markSettingsDirty();
deliver({...configured, model: 'stale-model'});
await loading;
assert.equal($('settings-model').value, 'draft-model');
assert.equal($('settings-key').value, 'new-draft-key');
assert.equal(state.settingsDirty, true);
let reads = 0;
api = async () => { reads++; return configured; };
await loadSettings();
assert.equal(reads, 0); // Navigating back must preserve unsaved edits.
state.settingsDirty = false;
const pending = [];
api = () => new Promise(resolve => pending.push(resolve));
const older = loadSettings(), newer = loadSettings();
pending[1]({...configured, model: 'newest-saved-model'}); await newer;
pending[0]({...configured, model: 'old-saved-model'}); await older;
assert.equal($('settings-model').value, 'newest-saved-model');
""")


def test_edits_during_save_are_retained_without_reenabling_provider_tests():
    run_ui_contract(r"""
applySettings(configured);
$('settings-model').value = 'submitted-model';
markSettingsDirty();
let deliver;
api = () => new Promise(resolve => {deliver = resolve;});
const saving = saveSettings({preventDefault() {}});
$('settings-model').value = 'newer-unsaved-model';
markSettingsDirty();
deliver({...configured, model: 'submitted-model'});
await saving;
assert.equal(state.settings.model, 'submitted-model');
assert.equal($('settings-model').value, 'newer-unsaved-model');
assert.equal(state.settingsDirty, true);
assert.equal($('settings-test').disabled, true);
assert.match($('settings-message').textContent, /尚未保存/);
""")


def test_calendar_scope_is_neutral_while_actionable_import_warnings_remain_visible():
    run_ui_contract(DATASET_SETUP + r"""
const calendar = {code: 'calendar_scope', severity: 'warning', message: '日期覆盖只比较文件内的日期。'};
renderDatasetInspection({...inspection, issues: [calendar]});
assert.match($('dataset-preview-issues').textContent, /格式与数据检查通过/);
assert.equal($('dataset-preview-issues').children[0].className, 'preview-success');
assert.match($('dataset-preview-adjustment').textContent, /日期覆盖/);
assert.match($('dataset-preview-adjustment').textContent, /复权/);
renderDatasetInspection({...inspection, eligible_for_research: false, issues: [calendar,
  {code: 'research_history', severity: 'warning', message: '自主研究至少需要 468 个共同交易日。'}]});
assert.doesNotMatch($('dataset-preview-issues').textContent, /检查通过/);
assert.match($('dataset-preview-issues').textContent, /468/);
assert.equal($('dataset-preview-issues').children[0].className, 'notice');
assert.equal($('dataset-confirm-import').disabled, false);
""")
