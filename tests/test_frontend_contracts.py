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


BACKTEST_SETUP = r"""
renderBacktest = report => calls.push(report);
renderBacktestSelection = () => {};
renderBacktestDataset = () => {};
refreshBacktests = async () => {};
const pending = [];
api = () => new Promise((resolve, reject) => pending.push({resolve, reject}));
"""


def test_backtest_polls_cannot_replace_a_newer_report_or_skip_selected_configuration():
    run_ui_contract(BACKTEST_SETUP + r"""
const older = loadBacktest('run-a'), newer = loadBacktest('run-a');
const completed = {id: 'run-a', status: 'completed', config: {dataset: 'saved.csv',
  factor_ids: ['saved-factor'], cost_bps: 12}, factors: [{id: 'saved-factor'}]};
pending[1].resolve(completed); await newer;
assert.equal(state.backtestReport, completed);
assert.equal($('backtest-dataset').value, 'saved.csv');
assert.equal($('backtest-cost').value, '12');
assert.deepEqual([...state.selectedFactors], ['saved-factor']);
pending[0].resolve({id: 'run-a', status: 'running'}); await older;
assert.equal(state.backtestReport, completed);
assert.deepEqual(calls, [completed]);
""")


@pytest.mark.parametrize("reselect", [False, True])
def test_backtest_reads_are_invalidated_when_selection_is_cleared_or_revisited(reselect):
    run_ui_contract(BACKTEST_SETUP + "const reselect = " + json.dumps(reselect) + r""";
const first = loadBacktest('run-a'), second = loadBacktest('run-b');
const latest = loadBacktest(reselect ? 'run-a' : '');
const completed = {id: 'run-a', status: 'completed'};
if (reselect) pending[2].resolve(completed);
await latest;
pending[0].resolve({id: 'run-a', status: 'running'}); await first;
pending[1].resolve({id: 'run-b', status: 'running'}); await second;
assert.equal(state.selectedBacktestId, reselect ? 'run-a' : null);
assert.equal(state.backtestReport, reselect ? completed : null);
assert.deepEqual(calls, [reselect ? completed : {}]);
""")


def test_backtest_read_errors_only_surface_for_the_current_request():
    run_ui_contract(BACKTEST_SETUP + r"""
const older = loadBacktest('run-a'), newer = loadBacktest('run-a');
pending[0].reject(new Error('outdated read failed')); await older;
pending[1].reject(new Error('current read failed'));
await assert.rejects(newer, /current read failed/);
const cleared = loadBacktest('run-a');
await loadBacktest('');
pending[2].reject(new Error('cleared read failed')); await cleared;
""")


@pytest.mark.parametrize("provider", ["model", "jev"])
@pytest.mark.parametrize("outcome", ["success", "failure", "error"])
@pytest.mark.parametrize("change", ["save", "edit", "refresh"])
def test_connection_results_cannot_update_changed_settings(provider, outcome, change):
    run_ui_contract("const scenario = " + json.dumps({
        "provider": provider, "outcome": outcome, "change": change,
    }) + r""";
const saved = {...configured, jev_enabled: true, jev_configured: true};
applySettings(saved);
const runTest = scenario.provider === 'model' ? testSettings : testJevSettings;
const button = $(scenario.provider === 'model' ? 'settings-test' : 'settings-jev-test');
const message = $(scenario.provider === 'model' ? 'settings-message' : 'settings-jev-message');
let deliver, reject;
api = (path, options) => path.endsWith('/test')
  ? new Promise((resolve, fail) => {deliver = resolve; reject = fail;})
  : Promise.resolve({...saved, base_url: 'https://new.example/v1', jev_base_url: 'https://new-jev.example/v1'});
const testing = runTest();
if (scenario.change === 'refresh') await loadSettings({discardChanges: true});
else {
  $('settings-url').value = 'https://new.example/v1';
  $('settings-jev-url').value = 'https://new-jev.example/v1';
  markSettingsDirty();
  if (scenario.change === 'save') await saveSettings({preventDefault() {}});
}
const before = {message: message.textContent, hidden: message.hidden, className: message.className,
  status: $('settings-status').textContent, statusClass: $('settings-status').className};
if (scenario.outcome === 'error') reject(new Error('old endpoint failed'));
else deliver({ok: scenario.outcome === 'success', message: 'old endpoint result', latency_ms: 4});
await testing;
assert.deepEqual({message: message.textContent, hidden: message.hidden, className: message.className,
  status: $('settings-status').textContent, statusClass: $('settings-status').className}, before);
assert.equal(button.disabled, scenario.change === 'edit');
assert.doesNotMatch(button.textContent, /正在测试/);
""")


@pytest.mark.parametrize("provider", ["model", "jev"])
def test_connection_tests_stay_disabled_during_pending_requests_and_saves(provider):
    run_ui_contract("const provider = " + json.dumps(provider) + r""";
const saved = {...configured, jev_enabled: true, jev_configured: true};
applySettings(saved);
const runTest = provider === 'model' ? testSettings : testJevSettings;
const button = $(provider === 'model' ? 'settings-test' : 'settings-jev-test');
const pending = [];
api = (path, options) => {calls.push(path); return new Promise(resolve => pending.push(resolve));};
const testing = runTest();
renderProviderFields(); renderJevFields();
assert.equal(button.disabled, true);
const duplicate = runTest();
assert.equal(calls.length, 1);
await duplicate;
const saving = saveSettings({preventDefault() {}});
pending[0]({ok: true}); await testing;
assert.equal(button.disabled, true);
const duringSave = runTest();
assert.equal(calls.length, 2);
await duringSave;
pending[1](saved); await saving;
assert.equal(button.disabled, false);
const secondTest = runTest();
pending[2]({ok: true, message: 'current endpoint works'}); await secondTest;
assert.equal(calls.length, 3);
const message = $(provider === 'model' ? 'settings-message' : 'settings-jev-message');
assert.match(message.textContent, /current endpoint works/);
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
