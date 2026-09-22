"use strict";

const state = {runs: [], datasets: [], report: null, selectedId: null, candidateId: null, llmConfigured: false, loading: false, submitting: false, requestSequence: 0, page: "research", factors: [], selectedFactors: new Set(), seedFactorIds: [], libraryDetailId: null, backtests: [], backtestReport: null, selectedBacktestId: null, backtestSubmitting: false, settings: null};
const $ = (id) => document.getElementById(id);
const numeric = (value) => typeof value === "number" && Number.isFinite(value);
const number = (value, digits = 2) => numeric(value) ? value.toFixed(digits) : "—";
const percent = (value, digits = 2) => numeric(value) ? `${(value * 100).toFixed(digits)}%` : "—";
const terminal = (status) => ["completed", "complete", "paused", "failed", "interrupted", "stopped", "cancelled", "error"].includes(status);
const statusText = (status) => ({queued: "等待执行", running: "运行中", completed: "已完成", complete: "已完成", paused: "已暂停", failed: "执行失败", error: "执行失败", interrupted: "已中断", stopped: "已停止", cancelled: "已取消", ok: "通过验证", rejected: "已拒绝"}[status] || status || "等待任务");
const statusClass = (status) => ({completed: "ok", complete: "ok", ok: "ok", running: "running", queued: "running", failed: "failed", error: "failed", rejected: "rejected", interrupted: "rejected", stopped: "rejected", paused: "rejected"}[status] || "neutral");
const stopReasonText = (reason) => ({user_pause: "已暂停，进度已保存。", time_budget: "已达到运行时间预算。", llm_budget: "已达到模型调用预算。", model_budget: "已达到模型调用预算。", token_budget: "已达到 Token 预算。", call_budget: "已达到模型请求次数上限。", checkpoint: "已保存检查点，可以恢复继续研究。", no_valid_candidates: "没有候选通过验证，请查看拒绝原因与日志。", no_reviewed_candidates: "没有候选通过验证与模型复核。", research_complete: "研究已完成。", reviewer_stop: "复核建议结束研究。"}[reason] || reason);
const modelLabel = (model) => ({rank: "因子排名", ridge: "岭回归", hist_gbdt: "梯度提升树"}[model] || model || "—");
const reviewLabel = (decision) => ({approve: "通过复核", revise: "建议修改", reject: "复核拒绝"}[decision] || decision || "未复核");
const actionLabel = (action) => ({explore: "探索", refine: "优化", combine: "组合", crossover: "交叉", model_switch: "模型切换", mutate: "变异", revise: "修订", exploit: "改进", seed: "初始提案"}[action] || action || "未记录");
const trialMetrics = (trial) => trial.development?.metrics || trial.metrics || {};
const trialFeatures = (trial) => Array.isArray(trial.features) && trial.features.length ? trial.features.map((feature) => typeof feature === "string" ? feature : feature.expression || feature.name || "—") : [trial.expression || "—"];
const trialUsages = (trial) => [["提案", trial.proposer_usage || trial.llm_usage], ["复核", trial.reviewer_usage]].filter(([, usage]) => usage && typeof usage === "object");
const technicalIntegrityText = /sha-?256|checksum|fingerprint|文件校验|源码指纹|数据指纹/i;
function userMessage(message) {
  const text = String(message);
  if (/Snapshot checksum changed/i.test(text)) return "数据快照已变更，请重新创建研究。";
  if (/Research code changed since creation/i.test(text)) return "研究程序已更新，请重新创建研究。";
  if (/Numerical dependencies changed/i.test(text)) return "计算依赖已变化，请恢复项目依赖或重新创建研究。";
  return technicalIntegrityText.test(text) ? text.replace(/sha-?256|checksum|fingerprint|文件校验|源码指纹|数据指纹/gi, "文件标识").replace(/\b[a-f0-9]{64}\b/gi, "（内部标识）") : text;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function showError(message) {
  $("global-error").textContent = message ? userMessage(message) : "";
  $("global-error").hidden = !message;
}

async function api(path, options = {}) {
  const {timeoutMs = 20000, ...requestOptions} = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, {...requestOptions, signal: controller.signal, headers: {"Content-Type": "application/json", ...requestOptions.headers}});
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.error === "string" ? body.error : typeof body.detail === "string" ? body.detail : `请求失败 (${response.status})`;
      throw new Error(detail);
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("请求超时，请检查本地研究引擎是否仍在运行。");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function formatDate(value, withTime = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", withTime ? {month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false} : {year: "numeric", month: "2-digit", day: "2-digit"});
}

function renderRuns() {
  const container = $("run-list");
  container.replaceChildren();
  if (!state.runs.length) {
    container.append(element("p", "muted small", "还没有实验记录"));
    return;
  }
  state.runs.forEach((run) => {
    const button = element("button", `run-list-item${state.selectedId === run.id ? " selected" : ""}`);
    button.type = "button";
    button.setAttribute("aria-current", state.selectedId === run.id ? "true" : "false");
    button.append(element("strong", "", run.id));
    const meta = element("small");
    meta.append(element("span", "", statusText(run.status)), element("span", "", formatDate(run.created_at, true)));
    button.append(meta);
    button.addEventListener("click", () => {showPage("research"); selectRun(run.id);});
    container.append(button);
  });
}

async function refreshRuns(selectLatest = false) {
  const data = await api("/api/runs");
  state.runs = Array.isArray(data.runs) ? data.runs : [];
  renderRuns();
  if (selectLatest && !state.selectedId && state.runs.length) await selectRun(state.runs[0].id);
}

async function selectRun(id) {
  state.selectedId = id;
  state.candidateId = null;
  state.report = null;
  renderReport({id, status: "queued", progress: {}});
  renderRuns();
  await loadReport();
}

async function loadReport() {
  if (!state.selectedId) return;
  const id = state.selectedId;
  const sequence = ++state.requestSequence;
  try {
    const report = await api(`/api/runs/${encodeURIComponent(id)}`);
    if (id !== state.selectedId || sequence !== state.requestSequence) return;
    state.report = report;
    renderReport(report);
    showError("");
  } catch (error) {
    if (id === state.selectedId && sequence === state.requestSequence) showError(`读取实验失败：${error.message}`);
  }
}

function updateMetric(id, value, formatter = percent, colorize = false) {
  const node = $(id);
  node.textContent = formatter(value);
  node.className = colorize && numeric(value) ? (value >= 0 ? "positive" : "negative") : "";
}

function renderReport(report) {
  const progress = report.progress || {};
  const source = report.source || {};
  const completed = numeric(progress.completed) ? progress.completed : 0;
  const total = numeric(progress.total) ? progress.total : report.config?.trials;
  const holdout = report.holdout || {};
  const metrics = holdout.metrics || {};
  const hasHoldout = Array.isArray(holdout.curve) && holdout.curve.length > 0;
  $("run-title").textContent = report.id || "研究任务";
  $("run-status").textContent = statusText(report.status);
  $("run-status").className = `status ${statusClass(report.status)}`;
  $("run-subtitle").textContent = report.config?.direction || "研究引擎正在准备任务与数据。";
  if (report.config?.mode) {
    const usages = (Array.isArray(report.trials) ? report.trials : []).flatMap((trial) => trialUsages(trial).map(([, usage]) => usage));
    const providers = [...new Set(usages.map((usage) => usage.provider === "codex_cli" ? "本机 Codex" : "API 模型"))];
    const modeName = report.config.mode === "agent" ? "自主研究" : report.config.mode === "local" ? "历史本地搜索" : "历史 LLM 辅助";
    const execution = `本次：${modeName}${usages.length ? ` · ${providers.join(" / ")} · ${usages.length} 次模型返回` : report.config.mode !== "local" ? " · 尚无模型返回" : ""}`;
    $("run-subtitle").append(element("span", "run-provider-note", execution));
  }
  $("progress-label").textContent = numeric(total) ? `${completed} / ${total} 个候选已评估` : "正在准备候选验证";
  $("progress").max = numeric(total) && total > 0 ? total : 1;
  $("progress").value = completed;
  $("elapsed").textContent = numeric(progress.elapsed_seconds) ? `耗时 ${number(progress.elapsed_seconds, 1)} s` : "—";
  const assetCount = Array.isArray(source.assets) ? source.assets.length : source.assets;
  $("source-label").textContent = source.label ? `${source.label} · ${assetCount ?? "—"} 个资产 · ${source.rows ?? "—"} 个交易日` : "正在加载研究数据";
  $("resume-run").hidden = Boolean(report.externally_active) || !["paused", "interrupted", "failed", "stopped", "error"].includes(report.status);
  $("resume-run").disabled = state.submitting || Boolean(report.active);
  $("pause-run").hidden = !["running", "queued"].includes(report.status) || report.active === false || Boolean(report.externally_active);
  $("pause-run").disabled = Boolean(report.pause_requested);
  $("pause-run").textContent = report.pause_requested ? "等待暂停…" : "暂停";
  const stage = report.research?.stage ? ({propose: 1, evaluate: 2, review: 3, select: 4, complete: 4}[report.research.stage] || 0) : hasHoldout ? 4 : report.selected ? 3 : completed > 0 ? 2 : report.status === "running" ? 1 : 0;
  const stages = report.config?.mode === "agent" ? ["01 模型提案", "02 训练与验证", "03 模型复核", "04 留出检验"] : ["01 提案", "02 因果检查", "03 滚动验证", "04 留出检验"];
  [...$("stage-track").children].forEach((node, index) => {node.classList.toggle("active", index < stage); node.textContent = stages[index];});
  if (source.kind === "synthetic" || source.kind === "demo" || report.config?.dataset === "demo") $("source-label").append(element("span", "status rejected inline-badge", "合成数据"));
  renderWarnings("run-warnings", report.warnings, report.error || (report.stop_reason && report.stop_reason !== "trial_budget" ? stopReasonText(report.stop_reason) : null));
  updateMetric("metric-return", metrics.total_return, percent, true);
  updateMetric("metric-sharpe", metrics.sharpe, number);
  updateMetric("metric-drawdown", metrics.max_drawdown, percent);
  updateMetric("metric-excess", metrics.excess_return, percent, true);
  $("metric-return-note").textContent = numeric(metrics.days) ? `${metrics.days} 个留出交易日 · 扣除成本` : "等待留出检验";
  $("metric-sharpe-note").textContent = Array.isArray(holdout.sharpe_interval) && holdout.sharpe_interval.length === 2 ? `自助法区间 [${number(holdout.sharpe_interval[0])}, ${number(holdout.sharpe_interval[1])}]` : "基于扣费后的日收益";
  $("holdout-badge").textContent = hasHoldout ? "留出期已评估" : "等待最终候选";
  $("holdout-badge").className = `status ${hasHoldout ? "ok" : "neutral"}`;
  $("chart-period").textContent = report.split?.holdout_start ? `${String(report.split.holdout_start).slice(0,10)} → ${String(report.split.holdout_end || "").slice(0,10)}` : "留出期未评估";
  renderChart(holdout.curve || []);
  renderBaselines(holdout);
  renderResearchMemory(report);
  renderTrials(report);
  renderCandidate(report);
  renderEvents(report.events || []);
  renderArtifacts(report);
}

function renderBaselines(holdout) {
  const body = $("baseline-body");
  body.replaceChildren();
  const baselines = [
    ["最终候选", holdout.metrics],
    ["等权基准", holdout.baseline_metrics],
    ["20 日动量", holdout.fixed_baselines?.["20"]],
    ["60 日动量", holdout.fixed_baselines?.["60"] ?? holdout.momentum_metrics],
  ];
  baselines.forEach(([label, metrics], index) => {
    const values = metrics || {};
    const row = element("tr", index === 0 ? "baseline-selected" : "");
    const name = element("th", "", label); name.scope = "row";
    row.append(name, element("td", "", percent(values.total_return)), element("td", "", number(values.sharpe)), element("td", "", percent(values.max_drawdown)));
    body.append(row);
  });
}

function svgElement(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  if (text !== undefined) node.textContent = String(text);
  return node;
}

function renderChart(rawCurve, targetId = "equity-chart", drawdown = false) {
  const container = $(targetId);
  container.replaceChildren();
  const rows = Array.isArray(rawCurve) ? rawCurve : [];
  const curve = (drawdown ? rows.map((row) => ({...row, equity: row.drawdown, benchmark_equity: row.benchmark_drawdown})) : rows).filter((row) => numeric(row.equity) && numeric(row.benchmark_equity));
  if (!curve.length) {
    const empty = element("div", "empty-state");
    empty.append(element("span", "empty-symbol", "⌁"), element("p", "", "暂无数据"));
    container.append(empty);
    return;
  }
  const width = Math.max(360, container.clientWidth || 640);
  const height = Math.max(165, container.clientHeight || 265), left = 56, right = 24, top = 23, bottom = 34;
  const values = curve.flatMap((row) => [row.equity, row.benchmark_equity]);
  const min = Math.min(...values, drawdown ? 0 : 1), max = Math.max(...values, drawdown ? 0 : 1);
  const padding = Math.max((max - min) * .16, .012);
  const yMin = min - padding, yMax = max + padding;
  const x = (index) => left + index / Math.max(1, curve.length - 1) * (width - left - right);
  const y = (value) => top + (yMax - value) / (yMax - yMin) * (height - top - bottom);
  const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `策略与等权基准${drawdown ? "回撤" : "净值"}曲线，共 ${curve.length} 个交易日`});
  for (let index = 0; index < 5; index++) {
    const value = yMin + (yMax - yMin) * index / 4;
    svg.append(svgElement("line", {x1: left, y1: y(value), x2: width - right, y2: y(value), stroke: "#253149", "stroke-width": .8, "stroke-dasharray": "3 5"}));
    svg.append(svgElement("text", {x: left - 13, y: y(value) + 3, fill: "#7087a8", "font-size": 10, "text-anchor": "end", "font-family": "monospace"}, drawdown ? percent(value, 1) : value.toFixed(2)));
  }
  const ticks = [...new Set([0, Math.floor((curve.length - 1) / 3), Math.floor(2 * (curve.length - 1) / 3), curve.length - 1])];
  ticks.forEach((index) => svg.append(svgElement("text", {x: x(index), y: height - 10, fill: "#7087a8", "font-size": 9, "text-anchor": index === 0 ? "start" : index === curve.length - 1 ? "end" : "middle", "font-family": "monospace"}, String(curve[index].date || "").slice(0, 10))));
  [ ["benchmark_equity", "#7185c9"], ["equity", "#55d8b4"] ].forEach(([field, color]) => {
    const points = curve.map((row, index) => `${x(index).toFixed(2)},${y(row[field]).toFixed(2)}`).join(" ");
    svg.append(svgElement("polyline", {points, fill: "none", stroke: color, "stroke-width": field === "equity" ? 2.7 : 1.8, "stroke-linecap": "round", "stroke-linejoin": "round"}));
    svg.append(svgElement("circle", {cx: x(curve.length - 1), cy: y(curve.at(-1)[field]), r: 3, fill: color}));
  });
  container.append(svg);
}

function renderTrials(report) {
  const trials = Array.isArray(report.trials) ? report.trials : [];
  $("trial-count").textContent = String(trials.length);
  const filter = $("trial-filter").value;
  const visible = trials.filter((trial) => filter === "all" || trial.status === filter).sort((a, b) => (numeric(b.score) ? b.score : -Infinity) - (numeric(a.score) ? a.score : -Infinity));
  const body = $("trials-body");
  body.replaceChildren();
  if (!visible.length) {
    const row = element("tr"), cell = element("td", "table-empty", trials.length ? "没有匹配此筛选条件的候选。" : "尚无候选。开始实验后，验证结果会自动更新。");
    cell.colSpan = 6; row.append(cell); body.append(row); return;
  }
  const selectedId = state.candidateId || report.selected?.id;
  visible.forEach((trial) => {
    const row = element("tr", trial.id === selectedId ? "selected" : "");
    row.dataset.trialId = String(trial.id);
    row.tabIndex = 0;
    row.setAttribute("aria-label", `查看候选 ${trial.name || trial.id}`);
    const title = element("td");
    title.append(element("strong", "", `${report.selected?.id === trial.id ? "★ " : ""}${trial.name || trial.id}`), element("small", "", trial.hypothesis || "未提供假设"));
    const features = trialFeatures(trial), expression = element("td");
    features.slice(0, 2).forEach((feature) => expression.append(element("code", "", feature)));
    if (features.length > 2) expression.append(element("small", "", `共 ${features.length} 个特征，详情中查看`));
    const status = element("td"); status.append(element("span", `status ${statusClass(trial.status)}`, statusText(trial.status)));
    if (trial.review || report.config?.mode === "agent") status.append(element("small", "review-state", reviewLabel(trial.review?.decision)));
    row.append(title, expression, element("td", "", number(trial.score ?? trial.development?.score, 3)), element("td", "", number(trialMetrics(trial).sharpe)), element("td", "", modelLabel(trial.model || (report.config?.mode === "agent" ? null : "rank"))), status);
    const select = () => {state.candidateId = trial.id; renderTrials(report); renderCandidate(report);};
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); select();}});
    body.append(row);
  });
}

function renderCandidate(report) {
  const trials = Array.isArray(report.trials) ? report.trials : [];
  const selected = trials.find((trial) => trial.id === state.candidateId) || report.selected;
  const container = $("candidate-detail");
  container.replaceChildren();
  if (!selected) {container.append(element("p", "muted", "选择表格中的候选，查看研究假设与推导过程。")); return;}
  container.append(element("h3", "", selected.name || selected.id), element("p", "", selected.hypothesis || "未提供假设"));
  const features = trialFeatures(selected);
  const candidateType = element("div", "detail-row");
  candidateType.append(element("span", "", `模型：${modelLabel(selected.model || (report.config?.mode === "agent" ? null : "rank"))}`), element("span", "", `${features.length} 个特征`));
  container.append(candidateType);
  const featureList = element("div", "candidate-features");
  features.forEach((feature, index) => {const row = element("div", "candidate-feature"); row.append(element("span", "small muted", `F${index + 1}`), element("code", "", feature)); featureList.append(row);});
  container.append(featureList);
  const metadata = element("div", "detail-row");
  metadata.append(element("span", "", `来源：${selected.origin || "—"}`), element("span", "", `父候选：${(selected.parents || []).join("、") || "初始提案"}`));
  container.append(metadata);
  for (const [role, usage] of trialUsages(selected)) {
    const details = element("div", "detail-row");
    details.append(element("span", "", `${role} · ${usage.provider === "codex_cli" || selected.origin === "codex_cli" ? "本机 Codex" : "API 模型"}`));
    details.append(element("span", "", `模型：${usage.model || "—"}`));
    details.append(element("span", "", `Token：${numeric(usage.reported_total_tokens) ? number(usage.reported_total_tokens, 0) : "未报告"}`));
    details.append(element("span", "", `用时：${numeric(usage.seconds) ? `${number(usage.seconds, 1)} s` : "未报告"}`));
    container.append(details);
  }
  const folds = selected.development?.folds || selected.folds;
  if (Array.isArray(folds) && folds.length) {
    container.append(element("p", "", folds.map((fold, index) => `窗口 ${index + 1}：夏普 ${number(fold.metrics?.sharpe ?? fold.sharpe)}`).join(" · ")));
  }
  if (report.config?.mode === "agent" || selected.review || selected.training_audit) renderTrialTrajectory(container, selected, report);
  if (selected.reason) container.append(element("div", "notice", selected.reason));
  if (report.selected?.id === selected.id) container.append(element("p", "", "★ 最终候选，由开发期验证结果选定。"));
}

function renderResearchMemory(report) {
  const research = report.research;
  $("research-memory").hidden = !research;
  if (!research) return;
  const graph = research.graph || {}, memory = research.memory || {};
  const summary = $("memory-summary"); summary.replaceChildren();
  const counts = [["研究节点", Array.isArray(graph.nodes) ? graph.nodes.length : null], ["轨迹连接", Array.isArray(graph.edges) ? graph.edges.length : null], ["召回经验", memory.retrieved], ["新增经验", memory.stored]];
  counts.forEach(([label, value]) => {const cell = element("div", "memory-stat"); cell.append(element("strong", "", numeric(value) ? number(value, 0) : "—"), element("span", "", label)); summary.append(cell);});
  const models = research.model_counts || {};
  $("research-models").textContent = Object.entries(models).filter(([, count]) => numeric(count) && count > 0).map(([model, count]) => `${modelLabel(model)} ${count}`).join(" · ");
}

function renderTrialTrajectory(container, trial, report) {
  const graphNodes = report.research?.graph?.nodes || [];
  const graphNode = graphNodes.find((node) => node.id === `${report.id}:${trial.id}`)
    || graphNodes.find((node) => node.id === trial.id || (node.trial_id === trial.id && (!node.run_id || node.run_id === report.id))) || {};
  const trajectory = element("section", "trial-trajectory");
  trajectory.append(element("h4", "", "研究轨迹"));
  const path = element("div", "trajectory-path");
  const parents = Array.isArray(trial.parents) && trial.parents.length ? trial.parents : Array.isArray(graphNode.parents) ? graphNode.parents : [];
  if (!parents.length) path.append(element("span", "muted small", "初始探索"));
  parents.forEach((id) => {
    const candidate = (report.trials || []).find((item) => item.id === id || `${report.id}:${item.id}` === id);
    const button = element("button", "button small secondary", candidate?.id || String(id).split(":").at(-1)); button.type = "button";
    button.disabled = !candidate;
    button.title = candidate ? `查看 ${candidate.id}` : `跨运行候选：${id}`;
    if (candidate) button.addEventListener("click", () => {state.candidateId = candidate.id; renderTrials(report); renderCandidate(report);});
    path.append(button);
  });
  path.append(element("span", "muted", "→"), element("span", "status neutral", trial.id || "当前候选"), element("span", "small muted", actionLabel(trial.action || graphNode.action)));
  trajectory.append(path);
  if (trial.model_params && typeof trial.model_params === "object" && Object.keys(trial.model_params).length) trajectory.append(element("p", "audit-caption", `模型参数：${Object.entries(trial.model_params).map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · ")}`));
  const review = trial.review || {};
  const reviewHeading = element("div", "review-heading");
  reviewHeading.append(element("h4", "", "模型复核"), element("span", `status ${review.decision === "approve" ? "ok" : review.decision === "reject" ? "failed" : review.decision === "revise" ? "rejected" : "neutral"}`, reviewLabel(review.decision)));
  trajectory.append(reviewHeading);
  if (review.critique) trajectory.append(element("p", "", review.critique));
  if (Array.isArray(review.risks) && review.risks.length) {
    const risks = element("ul", "review-risks"); review.risks.forEach((risk) => risks.append(element("li", "", typeof risk === "string" ? risk : JSON.stringify(risk)))); trajectory.append(risks);
  }
  if (review.suggested_action) trajectory.append(element("p", "", `下一步：${actionLabel(review.suggested_action)}`));
  const audit = trial.training_audit || trial.development?.training_audit;
  if (audit && typeof audit === "object") {
    trajectory.append(element("h4", "", "训练记录"));
    const auditRow = element("div", "detail-row");
    const folds = Array.isArray(audit.folds) ? audit.folds : [];
    if (folds.length) auditRow.append(element("span", "", `${folds.length} 折验证`));
    if (numeric(audit.fit_count)) auditRow.append(element("span", "", `拟合 ${audit.fit_count} 次`));
    if (audit.fit_cutoff_date) auditRow.append(element("span", "", `拟合截止：${audit.fit_cutoff_date}`));
    if (audit.first_prediction_date) auditRow.append(element("span", "", `首个预测：${audit.first_prediction_date}`));
    if (typeof audit.training_frozen === "boolean") auditRow.append(element("span", "", audit.training_frozen ? "训练已冻结" : "滚动训练"));
    if (numeric(audit.finite_fraction)) auditRow.append(element("span", "", `有效预测 ${percent(audit.finite_fraction, 1)}`));
    trajectory.append(auditRow);
    folds.forEach((fold, index) => {
      const fitBlocks = Array.isArray(fold.fit_blocks) ? fold.fit_blocks : [];
      const lastLabelDate = fitBlocks.at(-1)?.label_end_date;
      trajectory.append(element("p", "audit-caption", `第 ${index + 1} 折 · 拟合 ${number(fold.fit_count, 0)} 次${lastLabelDate ? ` · 标签截止 ${lastLabelDate}` : ""}`));
    });
    const blocks = Array.isArray(audit.fit_blocks) ? audit.fit_blocks : [];
    const latest = blocks.at(-1);
    if (latest) trajectory.append(element("p", "audit-caption", `最近训练：${latest.train_feature_start_date || "—"} — ${latest.train_feature_end_date || "—"} · 标签截止 ${latest.label_end_date || "—"} · 样本 ${latest.samples ?? "—"}`));
    const details = element("details", "training-audit"); details.append(element("summary", "", "查看训练记录详情"), element("pre", "", JSON.stringify(audit, null, 2))); trajectory.append(details);
  }
  container.append(trajectory);
}

function renderEvents(events) {
  const entries = Array.isArray(events) ? events : [];
  $("event-count").textContent = String(entries.length);
  const container = $("event-log");
  container.replaceChildren();
  if (!entries.length) {container.append(element("p", "muted", "等待研究事件…")); return;}
  entries.slice(-60).reverse().forEach((event) => {
    const row = element("div", `event ${["warning", "error"].includes(event.level) ? event.level : ""}`);
    const time = event.time ? new Date(event.time) : null;
    row.append(element("time", "", time && Number.isFinite(time.getTime()) ? time.toLocaleTimeString("zh-CN", {hour12: false}) : "—"), element("span", "", event.message || ""));
    container.append(row);
  });
}

function renderArtifacts(report) {
  const container = $("artifact-links");
  container.replaceChildren();
  if (!terminal(report.status)) {container.append(element("span", "muted small", "实验结束后可导出研究产物")); return;}
  const artifacts = [["report.html", "↗ 离线报告"], ["report.json", "↓ JSON"], ["trials.csv", "↓ 候选 CSV"]];
  if (report.selected) artifacts.push(["selected_factor.py", "↓ 候选代码"]);
  if (Array.isArray(report.artifacts)) {
    for (const [file, label] of [["selected_candidate.json", "↓ 候选配置"], ["training_audit.json", "↓ 训练记录"], [report.artifacts.includes("research_graph.json") ? "research_graph.json" : "research.json", "↓ 研究轨迹"]]) if (report.artifacts.includes(file)) artifacts.push([file, label]);
  }
  artifacts.filter(([file]) => !Array.isArray(report.artifacts) || report.artifacts.includes(file)).forEach(([file, label]) => {
    const link = element("a", "button secondary", label);
    link.href = `/artifacts/${encodeURIComponent(report.id)}/${file}`;
    if (file === "report.html") {link.target = "_blank"; link.rel = "noopener";} else link.download = file;
    container.append(link);
  });
}

function renderDatasetNote() {
  const selected = state.datasets.find((dataset) => dataset.id === $("dataset").value);
  $("dataset-note").textContent = selected ? `${selected.assets ?? "—"} 个资产 · ${selected.rows ?? "—"} 个交易日 · ${String(selected.start || "").slice(0,10)} — ${String(selected.end || "").slice(0,10)}${selected.rows < 468 ? "。可用于独立回测；自动研究至少需 468 个交易日。" : ""}` : "导入行情 CSV 后选择研究数据。";
  renderModeNote();
}

function renderModeNote() {
  const isCodex = state.settings?.provider === "codex_cli";
  $("mode-note").textContent = !state.llmConfigured ? "请先在「设置」中配置模型连接。" : isCodex ? "本机 Codex · 提案与复核；每次请求预留 65,536 Token。" : "API 模型 · 提案与复核。";
  const dataset = state.datasets.find((item) => item.id === $("dataset").value);
  $("start-run").disabled = !state.llmConfigured || !dataset || dataset.rows < 468 || state.submitting || Boolean(state.importingDataset);
}

function renderDatasets(preferredId) {
  for (const id of ["dataset", "backtest-dataset"]) {
    const previous = $(id).value;
    const options = state.datasets.map((dataset) => {const option = element("option", "", dataset.label || dataset.id); option.value = dataset.id; return option;});
    if (!options.length) {const empty = element("option", "", "请先导入 CSV"); empty.value = ""; options.push(empty);}
    $(id).replaceChildren(...options);
    $(id).disabled = !state.datasets.length;
    const selected = preferredId || previous;
    if (state.datasets.some((dataset) => dataset.id === selected)) $(id).value = selected;
  }
  renderDatasetNote(); renderBacktestDataset(); renderBacktestSelection();
}

async function importDataset(file) {
  if (!file || state.importingDataset) return;
  $("dataset-file").value = "";
  const message = $("dataset-import-message"); message.hidden = true;
  if (!/\.csv$/i.test(file.name)) {showError("请选择 .csv 文件。"); return;}
  if (file.size > 20000000) {showError("CSV 文件不能超过 20 MB。"); return;}
  state.importingDataset = true;
  document.querySelectorAll("[data-import-dataset]").forEach((button) => {button.disabled = true; button.textContent = "正在导入…";});
  renderModeNote(); renderBacktestSelection(); showError("");
  try {
    let content;
    try {content = new TextDecoder("utf-8", {fatal: true}).decode(await file.arrayBuffer());}
    catch {throw new Error("请将 CSV 保存为 UTF-8 编码后重试。");}
    const result = await api("/api/datasets", {method: "POST", body: JSON.stringify({name: file.name, content}), timeoutMs: 60000});
    state.datasets.push(result.dataset);
    state.datasets.sort((a, b) => a.id.localeCompare(b.id));
    $("backtest-start").value = ""; $("backtest-end").value = "";
    renderDatasets(result.dataset.id);
    message.textContent = `已导入 ${result.dataset.id} · ${result.dataset.assets} 个资产 · ${result.dataset.rows} 个交易日`;
    message.hidden = false;
  } catch (error) {
    showError(`导入失败：${error.message}`);
  } finally {
    state.importingDataset = false;
    $("dataset-file").value = "";
    document.querySelectorAll("[data-import-dataset]").forEach((button) => {button.disabled = false; button.textContent = "导入 CSV";});
    renderModeNote(); renderBacktestSelection();
  }
}

async function submitRun(event) {
  event.preventDefault();
  if (state.submitting) return;
  const form = new FormData($("run-form"));
  const dataset = state.datasets.find((item) => item.id === form.get("dataset"));
  if (!dataset) {showError("请先导入并选择 CSV 数据集。"); return;}
  if (dataset.rows < 468) {showError("自动研究至少需要 468 个共同交易日；当前数据可用于独立回测。"); return;}
  if (!state.llmConfigured) {showError("请先在「设置」中配置模型连接。"); return;}
  const config = {direction: String(form.get("direction") || "").trim(), mode: "agent", dataset: form.get("dataset")};
  ["trials", "seed", "cost_bps", "top_k", "rebalance_every", "max_seconds", "max_llm_calls", "max_llm_tokens"].forEach((key) => {config[key] = Number(form.get(key));});
  config.constraints = {required_window: form.get("required_window") ? Number(form.get("required_window")) : null, forbidden_fields: form.getAll("forbidden_fields"), max_turnover: form.get("max_turnover") !== "" ? Number(form.get("max_turnover")) / 100 : null};
  if (state.seedFactorIds.length) config.factor_ids = state.seedFactorIds;
  if (state.seedFactorIds.length > Math.min(config.trials, 50)) {showError("初始因子数量不能超过候选预算与 50 个的上限。"); return;}
  state.submitting = true;
  $("start-run").disabled = true;
  $("start-run").textContent = "正在创建研究任务…";
  showError("");
  try {
    const run = await api("/api/runs", {method: "POST", body: JSON.stringify(config)});
    await selectRun(run.id);
    await refreshRuns();
  } catch (error) {showError(`无法启动研究：${error.message}`);}
  finally {state.submitting = false; renderModeNote(); $("start-run").textContent = "开始研究 ↗";}
}

async function resumeRun() {
  if (state.submitting || !state.selectedId) return;
  state.submitting = true; $("resume-run").disabled = true;
  try {await api(`/api/runs/${encodeURIComponent(state.selectedId)}/resume`, {method: "POST", body: "{}"}); await loadReport(); await refreshRuns();}
  catch (error) {showError(`恢复失败：${error.message}`);}
  finally {state.submitting = false; $("resume-run").disabled = Boolean(state.report?.active);}
}

async function initialize() {
  $("run-form").addEventListener("submit", submitRun);
  $("resume-run").addEventListener("click", resumeRun);
  $("pause-run").addEventListener("click", pauseRun);
  initializeProduct();
  $("dataset").addEventListener("change", renderDatasetNote);
  $("trial-filter").addEventListener("change", () => {if (state.report) renderTrials(state.report);});
  $("refresh-runs").addEventListener("click", () => refreshRuns(true).catch((error) => showError(`刷新失败：${error.message}`)));
  document.querySelectorAll("[data-import-dataset]").forEach((button) => button.addEventListener("click", () => $("dataset-file").click()));
  $("dataset-file").addEventListener("change", () => importDataset($("dataset-file").files[0]));
  const results = await Promise.allSettled([api("/api/health"), api("/api/datasets"), refreshRuns(true), api("/api/settings")]);
  const [health, datasets] = results;
  if (health.status === "fulfilled" && health.value.ok) {
    $("connection-state").textContent = "本地研究引擎已连接";
    $("connection-dot").classList.add("online");
    $("version").textContent = `ALPHARESEARCHOS ${health.value.version || ""}`;
    state.llmConfigured = Boolean(health.value.llm_configured);
    updateModelStatus();
  } else {$("connection-state").textContent = "本地引擎连接失败"; showError("无法连接本地研究引擎。请确认服务正在运行后刷新页面。");}
  if (datasets.status === "fulfilled") {
    state.datasets = Array.isArray(datasets.value.datasets) ? datasets.value.datasets : [];
    renderDatasets();
  } else showError(`读取数据集失败：${datasets.reason.message}`);
  if (results[2].status === "rejected") showError(`读取历史实验失败：${results[2].reason.message}`);
  if (results[3].status === "fulfilled") applySettings(results[3].value);
  else showError(`读取模型配置失败：${results[3].reason.message}`);
  renderDatasetNote(); renderBacktestDataset(); renderModeNote();
  setTimeout(poll, 2000);
}

async function poll() {
  try {
    if (!document.hidden && !state.loading) {
      state.loading = true;
      if (state.selectedId && (!state.report || state.report.active || !terminal(state.report.status))) await loadReport();
      if (state.selectedBacktestId && (!state.backtestReport || state.backtestReport.active || !terminal(state.backtestReport.status))) await loadBacktest(state.selectedBacktestId);
      await refreshRuns(!state.selectedId);
    }
  } catch (error) {showError(`同步失败：${error.message}`);}
  finally {state.loading = false; setTimeout(poll, 2500);}
}

function renderWarnings(targetId, warnings, error) {
  const target = $(targetId);
  target.replaceChildren();
  if (error) target.append(element("div", "notice", userMessage(typeof error === "string" ? error : JSON.stringify(error))));
  const items = Array.isArray(warnings) ? [...new Set(warnings.map((warning) => userMessage(typeof warning === "string" ? warning : JSON.stringify(warning))))] : [];
  if (items.length) {
    const details = element("details", "validation-notes");
    details.append(element("summary", "", `验证说明 · ${items.length}`));
    items.forEach((warning) => details.append(element("p", "", typeof warning === "string" ? warning : JSON.stringify(warning))));
    target.append(details);
  }
}

function updateModelStatus() {
  $("model-status").textContent = state.llmConfigured ? state.settings?.provider === "codex_cli" ? "本机 Codex 已就绪" : "API 模型已配置" : "模型未配置";
  $("model-status").className = `status ${state.llmConfigured ? "ok" : "neutral"}`;
  renderModeNote();
}

function showPage(page) {
  if (!["research", "library", "backtest", "settings"].includes(page)) return;
  state.page = page;
  document.querySelectorAll(".page-pane").forEach((pane) => {pane.hidden = pane.id !== `page-${page}`;});
  document.querySelectorAll(".app-navigation [data-page]").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page);
    button.setAttribute("aria-current", button.dataset.page === page ? "page" : "false");
  });
  $("page-title").textContent = {research: "自动研究", library: "因子库", backtest: "独立回测", settings: "设置"}[page];
  showError("");
  if (page === "library") loadFactors().catch((error) => libraryMessage(error.message, true));
  if (page === "backtest") {
    renderBacktestSelection();
    refreshBacktests().catch((error) => showError(`读取回测记录失败：${error.message}`));
  }
  if (page === "settings") loadSettings().catch((error) => settingsMessage(`读取失败：${error.message}`, true));
  redrawVisibleCharts();
}

function redrawVisibleCharts() {
  if (state.page === "research") renderChart(state.report?.holdout?.curve || []);
  if (state.page === "backtest") {
    renderChart(state.backtestReport?.curve || [], "backtest-chart");
    renderChart(state.backtestReport?.curve || [], "backtest-drawdown-chart", true);
  }
}

async function pauseRun() {
  if (!state.selectedId) return;
  $("pause-run").disabled = true;
  try {
    await api(`/api/runs/${encodeURIComponent(state.selectedId)}/pause`, {method: "POST", body: "{}"});
    $("pause-run").textContent = "等待暂停…";
    await loadReport();
  } catch (error) {showError(`暂停失败：${error.message}`); $("pause-run").disabled = false;}
}

function libraryMessage(message, isError = false) {
  $("library-message").textContent = message;
  $("library-message").className = `notice${isError ? " error" : ""}`;
  $("library-message").hidden = !message;
}

async function loadFactors() {
  const data = await api("/api/factors");
  state.factors = Array.isArray(data.factors) ? data.factors : [];
  const available = new Set(state.factors.map((factor) => factor.id));
  state.selectedFactors = new Set([...state.selectedFactors].filter((id) => available.has(id)));
  renderLibrary();
  renderBacktestSelection();
}

function filteredFactors() {
  const query = $("factor-search").value.trim().toLocaleLowerCase();
  const sort = $("factor-sort").value;
  return state.factors.filter((factor) => [factor.name, factor.expression, factor.hypothesis, factor.run_id].some((value) => String(value || "").toLocaleLowerCase().includes(query))).sort((a, b) => {
    if (sort === "score") return (numeric(b.score) ? b.score : -Infinity) - (numeric(a.score) ? a.score : -Infinity);
    if (sort === "name") return String(a.name || "").localeCompare(String(b.name || ""), "zh-CN");
    return String(b.created_at || "").localeCompare(String(a.created_at || ""));
  });
}

function renderLibrary() {
  const rows = filteredFactors();
  $("library-count").textContent = `${rows.length} / ${state.factors.length}`;
  const selectedCount = state.selectedFactors.size;
  $("factor-selection-count").textContent = `已选 ${selectedCount} 个`;
  $("factor-to-backtest").disabled = selectedCount < 1 || selectedCount > 10;
  $("factor-to-backtest").title = selectedCount > 10 ? "单次回测最多选择 10 个因子" : "";
  $("factor-to-research").disabled = selectedCount < 1 || selectedCount > 50;
  $("library-export").disabled = !state.factors.length;
  $("library-export").textContent = selectedCount ? `导出所选 (${selectedCount})` : "导出 JSON";
  const selectedVisible = rows.filter((factor) => state.selectedFactors.has(factor.id)).length;
  $("factor-select-all").checked = rows.length > 0 && selectedVisible === rows.length;
  $("factor-select-all").indeterminate = selectedVisible > 0 && selectedVisible < rows.length;
  const body = $("library-body");
  body.replaceChildren();
  if (!rows.length) {
    const row = element("tr"), cell = element("td", "table-empty", state.factors.length ? "没有匹配的因子。" : "暂无因子。完成研究或导入 JSON 后显示。");
    cell.colSpan = 6; row.append(cell); body.append(row);
  }
  rows.forEach((factor) => {
    const row = element("tr", factor.id === state.libraryDetailId ? "selected" : "");
    const selectCell = element("td"), checkbox = element("input");
    checkbox.type = "checkbox"; checkbox.checked = state.selectedFactors.has(factor.id);
    checkbox.setAttribute("aria-label", `选择 ${factor.name || factor.id}`);
    checkbox.addEventListener("change", () => {checkbox.checked ? state.selectedFactors.add(factor.id) : state.selectedFactors.delete(factor.id); renderLibrary(); renderBacktestSelection();});
    selectCell.append(checkbox);
    const name = element("td"); name.append(element("strong", "", factor.name || factor.id), element("small", "", factor.hypothesis || ""));
    const expression = element("td"); expression.append(element("code", "", factor.expression || "—"));
    const action = element("td"), button = element("button", "button small secondary", "详情");
    button.type = "button"; button.addEventListener("click", () => {state.libraryDetailId = factor.id; renderLibrary(); renderLibraryDetail(factor);}); action.append(button);
    const sourceCell = element("td"), source = factor.provenance?.[0]?.source;
    sourceCell.append(element("strong", "", source?.label || factor.origin || "导入"), element("small", "", factor.run_id || "未回测"));
    if (source?.kind === "synthetic" || source?.kind === "demo") sourceCell.append(element("span", "status rejected", "合成数据"));
    row.append(selectCell, name, expression, element("td", "", number(factor.score, 3)), sourceCell, action);
    body.append(row);
  });
  const detail = state.factors.find((factor) => factor.id === state.libraryDetailId);
  if (detail) renderLibraryDetail(detail);
}

function renderLibraryDetail(factor) {
  const target = $("library-detail");
  target.replaceChildren(element("h3", "", factor.name || factor.id), element("p", "", factor.hypothesis || "未提供假设"), element("code", "", factor.expression || "—"));
  const metadata = element("div", "detail-row");
  metadata.append(element("span", "", `来源：${factor.origin || "导入"}`), element("span", "", `开发期评分：${number(factor.score, 3)}`), element("span", "", `开发期夏普：${number(factor.metrics?.sharpe)}`));
  if (Array.isArray(factor.parents) && factor.parents.length) metadata.append(element("span", "", `父候选：${factor.parents.join("、")}`));
  target.append(metadata);
  const sources = Array.isArray(factor.provenance) ? factor.provenance : [];
  const source = sources[0]?.source, split = sources[0]?.split;
  if (source) target.append(element("p", "muted small", `验证数据：${source.label || source.kind || "—"}${["synthetic", "demo"].includes(source.kind) ? " · 合成数据" : ""}${split?.development_start ? ` · 开发期 ${split.development_start} — ${split.development_end}` : ""}`));
  const runIds = [...new Set([factor.run_id, ...sources.map((source) => source.run_id)].filter(Boolean))];
  if (runIds.length) {
    const links = element("div", "toolbar-actions provenance-links");
    runIds.forEach((id) => {const button = element("button", "button small secondary", `查看实验 ${id}`); button.type = "button"; button.addEventListener("click", () => {showPage("research"); selectRun(id);}); links.append(button);});
    target.append(links);
  }
  target.append(element("p", "muted small", `入库：${formatDate(factor.created_at, true)} · ${factor.id}`));
}

async function importFactors(file) {
  if (!file) return;
  if (file.size > 250000) {libraryMessage("JSON 文件不能超过 250 KB；请分批导入。", true); return;}
  $("library-import").disabled = true;
  libraryMessage("正在导入并验证表达式…");
  try {
    const parsed = JSON.parse(await file.text());
    const factors = Array.isArray(parsed) ? parsed : parsed.factors;
    if (!Array.isArray(factors) || !factors.length) throw new Error("需要因子数组，或包含 factors 数组的 JSON 对象。");
    const result = await api("/api/factors/import", {method: "POST", body: JSON.stringify({factors})});
    await loadFactors();
    libraryMessage(`导入完成：${result.imported ?? result.factors?.length ?? "—"} 个新增，${result.duplicates ?? 0} 个重复。`);
  } catch (error) {libraryMessage(`导入失败：${error.message}`, true);}
  finally {$("library-import").disabled = false; $("factor-file").value = "";}
}

function exportFactors() {
  const factors = state.selectedFactors.size ? state.factors.filter((factor) => state.selectedFactors.has(factor.id)) : state.factors;
  if (!factors.length) return;
  const content = {factors: factors.map((factor) => ({name: factor.name, expression: factor.expression, hypothesis: factor.hypothesis || "", format: "alphaos"}))};
  const url = URL.createObjectURL(new Blob([JSON.stringify(content, null, 2)], {type: "application/json"}));
  const link = element("a"); link.href = url; link.download = "alpharesearchos-factors.json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderResearchSeeds() {
  const target = $("research-seeds"); target.replaceChildren(); target.hidden = !state.seedFactorIds.length;
  if (!state.seedFactorIds.length) return;
  target.append(element("p", "small", `已选 ${state.seedFactorIds.length} 个初始因子`));
  const clear = element("button", "button small secondary", "清除"); clear.type = "button";
  clear.addEventListener("click", () => {state.seedFactorIds = []; renderResearchSeeds();}); target.append(clear);
}

function renderBacktestSelection() {
  const target = $("backtest-factors"); target.replaceChildren();
  const selected = state.factors.filter((factor) => state.selectedFactors.has(factor.id));
  if (!selected.length) target.append(element("p", "muted small", "请从因子库选择 1–10 个因子。"));
  selected.forEach((factor) => {
    const chip = element("button", "factor-chip", `${factor.name || factor.id} ×`);
    chip.type = "button"; chip.title = `移除 ${factor.name || factor.id}`;
    chip.addEventListener("click", () => {state.selectedFactors.delete(factor.id); renderBacktestSelection(); renderLibrary();}); target.append(chip);
  });
  if (selected.length > 10) target.append(element("p", "negative small", "单次回测最多选择 10 个因子。"));
  $("start-backtest").disabled = !selected.length || selected.length > 10 || !$("backtest-dataset").value || state.backtestSubmitting || Boolean(state.importingDataset);
}

function renderBacktestDataset() {
  const dataset = state.datasets.find((item) => item.id === $("backtest-dataset").value);
  $("backtest-dataset-note").textContent = dataset ? `${dataset.assets ?? "—"} 个资产 · ${dataset.rows ?? "—"} 个交易日 · ${dataset.start || "—"} — ${dataset.end || "—"}` : "导入行情 CSV 后选择回测数据。";
  for (const id of ["backtest-start", "backtest-end"]) {
    $(id).min = dataset?.start ? String(dataset.start).slice(0, 10) : "";
    $(id).max = dataset?.end ? String(dataset.end).slice(0, 10) : "";
  }
  renderBacktestSelection();
}

async function refreshBacktests() {
  const result = await api("/api/backtests");
  state.backtests = Array.isArray(result.backtests) ? result.backtests : [];
  const options = [element("option", "", "新建回测")]; options[0].value = "";
  state.backtests.forEach((report) => {const option = element("option", "", `${report.id} · ${statusText(report.status)}`); option.value = report.id; options.push(option);});
  $("backtest-history").replaceChildren(...options);
  $("backtest-history").value = state.selectedBacktestId || "";
}

async function loadBacktest(id) {
  if (!id) {state.selectedBacktestId = null; state.backtestReport = null; renderBacktest({}); return;}
  const newSelection = state.selectedBacktestId !== id;
  state.selectedBacktestId = id;
  const report = await api(`/api/backtests/${encodeURIComponent(id)}`);
  if (id !== state.selectedBacktestId) return;
  const previous = state.backtestReport;
  state.backtestReport = report;
  if (newSelection) {
    const config = report.config || {};
    $("backtest-dataset").value = config.dataset || "";
    $("backtest-start").value = config.start_date || "";
    $("backtest-end").value = config.end_date || "";
    for (const [field, input] of [["cost_bps", "backtest-cost"], ["top_k", "backtest-top-k"], ["rebalance_every", "backtest-rebalance"]]) {
      $(input).value = numeric(config[field]) ? config[field] : "";
    }
    const knownIds = new Set(state.factors.map((factor) => factor.id));
    for (const factor of Array.isArray(report.factors) ? report.factors : []) {
      if (factor.id && !knownIds.has(factor.id)) {state.factors.push(factor); knownIds.add(factor.id);}
    }
    state.selectedFactors = new Set(Array.isArray(config.factor_ids) ? config.factor_ids : []);
    renderBacktestSelection();
    renderBacktestDataset();
  }
  renderBacktest(report);
  if (previous?.id === id && previous.status !== report.status && terminal(report.status)) await refreshBacktests();
}

async function startBacktest(event) {
  event.preventDefault();
  if (state.backtestSubmitting || state.selectedFactors.size < 1 || state.selectedFactors.size > 10) return;
  const form = new FormData($("backtest-form"));
  if (!state.datasets.some((dataset) => dataset.id === form.get("dataset"))) {showError("请先导入并选择 CSV 数据集。"); return;}
  const body = {factor_ids: [...state.selectedFactors], dataset: form.get("dataset"), combination: "equal_rank"};
  ["cost_bps", "top_k", "rebalance_every"].forEach((key) => {body[key] = Number(form.get(key));});
  ["start_date", "end_date"].forEach((key) => {if (form.get(key)) body[key] = form.get(key);});
  if (body.start_date && body.end_date && body.start_date > body.end_date) {showError("开始日期不能晚于结束日期。"); return;}
  state.backtestSubmitting = true; renderBacktestSelection(); $("start-backtest").textContent = "正在创建回测…"; showError("");
  try {
    const result = await api("/api/backtests", {method: "POST", body: JSON.stringify(body)});
    state.backtestReport = null; state.selectedBacktestId = result.id;
    renderBacktest({id: result.id, status: result.status});
    await loadBacktest(result.id); await refreshBacktests();
  } catch (error) {showError(`回测失败：${error.message}`);}
  finally {state.backtestSubmitting = false; renderBacktestSelection(); $("start-backtest").textContent = "开始回测 ↗";}
}

function renderBacktest(report) {
  $("backtest-title").textContent = report.id || "尚无回测结果";
  $("backtest-status").textContent = statusText(report.status);
  $("backtest-status").className = `status ${statusClass(report.status)}`;
  const period = report.period || {}, metrics = report.metrics || {}, config = report.config || {};
  const sourceLabel = report.source?.label || config.dataset;
  const periodSummary = period.start ? `${period.start} → ${period.end} · ${period.sessions ?? "—"} 个交易日 · 单边成本 ${config.cost_bps ?? "—"} bps` : report.id ? "正在加载数据并执行固定因子回测…" : "从因子库选择因子后开始回测。";
  $("backtest-period").textContent = sourceLabel ? `${sourceLabel} · ${periodSummary}` : periodSummary;
  if (report.source?.kind === "synthetic" || config.dataset === "demo") $("backtest-period").append(element("span", "status rejected inline-badge", "合成数据"));
  updateMetric("bt-return", metrics.total_return, percent, true);
  updateMetric("bt-sharpe", metrics.sharpe, number);
  updateMetric("bt-drawdown", metrics.max_drawdown, percent);
  updateMetric("bt-excess", metrics.excess_return, percent, true);
  renderWarnings("backtest-warnings", report.warnings, report.error);
  renderChart(report.curve || [], "backtest-chart");
  renderChart(report.curve || [], "backtest-drawdown-chart", true);
  $("backtest-factor-summary").textContent = Array.isArray(report.factors) ? report.factors.map((factor) => factor.name || factor.id).join(" · ") : "";
  const artifacts = $("backtest-artifacts"); artifacts.replaceChildren();
  if (report.status !== "completed") {artifacts.append(element("span", "muted small", "回测完成后可导出")); return;}
  [["report.json", "报告 JSON"], ["curve.csv", "净值 CSV"], ["factors.json", "因子 JSON"], ["config.json", "配置 JSON"]].filter(([file]) => !Array.isArray(report.artifacts) || report.artifacts.includes(file)).forEach(([file, name]) => {
    const link = element("a", "button secondary", `↓ ${name}`); link.href = `/artifacts/backtests/${encodeURIComponent(report.id)}/${file}`; link.download = file; artifacts.append(link);
  });
}

function settingsMessage(message, isError = false) {
  $("settings-message").textContent = message;
  $("settings-message").className = `settings-message ${isError ? "negative" : "muted"}`;
}

function renderProviderFields() {
  const isCodex = $("settings-provider").value === "codex_cli";
  $("settings-api-fields").hidden = isCodex;
  $("settings-codex-fields").hidden = !isCodex;
  $("settings-codex-status").hidden = !isCodex;
  $("settings-api-fields").querySelectorAll("input,select").forEach((input) => {input.disabled = isCodex;});
  $("settings-url").required = !isCodex;
  $("settings-model").required = !isCodex;
  $("settings-key").disabled = isCodex || $("settings-clear-key").checked;
  $("settings-codex-model").disabled = !isCodex;
  $("settings-test").disabled = !state.settings?.configured || Boolean(state.settingsDirty);
  const target = $("settings-codex-status"); target.replaceChildren();
  const codex = state.settings?.codex || {};
  if (isCodex) {
    const detected = Boolean(codex.available), authenticated = Boolean(codex.authenticated);
    const badge = element("span", `status ${detected && authenticated ? "ok" : "rejected"}`, !detected ? "未检测到 Codex" : authenticated ? "已登录" : "尚未登录");
    target.append(badge);
    if (codex.version) target.append(element("p", "muted small", `版本：${codex.version}`));
    if (codex.login_method) target.append(element("p", "muted small", `登录方式：${codex.login_method}`));
    if (codex.default_model) target.append(element("p", "muted small", `当前默认模型：${codex.default_model}`));
    if (!detected) target.append(element("p", "muted small", "安装 Codex 并确保研究服务能够找到命令后，点击重新读取。"));
    else if (!authenticated) target.append(element("p", "muted small", "在本机完成 Codex 登录后，点击重新读取。"));
  }
}

function applySettings(settings) {
  state.settings = settings;
  state.settingsDirty = false;
  state.llmConfigured = Boolean(settings.configured); updateModelStatus();
  $("settings-provider").value = settings.provider || "openai_compatible";
  $("settings-codex-model").value = settings.codex_model || "";
  $("settings-codex-model").placeholder = settings.codex?.default_model ? `留空使用 ${settings.codex.default_model}` : "使用本机 Codex 当前模型";
  $("settings-url").value = settings.base_url || "";
  $("settings-model").value = settings.model || "";
  $("settings-key").value = ""; $("settings-key").disabled = false; $("settings-clear-key").checked = false;
  $("settings-key-status").textContent = settings.api_key_set ? "已保存" : "未设置";
  $("settings-token-field").value = settings.token_field || "max_completion_tokens";
  $("settings-temperature").value = numeric(settings.temperature) ? settings.temperature : "";
  $("settings-status").textContent = settings.provider === "codex_cli" ? settings.configured ? "已就绪" : settings.codex?.available ? "待登录" : "未检测到" : settings.configured ? "已配置" : "待配置";
  $("settings-status").className = `status ${settings.configured ? "ok" : "neutral"}`;
  $("settings-source").textContent = settings.source === "environment" ? "当前使用环境变量配置" : "当前使用本地保存的配置";
  $("settings-test").disabled = !settings.configured;
  renderProviderFields();
}

async function loadSettings() {
  const settings = await api("/api/settings"); applySettings(settings);
  settingsMessage(settings.provider === "codex_cli" ? settings.configured ? "本机 Codex 已就绪。点击测试连接可检查模型响应。" : "检查本机 Codex 安装与登录状态。" : settings.configured ? "已保存模型连接。点击测试连接可检查响应。" : "填写 API 地址、模型名称与密钥后保存。");
}

async function saveSettings(event) {
  event.preventDefault();
  const form = new FormData($("settings-form"));
  const provider = form.get("provider");
  const payload = provider === "codex_cli" ? {provider, codex_model: String(form.get("codex_model") || "").trim()} : {provider, base_url: String(form.get("base_url") || "").trim(), model: String(form.get("model") || "").trim(), api_key: String(form.get("api_key") || ""), clear_api_key: $("settings-clear-key").checked, token_field: form.get("token_field"), temperature: form.get("temperature") === "" ? null : Number(form.get("temperature"))};
  $("settings-save").disabled = true;
  try {applySettings(await api("/api/settings", {method: "POST", body: JSON.stringify(payload)})); settingsMessage("配置已保存。");}
  catch (error) {settingsMessage(`保存失败：${error.message}`, true);}
  finally {$("settings-save").disabled = false;}
}

async function testSettings() {
  if (!state.settings?.configured || state.settingsDirty) return;
  $("settings-test").disabled = true; $("settings-test").textContent = "正在测试…";
  settingsMessage("正在向已保存的模型发送连接测试请求…");
  try {
    const result = await api("/api/settings/test", {method: "POST", body: "{}", timeoutMs: state.settings.provider === "codex_cli" ? 105000 : 20000});
    settingsMessage(`${result.message || (result.ok ? "连接成功" : "连接失败")}${numeric(result.latency_ms) ? ` · ${number(result.latency_ms, 0)} ms` : ""}`, !result.ok);
    $("settings-status").textContent = result.ok ? "连接正常" : "连接失败";
    $("settings-status").className = `status ${result.ok ? "ok" : "failed"}`;
  } catch (error) {settingsMessage(`连接测试失败：${error.message}`, true); $("settings-status").textContent = "连接失败"; $("settings-status").className = "status failed";}
  finally {$("settings-test").disabled = !state.settings?.configured || Boolean(state.settingsDirty); $("settings-test").textContent = "测试连接";}
}

function initializeProduct() {
  ["max-llm-tokens", "max-llm-calls"].forEach((id) => $(id).addEventListener("input", () => {$(id).dataset.userEdited = "true"; renderModeNote();}));
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(redrawVisibleCharts, 120);
  });
  document.querySelectorAll("[data-page]").forEach((button) => button.addEventListener("click", () => showPage(button.dataset.page)));
  const directions = {动量: "寻找跨资产中期动量因子，结合成交量确认，控制换手和交易成本。", 反转: "研究短期价格反转因子，检验不同市场阶段的稳定性，控制交易成本。", 低波动: "寻找低波动与风险调整动量因子，降低回撤和换手。", 量价: "研究价格趋势与成交量变化的组合因子，检验滚动验证期的稳定性。"};
  document.querySelectorAll("[data-direction]").forEach((button) => button.addEventListener("click", () => {$("direction").value = directions[button.dataset.direction]; $("direction").focus();}));
  $("library-refresh").addEventListener("click", () => loadFactors().then(() => libraryMessage("")).catch((error) => libraryMessage(error.message, true)));
  $("library-import").addEventListener("click", () => $("factor-file").click());
  $("factor-file").addEventListener("change", () => importFactors($("factor-file").files[0]));
  $("library-export").addEventListener("click", exportFactors);
  $("factor-search").addEventListener("input", renderLibrary); $("factor-sort").addEventListener("change", renderLibrary);
  $("factor-select-all").addEventListener("change", (event) => {filteredFactors().forEach((factor) => {event.target.checked ? state.selectedFactors.add(factor.id) : state.selectedFactors.delete(factor.id);}); renderLibrary(); renderBacktestSelection();});
  $("factor-clear").addEventListener("click", () => {state.selectedFactors.clear(); renderLibrary(); renderBacktestSelection();});
  $("factor-to-backtest").addEventListener("click", () => showPage("backtest"));
  $("factor-to-research").addEventListener("click", () => {state.seedFactorIds = [...state.selectedFactors]; $("trials").value = Math.max(Number($("trials").value), state.seedFactorIds.length); renderResearchSeeds(); showPage("research"); $("direction").focus();});
  $("backtest-form").addEventListener("submit", startBacktest);
  $("backtest-dataset").addEventListener("change", renderBacktestDataset);
  $("backtest-history").addEventListener("change", () => loadBacktest($("backtest-history").value).catch((error) => showError(`读取回测失败：${error.message}`)));
  $("settings-form").addEventListener("submit", saveSettings);
  $("settings-form").addEventListener("input", () => {state.settingsDirty = true; $("settings-test").disabled = true;});
  $("settings-provider").addEventListener("change", () => {state.settingsDirty = true; renderProviderFields(); settingsMessage("提供方已切换，请保存配置后测试或开始研究。");});
  $("settings-refresh").addEventListener("click", () => loadSettings().catch((error) => settingsMessage(error.message, true)));
  $("settings-test").addEventListener("click", testSettings);
  $("settings-clear-key").addEventListener("change", () => {renderProviderFields(); if ($("settings-key").disabled) $("settings-key").value = "";});
  renderBacktest({});
}

initialize().catch((error) => showError(`初始化失败：${error.message}`));
