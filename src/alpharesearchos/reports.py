"""Dependency-free, offline reports built exclusively from recorded run results."""

from __future__ import annotations

import html
import json
import math
import numbers
import re
from pathlib import Path
from typing import Any

_CSS = """
:root{color-scheme:light;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:#223047;background:#f2f5f8;font-size:14px}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:36px auto;padding:0 28px 40px}header{background:#101d2f;color:#ecf3fa;border-radius:12px;padding:34px 38px;margin-bottom:22px}.brand{font-size:12px;letter-spacing:2px;color:#64d6b6}h1{font-size:29px;font-weight:600;letter-spacing:-.7px;margin:17px 0 12px}header p{color:#a9bad1;font-size:12px;line-height:1.8;overflow-wrap:anywhere}.tag{display:inline-block;font-size:10px;border:1px solid #3c5b65;color:#80e4c6;padding:5px 9px;border-radius:5px}.panel{background:white;border:1px solid #dce3ec;border-radius:9px;padding:24px 28px;margin-bottom:20px;overflow:hidden}h2{font-size:18px;margin:0 0 17px;font-weight:600}h3{font-size:13px;margin:20px 0 10px}p{line-height:1.85;font-size:12px}.muted,.note{color:#718199;font-size:11px;line-height:1.8}.eyebrow{font-size:9px;letter-spacing:1.5px;color:#78909f;margin-bottom:7px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}.metric{background:white;border:1px solid #dce3ec;border-radius:8px;padding:19px}.metric span{color:#6b7e97;font-size:11px}.metric strong{font:500 27px ui-monospace,SFMono-Regular,monospace;display:block;margin-top:14px;color:#177765}.metric small{display:block;font-size:9px;color:#8390a3;margin-top:7px}.warning{border-left:3px solid #ce9c4d;background:#fff7e8;color:#785a2a;font-size:12px;line-height:1.8;padding:13px 17px;margin-bottom:14px;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>.panel{min-width:0}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:11px;text-align:left}th{background:#f2f5f8;color:#6b7e96;font-size:10px;font-weight:500}td,th{padding:11px 12px;border-bottom:1px solid #e5ebf1;vertical-align:top}td{line-height:1.7;overflow-wrap:anywhere}td code{font-size:10px;white-space:normal}.selected{background:#eef8f5}pre{padding:16px;background:#f0f5f7;border:1px solid #dde6ed;border-radius:5px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px;line-height:1.8;color:#325e65}code{font-family:ui-monospace,SFMono-Regular,monospace;font-size:11px}.chart svg{display:block;width:100%;height:auto}.legend{display:flex;justify-content:flex-end;gap:20px;font-size:10px;color:#73859a;margin-bottom:5px}.legend span:before{content:"";display:inline-block;width:17px;height:3px;background:#178a76;margin:0 7px 3px 0}.legend span+span:before{background:#7e89be}.empty{padding:45px;text-align:center;color:#7c8ea6;font-size:12px}.kv{display:grid;grid-template-columns:120px 1fr;gap:10px;font-size:11px;line-height:1.8}.kv dt{color:#72849c}.kv dd{margin:0;overflow-wrap:anywhere}.links{display:flex;gap:12px;flex-wrap:wrap}a{color:#197c6c;text-decoration:none}.links a{border:1px solid #c9dcd7;padding:8px 13px;border-radius:5px;font-size:11px}details{margin-top:14px}summary{cursor:pointer;color:#607a8e;font-size:11px}footer{font-size:10px;color:#7d8da3;line-height:1.8;padding:5px 0}ul{padding-left:20px;line-height:1.8;font-size:12px}.events td:first-child{white-space:nowrap;color:#7f91a8}.status{font-size:10px;white-space:nowrap}.status-ok{color:#117e63}.status-failed{color:#c1595a}.status-rejected{color:#9b792e}@media(max-width:760px){main{padding:0 14px;margin-top:15px}header{padding:25px}.panel{padding:20px}.metrics{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr;gap:0}.metric{padding:15px}h1{font-size:24px}.kv{grid-template-columns:95px 1fr}}@media print{:root{background:white;font-size:11px}main{max-width:none;margin:0;padding:0}header{border-radius:0;padding:22px;print-color-adjust:exact;-webkit-print-color-adjust:exact}.panel{break-inside:avoid;padding:17px;margin-bottom:14px}.metrics{break-inside:avoid}.metric{padding:14px}.metric strong{font-size:23px}.links{display:none}details{display:none}table{font-size:9px}td,th{padding:7px}h1{font-size:24px}footer{margin-top:20px}}
"""


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else "—"), quote=True)


def _finite(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _number(value: Any, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if _finite(value) else "—"


def _percent(value: Any) -> str:
    return f"{value * 100:.2f}%" if _finite(value) else "—"


def _mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _strategy_text(candidate: dict) -> str:
    features = [value for value in _list(candidate.get("features")) if isinstance(value, str)]
    if not features:
        return str(candidate.get("expression") or "—")
    lines = [f'模型：{candidate.get("model", "未记录")}；特征数：{len(features)}']
    lines.extend(f"{index + 1}. {expression}" for index, expression in enumerate(features))
    return "\n".join(lines)


def _review_record(candidate: dict) -> dict:
    review = _mapping(candidate.get("review"))
    return {key: review[key] for key in ("decision", "critique", "risks", "suggested_action") if key in review}


def _usage_records(trials: list[dict]) -> list[dict]:
    records = []
    for trial in trials:
        proposer = _mapping(trial.get("proposer_usage")) or _mapping(trial.get("llm_usage"))
        for role, usage in [("提案", proposer), ("独立复核", _mapping(trial.get("reviewer_usage")))]:
            if usage:
                records.append({"trial": trial.get("id"), "role": role, "provider": usage.get("provider"),
                                "model": usage.get("model"), "tokens": usage.get("reported_total_tokens"),
                                "seconds": usage.get("seconds")})
    return records


def _status(value: Any) -> str:
    return {"completed": "已完成", "complete": "已完成", "running": "进行中", "queued": "等待执行", "paused": "已暂停", "failed": "执行失败", "interrupted": "已中断", "stopped": "已停止", "cancelled": "已取消", "ok": "通过验证", "evaluated": "已评估，待复核", "rejected": "已拒绝"}.get(str(value), str(value or "待运行"))


def _stop_reason(value: Any) -> str:
    return {"trial_budget": "候选预算已完成", "time_budget": "已达到运行时间预算", "model_budget": "模型预算已用尽", "user_pause": "已按用户请求暂停", "checkpoint": "已保存检查点，可恢复继续研究", "no_valid_candidates": "没有候选通过验证", "no_reviewed_candidates": "没有候选通过独立复核"}.get(str(value), str(value))


def _asset_count(source: dict) -> Any:
    assets = source.get("assets")
    return len(assets) if isinstance(assets, list) else assets


def _display_metadata(config: dict, source: dict, split: dict) -> dict:
    return {"config": config, "data": {key: source[key] for key in
            ("label", "kind", "start", "end", "rows", "assets") if key in source}, "split": split}


def _curve_svg(raw_curve: Any) -> str:
    """Render recorded equity points; never interpolate missing values as performance."""
    curve = [row for row in _list(raw_curve) if isinstance(row, dict) and _finite(row.get("equity")) and _finite(row.get("benchmark_equity"))]
    if not curve:
        return '<div class="empty">尚无留出期净值数据。</div>'
    width, height, left, right, top, bottom = 960, 320, 63, 26, 23, 36
    values = [float(row[field]) for row in curve for field in ("equity", "benchmark_equity")]
    minimum, maximum = min(*values, 1), max(*values, 1)
    padding = max((maximum - minimum) * .16, .012)
    lower, upper = minimum - padding, maximum + padding

    def x(index: int) -> float:
        return left + index / max(1, len(curve) - 1) * (width - left - right)

    def y(value: float) -> float:
        return top + (upper - value) / (upper - lower) * (height - top - bottom)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="留出期策略与等权基准净值曲线"><title>留出期净值，共 {len(curve)} 个交易日</title>']
    for index in range(5):
        value = lower + (upper - lower) * index / 4
        parts.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{width-right}" y2="{y(value):.2f}" stroke="#dce5ed" stroke-dasharray="3 5"/><text x="{left-14}" y="{y(value)+4:.2f}" text-anchor="end" fill="#7b8da4" font-size="11" font-family="monospace">{value:.2f}</text>')
    for index in sorted({0, (len(curve) - 1) // 3, 2 * (len(curve) - 1) // 3, len(curve) - 1}):
        anchor = "start" if index == 0 else "end" if index == len(curve) - 1 else "middle"
        parts.append(f'<text x="{x(index):.2f}" y="{height-10}" text-anchor="{anchor}" fill="#7b8da4" font-size="10" font-family="monospace">{_escape(str(curve[index].get("date", ""))[:10])}</text>')
    for field, color, stroke in [("benchmark_equity", "#7e89be", 2), ("equity", "#178a76", 3)]:
        points = " ".join(f"{x(index):.2f},{y(float(row[field])):.2f}" for index, row in enumerate(curve))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append(f'<circle cx="{x(len(curve)-1):.2f}" cy="{y(float(curve[-1][field])):.2f}" r="3" fill="{color}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _kv(items: list[tuple[str, Any]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{_escape(key)}</dt><dd>{_escape(value)}</dd>" for key, value in items) + "</dl>"


def _table(headers: list[str], rows: list[list[str]], *, css_class: str = "") -> str:
    if not rows:
        return '<p class="muted">暂无记录。</p>'
    return f'<div class="table-wrap"><table class="{css_class}"><thead><tr>' + "".join(f"<th>{_escape(header)}</th>" for header in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def _html_report(report: dict) -> str:
    source = _mapping(report.get("source"))
    config = _mapping(report.get("config"))
    progress = _mapping(report.get("progress"))
    split = _mapping(report.get("split"))
    selected = _mapping(report.get("selected"))
    holdout = _mapping(report.get("holdout"))
    metrics = _mapping(holdout.get("metrics"))
    trials = [item for item in _list(report.get("trials")) if isinstance(item, dict)]
    warnings = list(_list(report.get("warnings")))
    if source.get("kind") in {"synthetic", "demo"} or config.get("dataset") == "demo":
        warnings.insert(0, "本实验使用合成演示数据，仅用于验证研究流程。收益表现不能用于判断真实市场有效性。")
    if report.get("stop_reason") and report["stop_reason"] != "trial_budget":
        warnings.append(_stop_reason(report["stop_reason"]))
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', f'<title>AlphaResearchOS · {_escape(report.get("id", "研究报告"))}</title><style>{_CSS}</style></head><body><main>']
    parts.append(f'<header><div class="brand">α ALPHARESEARCHOS / RESEARCH REPORT</div><h1>自动量化研究 · 实验报告</h1><p>{_escape(config.get("direction", "研究方向未记录"))}</p><span class="tag">{_escape(_status(report.get("status")))}</span><p>实验 {_escape(report.get("id"))} &nbsp; · &nbsp; 创建于 {_escape(report.get("created_at"))}</p></header>')
    parts.extend(f'<div class="warning">{_escape(warning)}</div>' for warning in warnings)
    metric_cards = [("留出期累计收益", _percent(metrics.get("total_return"))), ("年化夏普比率", _number(metrics.get("sharpe"))), ("最大回撤", _percent(metrics.get("max_drawdown"))), ("相对基准收益差", _percent(metrics.get("excess_return")))]
    parts.append('<div class="metrics">' + "".join(f'<div class="metric"><span>{label}</span><strong>{value}</strong><small>留出期 · 已计交易成本</small></div>' for label, value in metric_cards) + "</div>")
    parts.append('<section class="panel"><div class="eyebrow">HOLDOUT PERFORMANCE</div><h2>留出期净值</h2><div class="legend"><span>最终候选</span><span>等权基准</span></div><div class="chart">' + _curve_svg(holdout.get("curve")) + "</div>")
    parts.append(f'<p class="note">留出期：{_escape(split.get("holdout_start"))} → {_escape(split.get("holdout_end"))}。起始资金归一化；累计收益均含成本。留出期用于检验开发期选定的最终候选。</p>')
    interval = _list(holdout.get("sharpe_interval"))
    if len(interval) == 2:
        parts.append(f'<p class="note">夏普比率自助法区间：[{_number(interval[0])}, {_number(interval[1])}]。该区间反映样本估计的不确定性。</p>')
    parts.append("</section><div class=\"grid\"><section class=\"panel\"><div class=\"eyebrow\">EXPERIMENT DESIGN</div><h2>数据与验证设计</h2>")
    parts.append(_kv([("数据来源", source.get("label")), ("来源类型", source.get("kind")), ("样本范围", f'{source.get("start", "—")} → {source.get("end", "—")}'), ("资产 / 交易日", f'{_asset_count(source) if _asset_count(source) is not None else "—"} / {source.get("rows", "—")}'), ("开发期", f'{split.get("development_start", "—")} → {split.get("development_end", "—")}'), ("留出期", f'{split.get("holdout_start", "—")} → {split.get("holdout_end", "—")}'), ("滚动验证窗口", len(_list(split.get("folds"))))]))
    parts.append('</section><section class="panel"><div class="eyebrow">SEARCH CONFIGURATION</div><h2>搜索与交易配置</h2>')
    parts.append(_kv([("提案引擎", config.get("mode")), ("候选预算", config.get("trials", progress.get("total"))), ("已评估候选", progress.get("completed", len(trials))), ("随机种子", config.get("seed")), ("单边成本 / bps", config.get("cost_bps")), ("持仓数量", config.get("top_k")), ("调仓间隔 / 日", config.get("rebalance_every")), ("信号滞后 / 日", config.get("signal_lag", 2)), ("运行耗时 / 秒", _number(progress.get("elapsed_seconds"), 1))]))
    parts.append("</section></div><section class=\"panel\"><div class=\"eyebrow\">SELECTED FACTOR</div><h2>最终候选与研究假设</h2>")
    if selected:
        parts.append(f'<h3>{_escape(selected.get("name", selected.get("id")))}</h3><p>{_escape(selected.get("hypothesis"))}</p><pre>{_escape(_strategy_text(selected))}</pre>')
        if selected.get("features"):
            parts.append(f'<h3>完整模型参数</h3><pre>{_escape(_json(_mapping(selected.get("model_params"))))}</pre>')
        if _review_record(selected):
            parts.append(f'<h3>独立复核记录</h3><pre>{_escape(_json(_review_record(selected)))}</pre>')
        parts.append(_kv([("候选 ID", selected.get("id")), ("提案来源", selected.get("origin")), ("父候选", ", ".join(map(str, _list(selected.get("parents")))) or "初始提案"), ("开发期评分", _number(selected.get("score"), 3)), ("表达式复杂度", selected.get("complexity"))]))
        folds = [fold for fold in _list(selected.get("folds")) if isinstance(fold, dict)]
        if folds:
            parts.append("<h3>开发期滚动验证</h3>" + _table(["窗口", "累计收益", "夏普比率", "最大回撤"], [[str(index + 1), _percent(_mapping(fold.get("metrics", fold)).get("total_return")), _number(_mapping(fold.get("metrics", fold)).get("sharpe")), _percent(_mapping(fold.get("metrics", fold)).get("max_drawdown"))] for index, fold in enumerate(folds)]))
    else:
        parts.append('<p class="muted">本次实验尚未选定最终候选。</p>')
    parts.append("</section><section class=\"panel\"><div class=\"eyebrow\">COST ROBUSTNESS</div><h2>交易成本压力检验</h2>")
    stress_rows = []
    for scenario in _list(holdout.get("stress")):
        if not isinstance(scenario, dict):
            continue
        scenario_metrics = _mapping(scenario.get("metrics"))
        stress_rows.append([_number(scenario.get("cost_bps"), 1), _percent(scenario_metrics.get("total_return")), _number(scenario_metrics.get("sharpe")), _percent(scenario_metrics.get("max_drawdown"))])
    parts.append(_table(["单边成本 / bps", "累计收益", "夏普比率", "最大回撤"], stress_rows))
    parts.append('<p class="note">压力检验用于观察最终候选对成本的敏感性，不参与开发期选优。</p></section><section class="panel"><div class="eyebrow">CANDIDATE ARENA</div><h2>全部候选记录</h2>')
    trial_rows = []
    for trial in trials:
        status = trial.get("status")
        css = status if status in {"ok", "rejected", "failed"} else "unknown"
        name = ("★ " if trial.get("id") == selected.get("id") else "") + str(trial.get("name", trial.get("id", "—")))
        trial_rows.append([f'<strong>{_escape(name)}</strong><br><span class="muted">{_escape(trial.get("id"))}</span>', '<code>' + _escape(_strategy_text(trial)).replace("\n", "<br>") + '</code>', _number(trial.get("score"), 3), _number(_mapping(trial.get("metrics")).get("sharpe")), f'<span class="status status-{css}">{_escape(_status(status))}</span>', _escape(trial.get("reason") or _review_record(trial).get("decision") or "—")])
    parts.append(_table(["候选", "模型与全部特征 / 表达式", "开发期评分", "开发期夏普", "状态", "说明"], trial_rows))
    parts.append('<p class="note">所有候选只通过开发期评分比较；开发期指标与留出期指标分别记录。</p></section><section class="panel"><h2>提案与独立复核用量</h2>')
    usage_rows = [[_escape(item["trial"]), _escape(item["role"]), _escape(item["provider"]), _escape(item["model"]), _number(item["tokens"], 0), _number(item["seconds"], 2)] for item in _usage_records(trials)]
    parts.append(_table(["候选", "角色", "提供方", "模型", "报告 tokens", "请求秒数"], usage_rows))
    budget = _mapping(_mapping(report.get("research")).get("budget"))
    if budget:
        parts.append(_kv([("已计数模型调用", budget.get("llm_calls")), ("已预留 token 预算", budget.get("llm_reserved_tokens"))]))
    parts.append('<p class="note">用量表只展示服务实际返回的记录；缺失值不计为零。预算预留不是计费金额或 CLI 硬上限。</p></section><section class="panel"><div class="eyebrow">AUDIT TRAIL</div><h2>研究日志与可复现记录</h2>')
    event_rows = [[_escape(event.get("time")), _escape(event.get("level")), _escape(event.get("message"))] for event in _list(report.get("events")) if isinstance(event, dict)]
    parts.append(_table(["时间", "级别", "事件"], event_rows, css_class="events"))
    parts.append(f'<details><summary>查看研究配置与数据范围</summary><pre>{_escape(_json(_display_metadata(config, source, split)))}</pre></details>')
    parts.append('<h3>实验产物</h3><div class="links"><a href="report.json">完整报告 JSON</a><a href="trials.csv">候选 CSV</a><a href="report.md">Markdown 报告</a>')
    if selected:
        parts.append('<a href="selected_factor.py">最终策略代码</a>' if selected.get("features") else '<a href="selected_factor.py">最终因子代码</a>')
        if config.get("mode") == "agent":
            parts.append('<a href="selected_candidate.json">完整策略 JSON</a><a href="training_audit.json">训练时点审计</a>')
    if _mapping(report.get("research")).get("graph") is not None:
        parts.append('<a href="research_graph.json">研究轨迹 DAG</a>')
    parts.append('</div><p class="note">本 HTML 报告不加载外部资源，可离线查看。下载整个实验目录即可保留关联产物。</p></section><footer>AlphaResearchOS · 研究、比较与策略分析。</footer></main></body></html>')
    return "".join(parts)


def _md(value: Any) -> str:
    return html.escape(str(value if value is not None else "—")).replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _fence(text: str, language: str = "") -> str:
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _markdown_report(report: dict) -> str:
    config = _mapping(report.get("config"))
    source = _mapping(report.get("source"))
    split = _mapping(report.get("split"))
    selected = _mapping(report.get("selected"))
    holdout = _mapping(report.get("holdout"))
    metrics = _mapping(holdout.get("metrics"))
    lines = ["# AlphaResearchOS 自动量化研究报告", "", f'- 实验：{_md(report.get("id"))}', f'- 状态：{_md(_status(report.get("status")))}', f'- 创建时间：{_md(report.get("created_at"))}', f'- 研究方向：{_md(config.get("direction"))}', ""]
    if source.get("kind") in {"synthetic", "demo"} or config.get("dataset") == "demo":
        lines.extend(["> 本实验使用合成演示数据，仅用于验证研究流程。收益表现不能用于判断真实市场有效性。", ""])
    for warning in _list(report.get("warnings")):
        lines.extend([f"> {_md(warning)}", ""])
    if report.get("stop_reason"):
        lines.extend([f'> 停止原因：{_md(_stop_reason(report["stop_reason"]))}', ""])
    lines.extend(["## 留出期结果", "", "| 指标 | 值 |", "| --- | ---: |"])
    for label, value in [("累计收益", _percent(metrics.get("total_return"))), ("年化收益", _percent(metrics.get("cagr"))), ("夏普比率", _number(metrics.get("sharpe"))), ("最大回撤", _percent(metrics.get("max_drawdown"))), ("年化波动率", _percent(metrics.get("annual_vol"))), ("基准累计收益", _percent(metrics.get("benchmark_total_return"))), ("相对基准收益差", _percent(metrics.get("excess_return"))), ("交易日数", _md(metrics.get("days")))]:
        lines.append(f"| {label} | {value} |")
    lines.extend(["", f'留出期：{_md(split.get("holdout_start"))} → {_md(split.get("holdout_end"))}。指标均基于扣除交易成本后的收益。', "", "净值曲线请查看同目录的 [离线 HTML 报告](report.html)。", "", "## 数据与实验配置", "", "| 配置 | 值 |", "| --- | --- |"])
    for label, value in [("数据来源", source.get("label")), ("来源类型", source.get("kind")), ("资产数", _asset_count(source)), ("交易日数", source.get("rows")), ("开发期起始", split.get("development_start")), ("开发期结束", split.get("development_end")), ("提案引擎", config.get("mode")), ("候选预算", config.get("trials")), ("随机种子", config.get("seed")), ("单边成本 / bps", config.get("cost_bps")), ("持仓数量", config.get("top_k")), ("调仓间隔 / 日", config.get("rebalance_every"))]:
        lines.append(f"| {label} | {_md(value)} |")
    lines.extend(["", "## 最终候选", ""])
    if selected:
        lines.extend([f'**{_md(selected.get("name", selected.get("id")))}**', "", _md(selected.get("hypothesis")), "", _fence(_strategy_text(selected), "text"), "", f'- 来源：{_md(selected.get("origin"))}', f'- 父候选：{_md(", ".join(map(str, _list(selected.get("parents")))) or "初始提案")}', f'- 开发期评分：{_number(selected.get("score"), 3)}', ""])
        if selected.get("features"):
            lines.extend(["### 完整模型参数", "", _fence(_json(_mapping(selected.get("model_params"))), "json"), ""])
        if _review_record(selected):
            lines.extend(["### 独立复核记录", "", _fence(_json(_review_record(selected)), "json"), ""])
    else:
        lines.extend(["本次实验尚未选定最终候选。", ""])
    lines.extend(["## 候选记录", "", "| 候选 | 模型与全部特征 / 表达式 | 开发期评分 | 状态 | 说明 |", "| --- | --- | ---: | --- | --- |"])
    for trial in _list(report.get("trials")):
        if isinstance(trial, dict):
            lines.append(f'| {_md(trial.get("name", trial.get("id")))} | {_md(_strategy_text(trial))} | {_number(trial.get("score"), 3)} | {_md(_status(trial.get("status")))} | {_md(trial.get("reason") or _review_record(trial).get("decision") or "—")} |')
    lines.extend(["", "## 提案与独立复核用量", "", "| 候选 | 角色 | 提供方 | 模型 | 报告 tokens | 请求秒数 |", "| --- | --- | --- | --- | ---: | ---: |"])
    for item in _usage_records([value for value in _list(report.get("trials")) if isinstance(value, dict)]):
        lines.append(f'| {_md(item["trial"])} | {_md(item["role"])} | {_md(item["provider"])} | {_md(item["model"])} | {_number(item["tokens"], 0)} | {_number(item["seconds"], 2)} |')
    budget = _mapping(_mapping(report.get("research")).get("budget"))
    if budget:
        lines.extend(["", f'已计数模型调用：{_md(budget.get("llm_calls"))}；已预留 token 预算：{_md(budget.get("llm_reserved_tokens"))}。'])
    lines.extend(["", "用量只展示服务实际返回的记录；缺失值不计为零，预留量不是账单或 CLI 硬上限。"])
    lines.extend(["", "## 交易成本压力检验", "", "| 单边成本 / bps | 累计收益 | 夏普比率 | 最大回撤 |", "| ---: | ---: | ---: | ---: |"])
    for scenario in _list(holdout.get("stress")):
        if isinstance(scenario, dict):
            scenario_metrics = _mapping(scenario.get("metrics"))
            lines.append(f'| {_number(scenario.get("cost_bps"), 1)} | {_percent(scenario_metrics.get("total_return"))} | {_number(scenario_metrics.get("sharpe"))} | {_percent(scenario_metrics.get("max_drawdown"))} |')
    lines.extend(["", "## 研究日志", ""])
    for event in _list(report.get("events")):
        if isinstance(event, dict):
            lines.append(f'- {_md(event.get("time"))} [{_md(event.get("level"))}] {_md(event.get("message"))}')
    lines.extend(["", "## 研究配置", "", _fence(_json(_display_metadata(config, source, split)), "json"), "", "[完整 JSON](report.json) · [候选 CSV](trials.csv) · [离线 HTML 报告](report.html)", "", "AlphaResearchOS · 研究、比较与策略分析。", ""])
    if selected and config.get("mode") == "agent":
        lines.extend(["[完整策略 JSON](selected_candidate.json) · [训练时点审计](training_audit.json)", ""])
    if _mapping(report.get("research")).get("graph") is not None:
        lines.extend(["[研究轨迹 DAG](research_graph.json)", ""])
    return "\n".join(lines)


def write_report(run_dir: Path, report: dict) -> None:
    """Write portable HTML and Markdown artifacts from a full or partial report.

    Generated HTML contains no scripts, remote fonts, CDNs, or external charts.
    Dynamic text is escaped; metrics absent from a partial report remain absent.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [("report.html", _html_report(report)), ("report.md", _markdown_report(report))]:
        destination = run_dir / name
        temporary = run_dir / f".{name}.tmp"
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)
