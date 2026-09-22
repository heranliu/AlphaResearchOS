"""Command line entry points, also available as python -m alpharesearchos."""

import argparse
import json
import sys
from pathlib import Path

from .backtest import BacktestConfig, backtest, summarize
from .config import ResearchConfig
from .data import download_yahoo, load_csv
from .engine import create_run, run_research, verify_run
from .factorminer_adapter import from_factorminer
from .factors import evaluate_expression
from .store import atomic_json, read_json


def parser():
    root = argparse.ArgumentParser(prog="alphaos", description="可复现的自动量化研究工作台")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="对导入的 CSV 运行自动研究")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["agent", "local", "llm"], default="agent")
    p.add_argument("--direction", default=ResearchConfig.direction)
    p.add_argument("--cost-bps", type=float, default=10)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--rebalance-every", type=int, default=5)
    p.add_argument("--max-seconds", type=int, default=900)
    p.add_argument("--max-llm-calls", type=int, default=12)
    p.add_argument("--max-llm-tokens", type=int)
    p.add_argument("--candidates", type=Path, help="额外候选 JSON，支持本地 DSL 或 FactorMiner formula")
    p.add_argument("--stop-after", type=int, help="保存检查点并退出，用于断点恢复")
    p = commands.add_parser("resume", help="恢复同一数据、代码和预算的实验")
    p.add_argument("run_dir", type=Path)
    p = commands.add_parser("replay", help="重算已冻结因子的最终测试，比较原指标")
    p.add_argument("run_dir", type=Path)
    p = commands.add_parser("fetch", help="下载 Yahoo Finance 复权日线")
    p.add_argument("--symbols", default="SPY,QQQ,IWM,EFA,EEM,TLT,GLD,VNQ")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", required=True, help="结束日期（不含），YYYY-MM-DD")
    p.add_argument("--out", type=Path, default=Path("datasets/etf.csv"))
    p = commands.add_parser("serve", help="启动本地交互界面")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--datasets", type=Path, default=Path("datasets"))
    p.add_argument("--port", type=int, default=8765)
    p = commands.add_parser("import-factors", help="转换 FactorMiner 公式到受限因子 DSL")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, default=Path("candidates.json"))
    return root


def load_candidates(path):
    values = read_json(path)
    if isinstance(values, dict):
        values = values.get("factors", values.get("candidates"))
    if not isinstance(values, list) or len(values) > 200:
        raise ValueError("Candidate file must be a list of at most 200 objects")
    result = []
    for i, item in enumerate(values):
        if isinstance(item, str):
            item = {"formula": item}
        if not isinstance(item, dict):
            raise ValueError("Candidate must be an object or FactorMiner formula string")
        expression = from_factorminer(item["formula"]) if "formula" in item else item["expression"]
        from .factors import validate_expression
        validate_expression(expression)
        result.append({"name": str(item.get("name", f"导入因子 {i + 1}"))[:100], "expression": expression,
                       "hypothesis": str(item.get("hypothesis", "外部候选，按统一实验协议重评"))[:1000],
                       "origin": "factorminer_import" if "formula" in item else "dsl_import", "parents": []})
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            if args.max_llm_tokens is None:
                args.max_llm_tokens = 786432 if args.mode == "agent" else 24000
            keys = ["direction", "mode", "trials", "seed", "cost_bps", "top_k", "rebalance_every", "max_seconds", "max_llm_calls", "max_llm_tokens"]
            config = ResearchConfig(**{k: getattr(args, k) for k in keys}, dataset=str(args.csv))
            if config.mode == "agent":
                from dataclasses import replace
                config = replace(config, warmup=252, max_complexity=150)
            candidates = load_candidates(args.candidates) if args.candidates else None
            from .model_settings import SettingsStore
            from .proposals import provider_context
            with provider_context(SettingsStore(args.runs.resolve().parent / ".alphaos").resolve()):
                directory = create_run(args.runs, config, args.csv, candidates)
                print(f"Run: {directory.resolve()}", flush=True)
                report = run_research(directory, stop_after=args.stop_after)
            print(json.dumps({"status": report["status"], "trials": len(report["trials"]),
                              "selected": (report.get("selected") or {}).get("expression"),
                              "holdout": (report.get("holdout") or {}).get("metrics"),
                              "report": str((directory / "report.html").resolve())}, ensure_ascii=False, indent=2))
            return 1 if report["status"] == "failed" else 0
        if args.command == "resume":
            from .model_settings import SettingsStore
            from .proposals import provider_context
            with provider_context(SettingsStore(args.run_dir.resolve().parent.parent / ".alphaos").resolve()):
                report = run_research(args.run_dir)
            print(f"{report['id']}: {report['status']}")
            return 1 if report["status"] == "failed" else 0
        if args.command == "replay":
            integrity = verify_run(args.run_dir)
            if not integrity["ok"]:
                raise ValueError("Saved experiment files have changed; restore the original files to replay")
            report = read_json(args.run_dir / "report.json")
            if not report.get("holdout"):
                raise ValueError("Experiment has no final holdout result")
            panel = load_csv(args.run_dir / "snapshot.csv")
            if report["config"]["mode"] == "agent":
                from .predictive import strategy_scores
                scores, _ = strategy_scores(report["selected"], panel, fit_cutoff=report["_state"]["development_end"])
            else:
                scores = evaluate_expression(report["selected"]["expression"], panel)
            config = report["config"]
            bt = BacktestConfig(**{k: config[k] for k in ["cost_bps", "top_k", "rebalance_every"]})
            curve = backtest(scores, panel["close"], bt, report["_state"]["holdout_start"], len(panel["close"]))
            metrics = summarize(curve)
            mismatch = {key: [value, report["holdout"]["metrics"].get(key)] for key, value in metrics.items()
                        if abs(value - report["holdout"]["metrics"][key]) > 1e-10}
            print(json.dumps({"ok": not mismatch, "metrics": metrics, "mismatch": mismatch}, indent=2))
            return 1 if mismatch else 0
        if args.command == "fetch":
            print(json.dumps(download_yahoo([s.strip().upper() for s in args.symbols.split(",") if s.strip()], args.start, args.end, args.out), indent=2))
        if args.command == "serve":
            from .server import serve
            serve(args.runs, args.datasets, args.port)
        if args.command == "import-factors":
            values = load_candidates(args.input)
            atomic_json(args.out, values)
            print(f"Converted {len(values)} factors -> {args.out}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"alphaos: {exc}", file=sys.stderr)
        return 2


def entrypoint():
    raise SystemExit(main())
