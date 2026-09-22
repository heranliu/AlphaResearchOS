"""A persistent expression library, with measured and imported provenance separate.

Research records are discovered from local reports. Manual imports never inherit
unverified performance numbers from the supplied JSON. External formulas are
translated by a bounded AST visitor; generated Python is never imported or run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .factorminer_adapter import from_factorminer
from .factors import FIELDS, MAX_EXPRESSION_LENGTH, MAX_NODES, validate_expression
from .store import atomic_json, clean, read_json, run_lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def factor_id(expression: str) -> str:
    canonical = validate_expression(expression)["canonical"]
    return "F" + hashlib.sha256(canonical.encode()).hexdigest()[:20]


def _external_expression(expression: str, dialect: str) -> str:
    """Translate only explicit equivalent operations; reject all other syntax.

    Qlib Ref is accepted only for positive lags. Its Ref(..., 0), rolling
    minimum-history rules, and time-series Rank are not the local DSL's
    operations and are intentionally unsupported.
    """
    if not isinstance(expression, str) or not 0 < len(expression) <= MAX_EXPRESSION_LENGTH:
        raise ValueError("External expression must contain 1–2048 characters.")

    def field(match):
        name = match.group(1)
        if name not in FIELDS:
            raise ValueError(f"Unsupported data field ${name}; available fields are OHLCV.")
        return name

    replaced = re.sub(r"\$([A-Za-z_][A-Za-z_0-9]*)", field, expression)
    try:
        tree = ast.parse(replaced.strip(), mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("External expression is not supported formula syntax.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ValueError("External expression exceeds the AST node limit.")
    unary = {"Abs": "abs", "Log": "log", "Sign": "sign"} if dialect == "qlib" else {
        "RANK": "rank", "ABS": "abs", "SIGN": "sign",
    }
    windowed = {"Ref": "delay"} if dialect == "qlib" else {"DELAY": "delay", "DELTA": "delta"}
    binary = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}
    operators = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def visit(node: ast.AST, depth: int = 0) -> str:
        if depth > 16:
            raise ValueError("External expression exceeds depth 16.")
        if isinstance(node, ast.Name) and node.id in FIELDS:
            return node.id
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return repr(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            symbol = "-" if isinstance(node.op, ast.USub) else "+"
            return f"({symbol}{visit(node.operand, depth + 1)})"
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return f"({visit(node.left, depth + 1)} {operators[type(node.op)]} {visit(node.right, depth + 1)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name in unary and len(node.args) == 1:
                return f"{unary[name]}({visit(node.args[0], depth + 1)})"
            if name in windowed and len(node.args) == 2:
                lag = node.args[1]
                if not isinstance(lag, ast.Constant) or type(lag.value) is not int or not 1 <= lag.value <= 504:
                    raise ValueError(f"{dialect} {name} import requires a positive integer lag <= 504.")
                return f"{windowed[name]}({visit(node.args[0], depth + 1)}, {lag.value})"
            if dialect == "qlib" and name in binary and len(node.args) == 2:
                return f"({visit(node.args[0], depth + 1)} {binary[name]} {visit(node.args[1], depth + 1)})"
            raise ValueError(
                f"Unsupported {dialect} operator or arity: {name}. "
                "Only the explicitly equivalent causal subset can be imported; "
                "rolling warmup/statistics and ranking semantics are not guessed."
            )
        raise ValueError(f"Unsupported {dialect} expression element: {type(node).__name__}.")

    return validate_expression(visit(tree.body))["canonical"]


def _items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "factors" in payload:
            values = payload["factors"]
        elif "factor_library" in payload:
            values = payload["factor_library"]
        else:
            raise ValueError("Import JSON requires a factors list (or QuantaAlpha factor_library).")
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError("Import payload must be a JSON object or list.")
    if isinstance(values, dict):
        # Some QuantaAlpha exports key records by factor name.
        values = [dict(value, name=value.get("name", value.get("factor_name", value.get("factorName", str(key))))) if isinstance(value, dict) else value
                  for key, value in values.items()]
    if not isinstance(values, list) or not 1 <= len(values) <= 50:
        raise ValueError("Import requires 1–50 factors per request.")
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("Each imported factor must be a JSON object containing an expression.")
    return values


def _normalize_import(item: dict[str, Any]) -> dict[str, Any]:
    external_keys = {"factor_expression", "factorExpression"}
    expression = item.get("expression", item.get("factor_expression", item.get("factorExpression", item.get("formula"))))
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Each imported factor needs an expression; executable code is not supported.")
    dialect = item.get("format")
    if dialect is None:
        if "formula" in item and "expression" not in item:
            dialect = "factorminer"
        elif external_keys.intersection(item):
            # Export record keys do not prove a dialect. Recognize explicit
            # operator spelling; ordinary local DSL remains valid in exports.
            if re.search(r"\b(?:Ref|Mean|Std|Rank|Abs|Log)\s*\(", expression):
                dialect = "qlib"
            elif re.search(r"\b[A-Z][A-Z_]+\s*\(", expression) or "$" in expression:
                dialect = "quantaalpha"
            else:
                dialect = "alphaos"
        else:
            dialect = "alphaos"
    if not isinstance(dialect, str):
        raise ValueError("format must be a string naming a supported expression dialect.")
    if dialect == "factorminer":
        converted = from_factorminer(expression)
    elif dialect in {"qlib", "quantaalpha"}:
        converted = _external_expression(expression, dialect)
    elif dialect == "alphaos":
        converted = expression
    else:
        raise ValueError("format must be alphaos, factorminer, qlib, or quantaalpha.")
    info = validate_expression(converted)
    name = item.get("name", item.get("factor_name", item.get("factorName", "Imported factor")))
    hypothesis = item.get("hypothesis", item.get("factor_description", item.get("factorDescription", "Manually imported expression")))
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 160:
        raise ValueError("Factor name must contain 1–160 characters.")
    if not isinstance(hypothesis, str) or len(hypothesis) > 2000:
        raise ValueError("Factor hypothesis must be text up to 2000 characters.")
    return {
        "id": factor_id(info["canonical"]), "name": name.strip(), "expression": info["canonical"],
        "hypothesis": hypothesis, "origin": "manual_import", "format": dialect,
        "original_expression": expression, "run_id": None, "trial_id": None,
        "score": None, "metrics": {}, "metrics_scope": None, "parents": [], "created_at": _now(),
        "lookback": info["lookback"], "complexity": info["complexity"], "provenance": [],
    }


class FactorLibrary:
    """Discover measured trials and atomically persist manually imported factors."""

    def __init__(self, runs_root: Path, state_root: Path):
        self.runs_root = Path(runs_root)
        self.directory = Path(state_root) / "factor-library"
        self.path = self.directory / "imports.json"

    def _manual(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.is_symlink():
            raise ValueError("Factor library persistence cannot be a symbolic link.")
        saved = read_json(self.path)
        if not isinstance(saved, dict) or not isinstance(saved.get("factors"), list):
            raise ValueError("Factor library state is malformed.")
        records = []
        for record in saved["factors"]:
            if not isinstance(record, dict) or record.get("id") != factor_id(record.get("expression", "")):
                raise ValueError("Factor library state contains an invalid expression or ID.")
            records.append(record)
        return records

    def list_factors(self) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.runs_root.glob("*/report.json"), reverse=True):
            if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > 20_000_000:
                continue
            try:
                report = read_json(path)
                trials = report.get("trials", [])
                if not isinstance(trials, list):
                    continue
                # A learned model's portfolio return is not a constituent
                # feature's measured return. Expose its features without scores.
                features = []
                for trial in trials:
                    if isinstance(trial, dict) and trial.get("model") and trial.get("status") == "ok":
                        for index, expression in enumerate(trial.get("features", [])):
                            features.append({**trial, "features": None, "model": None, "expression": expression,
                                             "id": f"{trial['id']}:F{index + 1}", "name": f"{trial.get('name', '')} / 特征 {index + 1}",
                                             "origin": "agent_feature", "score": None, "metrics": {}})
                trials = [trial for trial in trials if not isinstance(trial, dict) or not trial.get("model")] + features
                for trial in trials:
                    if not isinstance(trial, dict) or trial.get("status") != "ok":
                        continue
                    audit = trial.get("causality", {})
                    if not isinstance(audit, dict) or audit.get("passed") is False:
                        continue
                    try:
                        info = validate_expression(trial.get("expression", ""))
                    except (ValueError, TypeError):
                        continue
                    identifier = factor_id(info["canonical"])
                    provenance = {
                        "run_id": report.get("id", path.parent.name), "trial_id": trial.get("id"),
                        "created_at": report.get("created_at"), "source": deepcopy(report.get("source", {})),
                        "split": deepcopy(report.get("split", {})),
                    }
                    if identifier in records:
                        records[identifier]["provenance"].append(provenance)
                        continue
                    score = trial.get("score")
                    if type(score) not in {int, float} or not math.isfinite(score):
                        score = None
                    metrics = trial.get("metrics", {})
                    records[identifier] = {
                        "id": identifier, "name": str(trial.get("name", identifier))[:160],
                        "expression": info["canonical"], "hypothesis": str(trial.get("hypothesis", ""))[:2000],
                        "origin": str(trial.get("origin", "research")), "format": "alphaos",
                        "run_id": provenance["run_id"], "trial_id": provenance["trial_id"],
                        "score": score, "metrics": deepcopy(metrics) if isinstance(metrics, dict) else {},
                        "metrics_scope": None if trial.get("origin") == "agent_feature" else "development",
                        "parents": deepcopy(trial.get("parents", [])), "created_at": report.get("created_at"),
                        "lookback": info["lookback"], "complexity": info["complexity"],
                        "provenance": [provenance],
                    }
            except (ValueError, TypeError, KeyError, AttributeError, OSError):
                # An incomplete/unrelated report cannot create library entries.
                continue
        for record in self._manual():
            records.setdefault(record["id"], deepcopy(record))
        return clean(list(records.values()))

    def import_factors(self, payload: Any) -> dict[str, Any]:
        """Validate the entire batch before any mutation; deduplicate canonical DSL."""
        items = _items(payload)
        normalized = [_normalize_import(item) for item in items]
        if len(json.dumps(clean(normalized), ensure_ascii=False)) > 250_000:
            raise ValueError("Imported factor batch exceeds 250 KB.")
        with run_lock(self.directory):
            existing = {record["id"]: record for record in self.list_factors()}
            manual = self._manual()
            imported, returned = 0, []
            for record in normalized:
                if record["id"] not in existing:
                    manual.append(record)
                    existing[record["id"]] = record
                    imported += 1
                returned.append(deepcopy(existing[record["id"]]))
            if len(manual) > 5000:
                raise ValueError("Manual library is limited to 5000 unique factors.")
            atomic_json(self.path, {"version": 1, "factors": manual})
        return {"imported": imported, "duplicates": len(normalized) - imported, "factors": clean(returned)}
