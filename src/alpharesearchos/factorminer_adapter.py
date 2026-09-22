"""Execute the parity subset on the pinned, locally patched FactorMiner core.

Only documented equivalent operators cross this boundary. Unsupported formulas
are reported as skipped; the adapter never falls back while claiming a second
engine verified the result. No generated Python code is executed.
"""
from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd

from .vendor.factorminer.expression_tree import ConstantNode, LeafNode, OperatorNode
from .vendor.factorminer.parser import parse

UPSTREAM_SHA = "e21171dbd28b0986597f708816fe354750eb7497"
UPSTREAM_URL = "https://github.com/minihellboy/factorminer"
_FIELDS = {"open", "high", "low", "close", "volume"}
_BINARY = {ast.Add: "Add", ast.Sub: "Sub", ast.Mult: "Mul", ast.Div: "Div"}
_SIMPLE = {"rank": "CsRank", "abs": "Abs", "sign": "Sign"}
_WINDOWED = {"mean": "Mean", "std": "Std", "delay": "Delay", "delta": "Delta"}


class UnsupportedFormula(ValueError):
    """The expression is valid locally but outside the verified parity subset."""


def to_factorminer(expression: str) -> str:
    """Validate local DSL and translate its verified subset to FactorMiner DSL."""
    from .factors import validate_expression

    validate_expression(expression)

    def visit(node: ast.AST) -> str:
        if isinstance(node, ast.Name) and node.id in _FIELDS:
            return "$" + node.id
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return repr(node.value)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return f"Neg({visit(node.operand)})"
            if isinstance(node.op, ast.UAdd):
                return visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return f"{_BINARY[type(node.op)]}({visit(node.left)}, {visit(node.right)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in _SIMPLE:
                return f"{_SIMPLE[name]}({visit(node.args[0])})"
            if name in _WINDOWED or name == "ret":
                window = int(ast.literal_eval(node.args[1]))
                lower, upper = (2, 250) if name in {"mean", "std"} else (1, 60)
                if not lower <= window <= upper:
                    raise UnsupportedFormula(
                        f"FactorMiner parity {name} window is {lower}..{upper}; got {window}."
                    )
                child = visit(node.args[0])
                if name == "ret":
                    # Keep the same arithmetic order as the local evaluator.
                    return f"Sub(Div({child}, Delay({child}, {window})), 1)"
                formula = f"{_WINDOWED[name]}({child}, {window})"
                if name == "std":
                    # FactorMiner uses sample std; local DSL uses population std.
                    return f"Mul({formula}, {math.sqrt((window - 1) / window)!r})"
                return formula
            raise UnsupportedFormula(f"No verified FactorMiner parity for operator {name!r}.")
        raise UnsupportedFormula(f"Unsupported AST node {type(node).__name__}.")

    formula = visit(ast.parse(expression, mode="eval").body)
    return parse(formula).to_string()


def from_factorminer(formula: str) -> str:
    """Import a bounded capitalized formula into the local causal DSL.

    ``Std`` is converted to the local population convention with an explicit
    scale multiplier to retain the imported formula's original sample std.
    Operators with different semantics, e.g. ``Log`` and ``CsZScore``, are rejected.
    """
    from .factors import validate_expression

    tree = parse(formula)
    binary = {"Add": "+", "Sub": "-", "Mul": "*", "Div": "/"}
    simple = {"CsRank": "rank", "Abs": "abs", "Sign": "sign"}
    windowed = {"Mean": "mean", "Delay": "delay", "Delta": "delta", "Return": "ret"}

    def visit(node) -> str:
        if isinstance(node, LeafNode):
            field = node.feature_name.removeprefix("$")
            if field not in _FIELDS:
                raise UnsupportedFormula(f"Local panel does not provide feature {field!r}.")
            return field
        if isinstance(node, ConstantNode):
            return repr(node.value)
        if isinstance(node, OperatorNode):
            name = node.operator.name
            args = [visit(child) for child in node.children]
            if name in binary:
                return f"({args[0]} {binary[name]} {args[1]})"
            if name == "Neg":
                return f"(-{args[0]})"
            if name in simple:
                return f"{simple[name]}({args[0]})"
            if name in windowed:
                return f"{windowed[name]}({args[0]}, {int(node.params['window'])})"
            if name == "Std":
                n = int(node.params["window"])
                return f"(std({args[0]}, {n}) * {math.sqrt(n / (n - 1))!r})"
            raise UnsupportedFormula(f"Import of FactorMiner operator {name!r} is unsupported.")
        raise UnsupportedFormula(f"Unsupported FactorMiner node {type(node).__name__}.")

    expression = visit(tree.root)
    validate_expression(expression)
    return expression


def evaluate_factorminer(expression: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate through the actual vendored upstream NumPy tree implementation."""
    formula = to_factorminer(expression)
    template = panel["close"]
    data = {"$" + key: value.to_numpy(dtype=float).T for key, value in panel.items() if key in _FIELDS}
    with np.errstate(all="ignore"):
        values = parse(formula).evaluate(data).T
    values = np.where(np.isfinite(values), values, np.nan)
    return pd.DataFrame(values, index=template.index, columns=template.columns)


def differential_check(
    expression: str,
    panel: dict[str, pd.DataFrame],
    expected: pd.DataFrame | None = None,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Compare two independently implemented evaluators; never hide a mismatch."""
    try:
        formula = to_factorminer(expression)
    except UnsupportedFormula as exc:
        return {"status": "skipped", "passed": None, "reason": str(exc), "upstream_sha": UPSTREAM_SHA}
    if expected is None:
        from .factors import evaluate_expression
        expected = evaluate_expression(expression, panel)
    observed = evaluate_factorminer(expression, panel)
    if not observed.index.equals(expected.index) or not observed.columns.equals(expected.columns):
        return {"status": "failed", "passed": False, "reason": "Factor axes differ.", "formula": formula}
    a, b = expected.to_numpy(dtype=float), observed.to_numpy(dtype=float)
    equal = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    max_error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
    passed = bool(equal.all())
    return {
        "status": "passed" if passed else "failed", "passed": passed,
        "formula": formula, "compared_cells": int(a.size),
        "mismatched_cells": int((~equal).sum()), "max_abs_error": max_error,
        "upstream_sha": UPSTREAM_SHA,
        "reason": "Both evaluators agree." if passed else "Independent evaluator mismatch.",
    }


def provenance() -> dict[str, str]:
    return {"name": "FactorMiner", "url": UPSTREAM_URL, "sha": UPSTREAM_SHA,
            "commit_date": "2026-09-13", "license": "MIT", "integration": "vendored NumPy DSL core with documented patches"}
