"""Small, bounded factor language. Expressions are interpreted, never executed.

Rows are dates and columns are assets. Every time-series operator uses only
the current row and past rows. Rolling windows require a complete window;
cross-sectional rank uses average ties and percent ranks in (0, 1].
"""

from __future__ import annotations

import ast
import math
from typing import Any

import numpy as np
import pandas as pd

FIELDS = frozenset({"open", "high", "low", "close", "volume"})
WINDOW_CALLS = frozenset({"ret", "mean", "std", "delay", "delta"})
UNARY_CALLS = frozenset({"rank", "zscore", "abs", "log", "sign"})
MAX_EXPRESSION_LENGTH = 2048
MAX_NODES = 128
MAX_DEPTH = 16
MAX_WINDOW = 504
MAX_LOOKBACK = 1008


class ExpressionError(ValueError):
    """An expression is outside the permitted factor language."""


def _parse(expression: str) -> tuple[ast.Expression, dict[str, Any]]:
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("Expression must be a nonempty string.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters.")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise ExpressionError("Invalid expression syntax.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise ExpressionError(f"Expression exceeds {MAX_NODES} AST nodes.")

    complexity = 0
    referenced_fields: set[str] = set()

    def visit(node: ast.AST, depth: int = 0) -> int:
        nonlocal complexity
        complexity += 1
        if depth > MAX_DEPTH:
            raise ExpressionError(f"Expression exceeds depth {MAX_DEPTH}.")
        if isinstance(node, ast.Name) and node.id in FIELDS:
            referenced_fields.add(node.id)
            return 0
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float):
                raise ExpressionError("Only finite numeric constants are permitted.")
            if abs(node.value) > 1_000_000 or not math.isfinite(node.value):
                raise ExpressionError("Numeric constants must be finite and at most 1e6 in magnitude.")
            return 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return max(visit(node.left, depth + 1), visit(node.right, depth + 1))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ExpressionError("Only named DSL calls with positional arguments are permitted.")
            name = node.func.id
            if name in UNARY_CALLS:
                if len(node.args) != 1:
                    raise ExpressionError(f"{name} requires one argument.")
                return visit(node.args[0], depth + 1)
            if name in WINDOW_CALLS:
                if len(node.args) != 2:
                    raise ExpressionError(f"{name} requires an expression and an integer window.")
                period = node.args[1]
                minimum = 0 if name in {"delay", "delta"} else 1
                if not isinstance(period, ast.Constant) or type(period.value) is not int:
                    raise ExpressionError("Windows/lags must be literal integers.")
                if not minimum <= period.value <= MAX_WINDOW:
                    raise ExpressionError(f"{name} window must be between {minimum} and {MAX_WINDOW}.")
                lookback = visit(node.args[0], depth + 1)
                complexity += 1
                return lookback + (period.value - 1 if name in {"mean", "std"} else period.value)
        raise ExpressionError(f"Unsupported expression element: {type(node).__name__}.")

    lookback = visit(tree.body)
    if lookback > MAX_LOOKBACK:
        raise ExpressionError(f"Combined lookback exceeds {MAX_LOOKBACK} rows.")
    return tree, {
        "canonical": ast.unparse(tree.body),
        "complexity": complexity,
        "lookback": lookback,
        "fields": sorted(referenced_fields),
    }


def validate_expression(expression: str) -> dict[str, Any]:
    """Validate syntax, work limits and causal lags; return expression metadata."""
    return _parse(expression)[1]


def _validate_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not isinstance(panel, dict) or "close" not in panel:
        raise ValueError("Panel must contain a close DataFrame.")
    close = panel["close"]
    if not isinstance(close, pd.DataFrame) or close.empty:
        raise ValueError("close must be a nonempty DataFrame.")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("Panel requires a DatetimeIndex.")
    if close.index.has_duplicates or not close.index.is_monotonic_increasing or close.index.hasnans:
        raise ValueError("Panel dates must be unique, valid and ascending.")
    if close.columns.has_duplicates:
        raise ValueError("Panel assets must be unique.")
    for field, frame in panel.items():
        if field not in FIELDS:
            continue
        if not isinstance(frame, pd.DataFrame) or not frame.index.equals(close.index) or not frame.columns.equals(close.columns):
            raise ValueError(f"{field} does not align with close.")
        try:
            values = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric.") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"{field} contains missing or nonfinite data; implicit filling is forbidden.")
        if (values < 0).any() or (field != "volume" and (values == 0).any()):
            raise ValueError(f"{field} contains invalid negative/zero values.")
    return close


def evaluate_expression(expression: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Interpret an expression on an aligned, complete OHLCV panel.

    Undefined operations yield NaN, never an invented filled value. A NaN
    signal is ineligible for selection by the portfolio engine.
    """
    tree, metadata = _parse(expression)
    close = _validate_panel(panel)
    missing = set(metadata["fields"]) - panel.keys()
    if missing:
        raise ValueError(f"Panel is missing fields: {', '.join(sorted(missing))}.")

    def as_frame(value: float | pd.DataFrame) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value
        return pd.DataFrame(value, index=close.index, columns=close.columns, dtype=float)

    def divide(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        return left.div(right.where(right != 0))

    def evaluate(node: ast.AST) -> pd.DataFrame:
        if isinstance(node, ast.Name):
            return panel[node.id].astype(float, copy=False)
        if isinstance(node, ast.Constant):
            return as_frame(float(node.value))
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return divide(left, right)
        # _parse permits only Call nodes at this point.
        name = node.func.id
        value = evaluate(node.args[0]).replace([np.inf, -np.inf], np.nan)
        if name in WINDOW_CALLS:
            period = node.args[1].value
            if name == "delay":
                return value.shift(period)
            if name == "delta":
                return value - value.shift(period)
            if name == "ret":
                return divide(value, value.shift(period)) - 1
            rolling = value.rolling(period, min_periods=period)
            return rolling.mean() if name == "mean" else rolling.std(ddof=0)
        if name == "rank":
            return value.rank(axis=1, method="average", pct=True, na_option="keep")
        if name == "zscore":
            centered = value.sub(value.mean(axis=1), axis=0)
            return centered.div(value.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
        if name == "abs":
            return value.abs()
        if name == "sign":
            return np.sign(value)
        return np.log(value.where(value > 0))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = evaluate(tree.body)
    return result.replace([np.inf, -np.inf], np.nan).astype(float)


def causality_check(expression: str, panel: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Audit output invariance under prefix truncation and future mutation.

    These deterministic checks supplement the DSL's structural prohibition on
    future reads. They are not evidence of predictive value.
    """
    try:
        full = evaluate_expression(expression, panel)
        length = len(full)
        if length < 3:
            return {"passed": False, "detail": "At least three dates are required for causality checks."}
        cuts = sorted({max(1, length // 3), max(1, 2 * length // 3), length - 1})
        compared_cells = 0
        for cut in cuts:
            prefix = evaluate_expression(expression, {key: frame.iloc[:cut] for key, frame in panel.items()})
            expected = full.iloc[:cut].to_numpy()
            if not np.allclose(prefix.to_numpy(), expected, rtol=1e-12, atol=1e-12, equal_nan=True):
                return {"passed": False, "detail": f"Prefix invariance failed at row {cut}."}
            changed = {key: frame.copy() for key, frame in panel.items()}
            shape = (length - cut, len(full.columns))
            multipliers = 1.7 + np.arange(np.prod(shape)).reshape(shape) % 13 / 7.0
            for frame in changed.values():
                frame.iloc[cut:] = frame.iloc[cut:].to_numpy(dtype=float) * multipliers
            perturbed = evaluate_expression(expression, changed).iloc[:cut].to_numpy()
            if not np.allclose(perturbed, expected, rtol=1e-12, atol=1e-12, equal_nan=True):
                return {"passed": False, "detail": f"Future perturbation invariance failed at row {cut}."}
            compared_cells += int(np.isfinite(expected).sum())
        return {
            "passed": True,
            "detail": f"Passed {len(cuts)} prefix and {len(cuts)} future perturbation checks.",
            "prefix_checks": len(cuts),
            "perturbation_checks": len(cuts),
            "finite_cells_compared": compared_cells,
        }
    except (ValueError, TypeError, OverflowError) as exc:
        return {"passed": False, "detail": str(exc)}
