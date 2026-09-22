import math

import numpy as np
import pandas as pd
import pytest

from alpharesearchos.factorminer_adapter import (
    UnsupportedFormula,
    differential_check,
    evaluate_factorminer,
    from_factorminer,
)
from alpharesearchos.vendor.factorminer.parser import parse


@pytest.fixture
def panel():
    rng = np.random.default_rng(77)
    index = pd.bdate_range("2020-01-01", periods=100)
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .02, (100, 6)), axis=0)), index=index, columns=list("ABCDEF"))
    return {"close": close, "open": close * .999, "high": close * 1.01,
            "low": close * .99, "volume": close * 1000}


@pytest.mark.parametrize("formula", [
    "Delay($close, -1)", "Delay($close, 0)", "Delay($close, 1.5)",
    "Delay($close, 1, 99)", "Mean($close, 251)", "Add($close, $open, 7)",
    "1e999", "Power($close, 1e999)", "Clip($close, 2, 1)",
])
def test_rejects_invalid_upstream_parameters(formula):
    with pytest.raises((SyntaxError, ValueError)):
        parse(formula)


def test_parser_bounds_before_recursion():
    with pytest.raises(SyntaxError, match="nesting"):
        parse("Neg(" * 100 + "$close" + ")" * 100)


def test_nan_division_and_exact_zero():
    data = {"$close": np.array([[1., 0., 2., np.nan]]), "$open": np.array([[np.nan, 0., 1e-12, 1.]])}
    actual = parse("Div($close, $open)").evaluate(data)
    np.testing.assert_allclose(actual, [[np.nan, np.nan, 2e12, np.nan]], equal_nan=True)


def test_average_rank_is_invariant_to_asset_permutation():
    values = np.array([[1., 0.], [1., 0.], [3., 0.], [np.nan, 0.]])
    tree = parse("CsRank($close)")
    actual = tree.evaluate({"$close": values})
    expected = pd.DataFrame(values.T).rank(axis=1, method="average", pct=True).to_numpy().T
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(tree.evaluate({"$close": values[order]}), actual[order], equal_nan=True)


def test_nested_warmup_requires_complete_window():
    x = np.array([[1., 2., 3., 4., 5., 6.]])
    actual = parse("Mean(Delay($close, 2), 3)").evaluate({"$close": x})
    np.testing.assert_allclose(actual, [[np.nan, np.nan, np.nan, np.nan, 2., 3.]], equal_nan=True)


def test_serialization_preserves_double_precision():
    value = math.sqrt(19 / 20)
    source = f"Mul($close, {value!r})"
    assert parse(parse(source).to_string()).root.children[1].value == value


@pytest.mark.parametrize("expression", [
    "close", "rank(ret(close, 5))", "-rank(ret(close, 1))",
    "mean(ret(close, 1), 5)", "std(ret(close, 1), 5)",
    "rank(ret(close, 20) / std(ret(close, 1), 20))",
    "delta(close, 5)", "delay(volume, 3)", "abs(close-open)",
    "sign(ret(close, 1))", "close / (close-close)", "rank(close-close)",
    "rank(mean(ret(close, 1), 5) + std(ret(close, 1), 5))",
])
def test_independent_evaluators_agree(panel, expression):
    result = differential_check(expression, panel)
    assert result["status"] == "passed", result
    assert result["compared_cells"] == 600


@pytest.mark.parametrize("expression", ["log(close)", "zscore(close)", "ret(close, 100)", "delay(close, 0)"])
def test_unsupported_is_explicit(panel, expression):
    result = differential_check(expression, panel)
    assert result["status"] == "skipped"
    assert result["passed"] is None
    assert result["reason"]


def test_reports_real_mismatch(panel):
    expected = evaluate_factorminer("rank(ret(close, 5))", panel)
    expected.iloc[-1, 0] += 1
    result = differential_check("rank(ret(close, 5))", panel, expected)
    assert result["status"] == "failed"
    assert result["mismatched_cells"] == 1


@pytest.mark.parametrize("formula", [
    "Neg(CsRank(Return($close, 5)))", "Mean(Delta($close, 1), 5)",
    "Std(Return($close, 1), 5)", "Div($close, $open)",
])
def test_import_preserves_upstream_values(panel, formula):
    from alpharesearchos.factors import evaluate_expression
    expression = from_factorminer(formula)
    actual = evaluate_expression(expression, panel).to_numpy()
    expected = parse(formula).evaluate({"$" + k: v.to_numpy().T for k, v in panel.items()}).T
    np.testing.assert_allclose(actual, expected, equal_nan=True, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("formula", ["CsZScore($close)", "Log($close)", "$returns", "KAMA($close, 5)"])
def test_rejects_semantically_incompatible_import(formula):
    with pytest.raises(UnsupportedFormula):
        from_factorminer(formula)
