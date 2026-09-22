import unittest

import numpy as np
import pandas as pd

from alpharesearchos.factors import ExpressionError, causality_check, evaluate_expression, validate_expression


def panel(rows=30):
    index = pd.bdate_range("2020-01-01", periods=rows)
    x = np.arange(rows, dtype=float)[:, None]
    close = pd.DataFrame(20 + x * np.array([[1.0, 2.0, 0.5]]) + np.array([[0.0, 10.0, 5.0]]), index=index, columns=["A", "B", "C"])
    return {"close": close, "open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "volume": close * 1000}


class FactorLanguageTests(unittest.TestCase):
    def test_lookback_and_canonical(self):
        metadata = validate_expression(" rank( mean( ret(close, 5), 3 ) ) ")
        self.assertEqual(metadata["lookback"], 7)
        self.assertEqual(metadata["canonical"], "rank(mean(ret(close, 5), 3))")
        self.assertEqual(metadata["fields"], ["close"])
        self.assertGreater(metadata["complexity"], 3)

    def test_exact_returns_and_delay(self):
        data = panel(8)
        pd.testing.assert_frame_equal(evaluate_expression("ret(close, 2)", data), data["close"] / data["close"].shift(2) - 1)
        pd.testing.assert_frame_equal(evaluate_expression("delta(close, 2)", data), data["close"] - data["close"].shift(2))
        pd.testing.assert_frame_equal(evaluate_expression("delay(close, 0)", data), data["close"])

    def test_rolling_full_window_and_population_std(self):
        data = panel(8)
        result = evaluate_expression("std(ret(close, 1), 3)", data)
        self.assertTrue(result.iloc[:3].isna().all().all())
        expected = (data["close"] / data["close"].shift(1) - 1).rolling(3, min_periods=3).std(ddof=0)
        pd.testing.assert_frame_equal(result, expected)
        np.testing.assert_allclose(evaluate_expression("std(close, 1)", data), 0)

    def test_cross_section_rank_ties_and_zscore(self):
        data = panel(3)
        data["close"].iloc[:] = [10.0, 10.0, 20.0]
        np.testing.assert_allclose(evaluate_expression("rank(close)", data), [[0.5, 0.5, 1.0]] * 3)
        z = evaluate_expression("zscore(close)", data)
        np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-14)
        np.testing.assert_allclose(z.std(axis=1, ddof=0), 1)
        self.assertTrue(evaluate_expression("zscore(1)", data).isna().all().all())

    def test_undefined_operations_are_nan_without_fill(self):
        data = panel(4)
        for expression in ("close / 0", "log(-close)", "log(0)"):
            with self.subTest(expression=expression):
                self.assertTrue(evaluate_expression(expression, data).isna().all().all())
        self.assertTrue(evaluate_expression("mean(close / delta(close, 0), 2)", data).isna().all().all())

    def test_language_rejects_execution_and_future_reads(self):
        attacks = [
            "__import__('os').system('touch /tmp/factor-executed')",
            "close.shift(-1)", "delay(close, -1)", "ret(close, -2)",
            "mean(close, 0)", "delay(close, True)", "mean(close, 1.5)",
            "delay(close, 1 + 1)", "close[-1]", "close.iloc[1]",
            "(lambda: close)()", "[x for x in close]", "close if 1 else volume",
            "getattr(close, 'shift')(1)", "rank(close, axis=0)", "sum(close)",
            "close ** 2", "close @ close", "close > 0", "'close'", "True", "1e309",
            "1000001", "9" * 500, "(close := volume)", "mean(close, 505)",
        ]
        for attack in attacks:
            with self.subTest(attack=attack), self.assertRaises(ExpressionError):
                validate_expression(attack)

    def test_work_is_bounded(self):
        with self.assertRaises(ExpressionError):
            validate_expression("rank(" * 18 + "close" + ")" * 18)
        with self.assertRaises(ExpressionError):
            validate_expression("close+" * 60 + "close")
        with self.assertRaises(ExpressionError):
            validate_expression("delay(delay(delay(close, 504), 504), 1)")
        with self.assertRaises(ExpressionError):
            validate_expression(" " * 2049 + "close")

    def test_causality_prefix_and_future_perturbations(self):
        data = panel(50)
        expression = "rank(mean(ret(close, 3), 5)) - rank(std(ret(close, 1), 7))"
        audit = causality_check(expression, data)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["prefix_checks"], 3)
        self.assertEqual(audit["perturbation_checks"], 3)
        self.assertGreater(audit["finite_cells_compared"], 0)
        original = evaluate_expression(expression, data)
        perturbed = {key: value.copy() for key, value in data.items()}
        for value in perturbed.values():
            value.iloc[20:] *= 100
        pd.testing.assert_frame_equal(original.iloc[:20], evaluate_expression(expression, perturbed).iloc[:20])
        pd.testing.assert_frame_equal(original.iloc[:20], evaluate_expression(expression, {key: value.iloc[:20] for key, value in data.items()}))

    def test_missing_prices_and_misalignment_rejected(self):
        data = panel(10)
        data["close"].iloc[4, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_expression("rank(close)", data)
        data = panel(10)
        data["volume"] = data["volume"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "align"):
            evaluate_expression("close", data)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            evaluate_expression("volume", {"close": panel(10)["close"]})


if __name__ == "__main__":
    unittest.main()
