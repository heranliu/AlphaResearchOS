import unittest

import numpy as np
import pandas as pd

from alpharesearchos.backtest import (
    BacktestConfig,
    backtest,
    bootstrap_sharpe_interval,
    evaluate_development,
    summarize,
)


def frames(values, columns=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    close = pd.DataFrame(values, index=pd.bdate_range("2021-01-01", periods=len(values)), columns=columns or [f"A{i}" for i in range(values.shape[1])])
    scores = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    return scores, close


class BacktestTests(unittest.TestCase):
    def test_hand_calculated_costs_entry_and_terminal_liquidation(self):
        scores, close = frames([100, 100, 110, 121])
        curve = backtest(scores, close, BacktestConfig(cost_bps=100, top_k=1), 2, 4)
        self.assertAlmostEqual(curve.iloc[0]["return"], 1.1 / 1.01 - 1, places=12)
        self.assertAlmostEqual(curve.iloc[1]["return"], 1.1 * 0.99 - 1, places=12)
        self.assertAlmostEqual(curve.iloc[-1]["equity"], 1.21 * 0.99 / 1.01, places=12)
        self.assertAlmostEqual(curve.iloc[0]["turnover"], 1 / 1.01, places=12)
        self.assertAlmostEqual(curve.iloc[-1]["turnover"], 1.1, places=12)
        np.testing.assert_allclose(curve["equity"], curve["benchmark_equity"])

    def test_single_day_pays_both_costs(self):
        scores, close = frames([100, 100, 100])
        curve = backtest(scores, close, BacktestConfig(cost_bps=100), 2, 3)
        self.assertAlmostEqual(curve.iloc[0]["equity"], 0.99 / 1.01, places=12)
        self.assertAlmostEqual(curve.iloc[0]["turnover"], 2 / 1.01, places=12)

    def test_two_bar_signal_lag(self):
        scores, close = frames([[100, 100], [100, 100], [110, 50], [110, 50]])
        scores.iloc[:] = [[2, 1], [1, 2], [1, 2], [1, 2]]
        curve = backtest(scores, close, BacktestConfig(cost_bps=0, top_k=1, rebalance_every=1), 2, 4)
        self.assertAlmostEqual(curve.iloc[0]["return"], 0.1)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            BacktestConfig(signal_lag=1)

    def test_drift_adjusted_turnover(self):
        scores, close = frames([[100, 100], [100, 100], [200, 100], [200, 100], [200, 100]])
        curve = backtest(scores, close, BacktestConfig(cost_bps=0, top_k=2, rebalance_every=1), 2, 5)
        self.assertAlmostEqual(curve.iloc[0]["return"], 0.5)
        self.assertAlmostEqual(curve.iloc[1]["turnover"], 1 / 3)
        self.assertAlmostEqual(curve.iloc[2]["turnover"], 1)
        self.assertAlmostEqual(curve.iloc[-1]["equity"], 1.5)

    def test_drift_rebalance_costs_self_financing(self):
        scores, close = frames([[100, 100], [100, 100], [200, 100], [200, 100], [200, 100]])
        curve = backtest(scores, close, BacktestConfig(cost_bps=100, top_k=2, rebalance_every=1), 2, 5)
        # Equal-weight rebalance after A doubles sells and buys 1/6 of NAV;
        # equal cost-induced target reductions cancel in the L1 turnover.
        self.assertAlmostEqual(curve.iloc[1]["turnover"], 1 / 3, places=12)
        self.assertAlmostEqual(curve.iloc[1]["return"], -0.01 / 3, places=12)
        self.assertAlmostEqual(curve.iloc[-1]["equity"], 1.5 / 1.01 * (1 - 0.01 / 3) * 0.99, places=12)
        np.testing.assert_allclose(curve["exposure"], 1)

    def test_invalid_scores_exit_on_rebalance(self):
        scores, close = frames([100, 100, 110, 220, 440])
        scores.iloc[1:] = np.nan
        curve = backtest(scores, close, BacktestConfig(cost_bps=0, rebalance_every=1), 2, 5)
        np.testing.assert_allclose(curve["equity"], [1.1, 1.1, 1.1])
        np.testing.assert_allclose(curve["exposure"], [1, 0, 0])
        held = backtest(scores, close, BacktestConfig(cost_bps=0, rebalance_every=5), 2, 5)
        self.assertAlmostEqual(held.iloc[-1]["equity"], 4.4)

    def test_partial_nan_and_ties_choose_lexical_assets(self):
        scores, close = frames([[100, 100, 100], [100, 100, 100], [110, 120, 200]], columns=["B", "A", "C"])
        scores.iloc[0] = [1, 1, np.inf]
        curve = backtest(scores, close, BacktestConfig(cost_bps=0, top_k=1), 2, 3)
        self.assertAlmostEqual(curve.iloc[0]["return"], 0.2)

    def test_cold_start_each_fold_and_no_position_carry(self):
        scores, close = frames([100] * 7)
        config = BacktestConfig(cost_bps=100, rebalance_every=5)
        result = evaluate_development(scores, close, config, [(2, 4), (4, 7)])
        expected = 0.99 / 1.01
        for fold in result["folds"]:
            self.assertAlmostEqual(fold["metrics"]["total_return"], expected - 1, places=12)
        self.assertAlmostEqual(result["metrics"]["total_return"], expected ** 2 - 1, places=12)
        with self.assertRaisesRegex(ValueError, "nonoverlapping"):
            evaluate_development(scores, close, config, [(2, 5), (4, 7)])

    def test_future_change_cannot_change_past_returns(self):
        scores, close = frames([[100, 100], [101, 99], [99, 103], [102, 101], [103, 100], [104, 105], [102, 108], [106, 109]])
        scores.iloc[:] = np.arange(16).reshape(8, 2) % 3
        config = BacktestConfig(cost_bps=12, top_k=1, rebalance_every=1)
        original = backtest(scores, close, config, 2, 8)
        altered_close, altered_scores = close.copy(), scores.copy()
        altered_close.iloc[5:] *= np.array([[100, 0.1], [0.1, 100], [10, 10]])
        altered_scores.iloc[4:] *= -100
        altered = backtest(altered_scores, altered_close, config, 2, 8)
        pd.testing.assert_frame_equal(original.iloc[:3], altered.iloc[:3])
        prefix = backtest(scores.iloc[:5], close.iloc[:5], config, 2, 5)
        # Final row deliberately includes liquidation: compare earlier rows.
        pd.testing.assert_frame_equal(original.iloc[:2], prefix.iloc[:2])

    def test_rank_ic_uses_tradeable_lagged_returns(self):
        scores, close = frames([[100, 100, 100], [100, 100, 100], [101, 102, 103], [104.03, 104.04, 104.03]])
        scores.iloc[0] = [1, 2, 3]
        scores.iloc[1] = [3, 2, 1]
        scores.iloc[2] = [3, 2, 1]
        scores.iloc[3] = [1, 2, 3]
        result = evaluate_development(scores, close, BacktestConfig(cost_bps=0), [(2, 4)])
        self.assertAlmostEqual(result["rank_ic"], 1)

    def test_drawdown_includes_initial_capital(self):
        curve = pd.DataFrame({"return": [-0.1, 0.0, 0.1], "benchmark_return": [0, 0, 0], "turnover": [1, 0, 1]})
        metrics = summarize(curve, annualization=3)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)
        self.assertAlmostEqual(metrics["total_return"], -0.01)
        self.assertAlmostEqual(metrics["cagr"], -0.01)
        self.assertEqual(metrics["days"], 3)

    def test_no_implicit_price_fill_or_date_reorder(self):
        scores, close = frames([100, 100, 101, 102])
        close.iloc[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            backtest(scores, close, BacktestConfig(), 2, 4)
        scores, close = frames([100, 100, 101, 102])
        with self.assertRaisesRegex(ValueError, "matching"):
            backtest(scores.iloc[::-1], close, BacktestConfig(), 2, 4)

    def test_row_zero_is_cash_and_bootstrap_reproducible(self):
        scores, close = frames([100, 101, 100, 104, 103])
        curve = backtest(scores, close, BacktestConfig(cost_bps=0, rebalance_every=1), 0, 5)
        self.assertEqual(curve.iloc[0]["return"], 0)
        self.assertEqual(curve.iloc[0]["exposure"], 0)
        interval = bootstrap_sharpe_interval(curve["return"], seed=17)
        self.assertEqual(interval, bootstrap_sharpe_interval(curve["return"], seed=17))
        self.assertLessEqual(interval[0], interval[1])
        self.assertEqual(bootstrap_sharpe_interval([0, 0, 0]), [0, 0])


if __name__ == "__main__":
    unittest.main()
