"""Maturity, training isolation, actual learning, and reproducible model tests."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest

from alpharesearchos.predictive import strategy_scores, validate_strategy


def panel(rows=180, assets=8, nonlinear=False):
    rng = np.random.default_rng(71)
    volume = np.asarray([rng.permutation(np.arange(1, assets + 1)) for _ in range(rows)], dtype=float) * 1000
    ranks = volume / (1000 * assets)
    targets = np.where(ranks > 0.5, 0.008, -0.006) if nonlinear else 0.012 * (ranks - (assets + 1) / (2 * assets))
    prices = np.full((rows, assets), 100.0)
    for t in range(2, rows):
        # A feature observed at i determines the next executable one-day
        # return: close[i+2]/close[i+1]-1. No estimator sees this generator.
        prices[t] = prices[t - 1] * (1.0 + targets[t - 2])
    open_ = np.vstack([prices[0], prices[:-1]])
    data = {"close": prices, "open": open_, "high": np.maximum(open_, prices) * 1.01,
            "low": np.minimum(open_, prices) * 0.99, "volume": volume}
    index = pd.bdate_range("2020-01-02", periods=rows)
    return {name: pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(assets)]) for name, values in data.items()}


def candidate(model="ridge", features=None, **params):
    return {"features": features or ["volume"], "model": model,
            "model_params": {"train_window": 63, "retrain_every": 10, "horizon": 1, "smoothing": 1, **params}}


def test_schema_normalization_canonicalization_and_real_warmup():
    normalized = validate_strategy(candidate(features=[" ret(close, 5) ", "volume"], smoothing=3))
    assert normalized["features"] == ["ret(close, 5)", "volume"]
    assert normalized["feature_lookback"] == 5
    assert normalized["lookback"] == 5 + 1 + 32 + 2
    other = validate_strategy(candidate(features=["volume", "ret(close,5)"], smoothing=3))
    assert other["canonical"] == normalized["canonical"]
    rank_a = validate_strategy(candidate("rank", train_window=63, alpha=1))
    rank_b = validate_strategy(candidate("rank", train_window=504, alpha=100))
    assert rank_a["canonical"] == rank_b["canonical"]
    assert rank_a["lookback"] == 0
    assert "hypothesis" not in validate_strategy({**candidate(), "hypothesis": "descriptive text"})


@pytest.mark.parametrize("changes", [
    {"features": []}, {"features": ["close"] * 7}, {"features": ["close", "(close)"]},
    {"features": ["delay(close, -1)"]}, {"features": ["__import__('os').system('id')"]},
    {"features": ["delay(close, 253)"]}, {"model": "python"}, {"model": []},
    {"model_params": {"alpha": True}}, {"model_params": {"alpha": 0}},
    {"model_params": {"alpha": float("nan")}}, {"model_params": {"horizon": -1}},
    {"model_params": {"horizon": 1.5}}, {"model_params": {"horizon": True}},
    {"model_params": {"train_window": 62}}, {"model_params": {"retrain_every": 0}},
    {"model_params": {"smoothing": 21}}, {"model_params": {"random_state": 99}},
])
def test_schema_rejects_execution_unbounded_work_and_invalid_parameters(changes):
    with pytest.raises(ValueError):
        validate_strategy({**candidate(), **changes})


def test_rank_is_exact_equal_rank_ensemble_with_causal_smoothing():
    data = panel(50)
    strategy = candidate("rank", ["ret(close, 3)", "volume"], smoothing=3)
    scores, audit = strategy_scores(strategy, data)
    expected = ((data["close"] / data["close"].shift(3) - 1).rank(axis=1, pct=True)
                + data["volume"].rank(axis=1, pct=True)) / 2
    expected = expected.rolling(3, min_periods=3).mean()
    pd.testing.assert_frame_equal(scores, expected)
    assert audit["first_prediction_row"] == 5
    assert audit["fit_count"] == 0 and audit["fit_blocks"] == []
    assert audit["target"] is None and audit["training_frozen"] is False


def test_first_prediction_requires_real_mature_labels_and_complete_smoothing():
    data = panel(70)
    scores, audit = strategy_scores(candidate(smoothing=3), data)
    assert scores.iloc[:35].isna().all().all()
    assert scores.iloc[35].notna().all()
    assert audit["first_prediction_row"] == 35
    first = audit["fit_blocks"][0]
    assert first["fit_row"] == 33
    assert first["train_feature_start_row"] == 0
    assert first["train_feature_end_row"] == 31
    assert first["label_start_row"] == 2 and first["label_end_row"] == 33
    assert first["training_dates"] == 32 and first["samples"] == 32 * 8
    tiny, tiny_audit = strategy_scores(candidate(), {k: v.iloc[:33] for k, v in data.items()})
    assert tiny.isna().all().all() and tiny_audit["fit_count"] == 0
    frozen, frozen_audit = strategy_scores(candidate(), data, fit_cutoff=20)
    assert frozen.isna().all().all() and frozen_audit["fit_count"] == 0


def test_ridge_matches_hand_calculated_past_training_fit():
    data = panel(80)
    scores, audit = strategy_scores(candidate(alpha=2.5), data)
    x = data["volume"].iloc[:32].rank(axis=1, pct=True).to_numpy().reshape(-1)
    y = (data["close"].iloc[2:34].to_numpy() / data["close"].iloc[1:33].to_numpy() - 1).reshape(-1)
    mean, scale = x.mean(), x.std(ddof=0)
    z = (x - mean) / scale
    beta = np.dot(z, y - y.mean()) / (np.dot(z, z) + 2.5)
    prediction_x = data["volume"].iloc[33].rank(pct=True).to_numpy()
    expected = (prediction_x - mean) / scale * beta + y.mean()
    np.testing.assert_allclose(scores.iloc[33], expected, rtol=1e-12, atol=1e-14)
    fitted = audit["fit_blocks"][0]
    np.testing.assert_allclose(fitted["normalization"]["mean"], [mean])
    np.testing.assert_allclose(fitted["normalization"]["scale"], [scale])
    np.testing.assert_allclose(fitted["estimator"]["coefficients"], [beta])
    # This controlled signal is learned from labels, not an unrelated fallback.
    realized = data["close"].iloc[35].to_numpy() / data["close"].iloc[34].to_numpy() - 1
    assert np.corrcoef(scores.iloc[33], realized)[0, 1] > 0.999999


@pytest.mark.parametrize("model", ["rank", "ridge", "hist_gbdt"])
def test_prefix_and_future_perturbation_cannot_change_past_predictions(model):
    if model == "hist_gbdt":
        pytest.importorskip("sklearn")
    data = panel(140)
    strategy = candidate(model, ["log(volume - 3500)", "ret(close, 1)"], smoothing=3)
    full, audit = strategy_scores(strategy, data)
    cut = 90
    prefix, _ = strategy_scores(strategy, {k: v.iloc[:cut] for k, v in data.items()})
    pd.testing.assert_frame_equal(full.iloc[:cut], prefix, rtol=1e-10, atol=1e-12)
    changed = {k: v.copy() for k, v in data.items()}
    multipliers = 1.5 + np.arange((len(full) - cut) * len(full.columns)).reshape(len(full) - cut, -1) % 7
    for field in ["open", "high", "low", "close"]:
        changed[field].iloc[cut:] *= multipliers
    changed["volume"].iloc[cut:] = np.maximum(1.0, 9000 - changed["volume"].iloc[cut:].to_numpy())
    mutated, mutated_audit = strategy_scores(strategy, changed)
    pd.testing.assert_frame_equal(full.iloc[:cut], mutated.iloc[:cut], rtol=1e-10, atol=1e-12)
    prior_fits = [fit for fit in audit["fit_blocks"] if fit["fit_row"] < cut]
    mutated_fits = [fit for fit in mutated_audit["fit_blocks"] if fit["fit_row"] < cut]
    assert prior_fits == mutated_fits


@pytest.mark.parametrize("model", ["ridge", "hist_gbdt"])
def test_exclusive_cutoff_freezes_training_without_freezing_future_features(model):
    if model == "hist_gbdt":
        pytest.importorskip("sklearn")
    data = panel(180)
    strategy = candidate(model, smoothing=3)
    scores, audit = strategy_scores(strategy, data, fit_cutoff=100)
    assert audit["training_frozen"] is True
    assert audit["fit_blocks"][-1]["fit_row"] == 99
    assert audit["fit_cutoff_date"] == str(data["close"].index[99].date())
    assert all(fit["label_end_row"] <= fit["fit_row"] < 100 for fit in audit["fit_blocks"])
    assert audit["fit_blocks"][-1]["prediction_end_row"] == 179
    changed = {k: v.copy() for k, v in data.items()}
    for field in ["close", "open", "high", "low"]:
        changed[field].iloc[100:] *= np.exp(np.arange(80)[:, None] * 0.02)
    label_changed, changed_audit = strategy_scores(strategy, changed, fit_cutoff=100)
    pd.testing.assert_frame_equal(scores, label_changed, rtol=1e-12, atol=1e-14)
    assert audit["fit_blocks"] == changed_audit["fit_blocks"]
    changed["volume"].iloc[100:] = 9000 - changed["volume"].iloc[100:].to_numpy()
    features_changed, _ = strategy_scores(strategy, changed, fit_cutoff=100)
    assert not np.allclose(scores.iloc[110:], features_changed.iloc[110:])


def test_later_cutoff_preserves_prefix_protocol_and_actual_dates():
    data = panel(150)
    full, _ = strategy_scores(candidate(), data, fit_cutoff=110)
    prefix, audit = strategy_scores(candidate(), {k: v.iloc[:80] for k, v in data.items()}, fit_cutoff=110)
    pd.testing.assert_frame_equal(full.iloc[:80], prefix, rtol=1e-12, atol=1e-14)
    assert audit["fit_cutoff_date"] is None and audit["training_frozen"] is False
    with pytest.raises(ValueError):
        strategy_scores(candidate(), data, fit_cutoff=True)
    with pytest.raises(ValueError):
        strategy_scores(candidate(), data, fit_cutoff=-1)


def test_training_window_maturity_and_audit_are_exact_and_json_safe():
    data = panel(200)
    scores, audit = strategy_scores(candidate(horizon=5, smoothing=1), data)
    assert audit["refit_schedule"] == list(range(37, 200, 10))
    for fit in audit["fit_blocks"]:
        assert fit["label_start_row"] == fit["train_feature_start_row"] + 6
        assert fit["label_end_row"] == fit["train_feature_end_row"] + 6
        assert fit["label_end_row"] <= fit["fit_row"]
        assert 32 <= fit["training_dates"] <= 63
        assert fit["samples"] == fit["training_dates"] * 8
        assert fit["prediction_start_row"] >= fit["label_end_row"]
    assert audit["fit_blocks"][-1]["training_dates"] == 63
    json.dumps(audit, allow_nan=False)
    again, second_audit = strategy_scores(candidate(horizon=5, smoothing=1), data)
    pd.testing.assert_frame_equal(scores, again, check_exact=True)
    assert audit == second_audit


def test_histogram_boosting_learns_controlled_nonlinear_target_deterministically():
    pytest.importorskip("sklearn")
    data = panel(130, nonlinear=True)
    strategy = candidate("hist_gbdt", smoothing=1, retrain_every=20)
    scores, audit = strategy_scores(strategy, data)
    first = audit["first_prediction_row"]
    prediction = scores.iloc[first].to_numpy()
    realized = data["close"].iloc[first + 2].to_numpy() / data["close"].iloc[first + 1].to_numpy() - 1
    assert np.corrcoef(prediction, realized)[0, 1] > 0.99
    assert audit["fit_blocks"][0]["estimator"]["type"] == "sklearn_hist_gradient_boosting"
    assert audit["fit_blocks"][0]["estimator"]["early_stopping"] is False
    again, second_audit = strategy_scores(strategy, data)
    pd.testing.assert_frame_equal(scores, again, check_exact=True)
    assert audit == second_audit


def test_histogram_dependency_failure_is_explicit_and_never_falls_back(monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("simulated missing optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(RuntimeError, match="no fallback"):
        strategy_scores(candidate("hist_gbdt"), panel(20))


def test_no_missing_feature_imputation_or_invalid_price_filling():
    data = panel(100)
    scores, audit = strategy_scores(candidate(features=["volume", "close / 0"]), data)
    assert scores.isna().all().all()
    assert audit["fit_count"] == 0 and audit["skipped_fit_blocks"]
    data["close"].iloc[50, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        strategy_scores(candidate(), data)
