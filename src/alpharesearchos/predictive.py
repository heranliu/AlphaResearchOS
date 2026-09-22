"""Causal factor ensembles and chronological supervised return prediction.

For feature row i, the target is close[i + 1 + horizon] / close[i + 1] - 1.
It is usable for fitting at row t only when i + 1 + horizon <= t. This target
starts at the first close at which a close-derived signal can be executed by
the existing two-bar-lag backtest. Overlapping targets are allowed inside the
historical training window; no target crossing the fitting date is admitted.

Every feature is ranked across assets separately on each date. Supervised
normalization is fitted on training samples only. Refit dates are fixed from
the theoretical first fit, independent of performance or future availability.
An exclusive fit_cutoff adds a deterministic final fit at cutoff - 1 and then
freezes the model. Holdout features remain causal and may continue to change.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .factors import evaluate_expression, validate_expression

MIN_TRAIN_DATES = 32
MIN_TRAIN_SAMPLES = 64
DEFAULT_PARAMS = {"alpha": 1.0, "train_window": 504, "retrain_every": 63, "horizon": 5, "smoothing": 3}
_INTEGER_BOUNDS = {"train_window": (63, 1008), "retrain_every": (5, 252), "horizon": (1, 20), "smoothing": (1, 20)}
_MODELS = frozenset({"rank", "ridge", "hist_gbdt"})


def validate_strategy(candidate: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the public strategy contract and reject unsafe parameters.

    Extra top-level descriptive fields (name, hypothesis, parents) are not
    carried into the executable strategy. Unknown model parameters are rejected.
    ``lookback`` is the earliest theoretically complete smoothed prediction;
    insufficient finite training observations can delay it further.
    """
    if not isinstance(candidate, dict):
        raise ValueError("Strategy must be a JSON object.")
    features = candidate.get("features")
    if not isinstance(features, list) or not 1 <= len(features) <= 6:
        raise ValueError("Strategy requires 1–6 safe DSL feature expressions.")
    validated = [validate_expression(expression) for expression in features]
    canonical_features = sorted(metadata["canonical"] for metadata in validated)
    if len(set(canonical_features)) != len(canonical_features):
        raise ValueError("Strategy features must be distinct canonical expressions.")
    feature_complexity = sum(metadata["complexity"] for metadata in validated)
    feature_lookback = max(metadata["lookback"] for metadata in validated)
    if feature_complexity > 120 or feature_lookback > 252:
        raise ValueError("Strategy exceeds 120 total feature nodes or 252 rows of feature lookback.")
    model = candidate.get("model", "rank")
    if not isinstance(model, str) or model not in _MODELS:
        raise ValueError("model must be rank, ridge, or hist_gbdt.")
    supplied = candidate.get("model_params", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_PARAMS):
        raise ValueError("Unknown model parameter; supported parameters are alpha, train_window, retrain_every, horizon, smoothing.")
    params = {**DEFAULT_PARAMS, **supplied}
    alpha = params["alpha"]
    if type(alpha) not in {int, float} or not math.isfinite(alpha) or not 1e-6 <= alpha <= 10000:
        raise ValueError("alpha must be finite and between 1e-6 and 10000.")
    params["alpha"] = float(alpha)
    for name, (minimum, maximum) in _INTEGER_BOUNDS.items():
        value = params[name]
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")
    # Parameters that have no effect on an untrained rank ensemble must not
    # create separate canonical candidates merely through unused knob changes.
    effective_params = {"smoothing": params["smoothing"]} if model == "rank" else params
    canonical = json.dumps({"features": canonical_features, "model": model, "model_params": effective_params}, sort_keys=True, separators=(",", ":"))
    minimum_fit = 0 if model == "rank" else params["horizon"] + MIN_TRAIN_DATES
    return {
        "features": canonical_features, "model": model, "model_params": params,
        "canonical": canonical,
        "complexity": feature_complexity + {"rank": 0, "ridge": 6, "hist_gbdt": 18}[model] + (1 if params["smoothing"] > 1 else 0),
        "feature_lookback": feature_lookback,
        "lookback": feature_lookback + minimum_fit + params["smoothing"] - 1,
    }


def _fit_predict(
    model: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    constant = scale < 1e-12
    scale = np.where(constant, 1.0, scale)
    train = (x_train - mean) / scale
    predict = (x_predict - mean) / scale
    details: dict[str, Any] = {
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist(), "constant_features": np.flatnonzero(constant).tolist(), "fitted_on": "training samples only"},
    }
    if model == "ridge":
        intercept = float(y_train.mean())
        # Same objective as ordinary Ridge with an unpenalized intercept:
        # ||y - intercept - standardized_X @ beta||² + alpha * ||beta||².
        covariance = train.T @ train + alpha * np.eye(train.shape[1])
        coefficients = np.linalg.solve(covariance, train.T @ (y_train - intercept))
        predictions = predict @ coefficients + intercept
        details["estimator"] = {"type": "numpy_ridge", "coefficients": coefficients.tolist(), "intercept": intercept, "alpha": alpha}
    else:
        try:
            import sklearn
            from sklearn.ensemble import HistGradientBoostingRegressor
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise RuntimeError("hist_gbdt requires scikit-learn; install the project's predictive-model dependency. No fallback model was used.") from exc
        estimator = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_iter=64,
            max_leaf_nodes=7, max_depth=3, min_samples_leaf=20,
            l2_regularization=alpha, max_bins=63, early_stopping=False, random_state=42,
        )
        with threadpool_limits(limits=1):
            estimator.fit(train, y_train)
            predictions = estimator.predict(predict) if len(predict) else np.empty(0)
        details["estimator"] = {
            "type": "sklearn_hist_gradient_boosting", "sklearn_version": sklearn.__version__,
            "iterations": int(estimator.n_iter_), "learning_rate": 0.05,
            "max_leaf_nodes": 7, "max_depth": 3, "min_samples_leaf": 20,
            "max_bins": 63, "l2_regularization": alpha,
            "early_stopping": False, "random_state": 42,
        }
    if not np.isfinite(predictions).all():
        raise ValueError("Predictive estimator returned nonfinite scores.")
    return predictions, details


def strategy_scores(
    candidate: dict[str, Any],
    panel: dict[str, pd.DataFrame],
    *,
    fit_cutoff: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return causal daily scores and an explicit, JSON-safe training audit.

    ``fit_cutoff`` is an exclusive row index. Values beyond the current prefix
    length are permitted, so the same frozen protocol can be prefix-audited.
    Training starts with >=32 valid dates and >=64 pooled samples, expands to
    train_window dates, and subsequently rolls. No missing feature or label is
    filled. Rank ensembles require every constituent for each scored asset.

    Predictions before enough historical labels have matured remain NaN.
    A scheduled fit with too few samples leaves that prediction block NaN;
    no old model, hardcoded factor, or template is substituted.
    """
    strategy = validate_strategy(candidate)
    if fit_cutoff is not None and (type(fit_cutoff) is not int or fit_cutoff < 0):
        raise ValueError("fit_cutoff must be a nonnegative exclusive row index or None.")
    params, model = strategy["model_params"], strategy["model"]
    # evaluate_expression validates complete positive aligned input panels.
    frames = [evaluate_expression(expression, panel).rank(axis=1, method="average", pct=True, na_option="keep") for expression in strategy["features"]]
    template = frames[0]
    features = np.stack([frame.to_numpy(dtype=float) for frame in frames], axis=2)
    length, assets, feature_count = features.shape
    minimum_samples = max(MIN_TRAIN_SAMPLES, 8 * feature_count)
    horizon = params["horizon"]

    def date(row: int) -> str:
        return str(template.index[row].date())

    cutoff_date = date(fit_cutoff - 1) if fit_cutoff is not None and 0 < fit_cutoff <= length else None
    audit: dict[str, Any] = {
        "model": model, "model_params": dict(params), "feature_expressions": list(strategy["features"]),
        "feature_transform": "cross_sectional_average_percentile_rank",
        "feature_lookback": strategy["feature_lookback"], "theoretical_first_score_row": strategy["lookback"],
        "fit_cutoff": fit_cutoff, "fit_cutoff_date": cutoff_date,
        "training_frozen": False,
        "minimum_training_dates": MIN_TRAIN_DATES, "minimum_training_samples": minimum_samples,
        "target": {"formula": "close[i+1+horizon] / close[i+1] - 1", "horizon": horizon, "label_available_offset": horizon + 1},
        "fit_count": 0, "fit_blocks": [], "skipped_fit_blocks": [],
        "rows": length, "assets": assets, "smoothing": params["smoothing"],
        "first_prediction_row": None, "first_prediction_date": None,
        "finite_fraction": 0.0, "missing_policy": "no filling; all features required per sample",
    }
    if model == "rank":
        raw = np.mean(features, axis=2)
        audit["target"] = None
        audit["training_note"] = "Equal-rank feature ensemble; no predictive model is fitted. Only smoothing is an effective model parameter."
    else:
        if model == "hist_gbdt":
            # Fail explicitly even if a short panel would never reach a fit.
            try:
                from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("hist_gbdt requires scikit-learn; no fallback model was used.") from exc
        prices = panel["close"].to_numpy(dtype=float)
        labels = np.full((length, assets), np.nan)
        if length > horizon + 1:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                labels[:length - horizon - 1] = prices[horizon + 1:] / prices[1:length - horizon] - 1.0
        raw = np.full((length, assets), np.nan)
        first_fit = strategy["feature_lookback"] + horizon + MIN_TRAIN_DATES
        fit_end = length if fit_cutoff is None else min(length, fit_cutoff)
        schedule = list(range(first_fit, fit_end, params["retrain_every"]))
        if fit_cutoff is not None and first_fit <= fit_cutoff - 1 < length and (not schedule or schedule[-1] != fit_cutoff - 1):
            schedule.append(fit_cutoff - 1)
        for position, fit_row in enumerate(schedule):
            block_end = schedule[position + 1] if position + 1 < len(schedule) else length
            # Last usable feature date has a fully realized target at fit_row.
            last_feature = fit_row - horizon - 1
            first_feature = max(strategy["feature_lookback"], last_feature - params["train_window"] + 1)
            x = features[first_feature:last_feature + 1]
            y = labels[first_feature:last_feature + 1]
            eligible = np.isfinite(x).all(axis=2) & np.isfinite(y)
            valid_dates = np.flatnonzero(eligible.any(axis=1)) + first_feature
            samples = int(eligible.sum())
            if len(valid_dates) < MIN_TRAIN_DATES or samples < minimum_samples:
                audit["skipped_fit_blocks"].append({"fit_row": fit_row, "fit_date": date(fit_row), "training_dates": int(len(valid_dates)), "samples": samples, "reason": "insufficient complete matured training samples"})
                continue
            x_train, y_train = x[eligible], y[eligible]
            prediction_features = features[fit_row:block_end]
            prediction_valid = np.isfinite(prediction_features).all(axis=2)
            predicted, fitted = _fit_predict(model, x_train, y_train, prediction_features[prediction_valid], params["alpha"])
            block = raw[fit_row:block_end]
            block[prediction_valid] = predicted
            label_start = int(valid_dates[0] + horizon + 1)
            label_end = int(valid_dates[-1] + horizon + 1)
            if label_end > fit_row or (fit_cutoff is not None and label_end >= fit_cutoff):
                raise AssertionError("Internal error: unmatured label entered predictive training.")
            audit["fit_blocks"].append({
                "fit_row": fit_row, "fit_date": date(fit_row),
                "train_feature_start_row": int(valid_dates[0]), "train_feature_end_row": int(valid_dates[-1]),
                "train_feature_start_date": date(int(valid_dates[0])), "train_feature_end_date": date(int(valid_dates[-1])),
                "label_start_row": label_start, "label_end_row": label_end,
                "label_start_date": date(label_start), "label_end_date": date(label_end),
                "training_dates": int(len(valid_dates)), "samples": samples,
                "prediction_start_row": fit_row, "prediction_end_row": block_end - 1,
                "prediction_start_date": date(fit_row), "prediction_end_date": date(block_end - 1),
                **fitted,
            })
        audit["fit_count"] = len(audit["fit_blocks"])
        audit["training_frozen"] = bool(fit_cutoff is not None and fit_cutoff <= length and audit["fit_count"])
        audit["refit_schedule"] = schedule
        audit["training_note"] = "Expanding then rolling past-only training; fixed refit calendar and optional final boundary refit; no early stopping or holdout-based tuning."
    result = pd.DataFrame(raw, index=template.index, columns=template.columns)
    result = result.rolling(params["smoothing"], min_periods=params["smoothing"]).mean()
    finite = np.isfinite(result.to_numpy())
    finite_rows = np.flatnonzero(finite.any(axis=1))
    if len(finite_rows):
        audit["first_prediction_row"] = int(finite_rows[0])
        audit["first_prediction_date"] = date(int(finite_rows[0]))
    audit["finite_fraction"] = float(finite.mean())
    # Defensive check for all audit fields before an orchestration checkpoint.
    json.dumps(audit, allow_nan=False)
    return result, audit
