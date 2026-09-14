from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from .config import DEFAULT_CONFIG, TrainingConfig

ProgressCallback = Callable[[str, int, int], None]


def _candidate_models(config: TrainingConfig) -> dict[str, object]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", RobustScaler()),
                ("model", Ridge(alpha=25.0)),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=config.gradient_boosting_estimators,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=20,
                        max_features=0.70,
                        subsample=0.85,
                        loss="huber",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=config.extra_trees_estimators,
                        max_depth=8,
                        min_samples_leaf=12,
                        max_features=0.65,
                        n_jobs=-1,
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
    }


def _recency_weights(length: int, half_life: int) -> np.ndarray:
    age = (length - 1) - np.arange(length)
    weights = np.power(0.5, age / max(1, half_life))
    return weights / np.mean(weights)


def _fit_with_recency(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
) -> object:
    weights = _recency_weights(len(X), config.recency_half_life_sessions)
    model.fit(X, y.clip(-0.35, 0.35), model__sample_weight=weights)
    return model


def _sharpe(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2 or np.std(returns, ddof=1) < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(returns) / np.std(returns, ddof=1))


def _actions(predictions: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(predictions > threshold, 1, np.where(predictions < -threshold, -1, 0))


def _net_returns(
    actual: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    round_trip_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    actions = _actions(predictions, threshold)
    net = actions * np.asarray(actual) - (actions != 0) * round_trip_cost
    return actions, net


def _choose_threshold(
    actual: np.ndarray,
    predictions: np.ndarray,
    round_trip_cost: float,
) -> tuple[float, pd.DataFrame]:
    valid = np.isfinite(actual) & np.isfinite(predictions)
    actual, predictions = np.asarray(actual)[valid], np.asarray(predictions)[valid]
    absolute = np.abs(predictions)
    candidates = sorted(
        {round_trip_cost, *np.quantile(absolute, np.linspace(0.45, 0.85, 9)).tolist()}
    )
    rows = []
    min_trades = max(8, int(len(actual) * 0.10))
    for threshold in candidates:
        actions, net = _net_returns(actual, predictions, threshold, round_trip_cost)
        active = actions != 0
        trades = int(active.sum())
        accuracy = float(np.mean(np.sign(actual[active]) == actions[active])) if trades else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "trades": trades,
                "active_accuracy": accuracy,
                "net_sharpe": _sharpe(net, periods_per_year=52),
                "net_return": float(np.prod(1 + net) - 1),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["trades"] >= min_trades]
    chosen = eligible if not eligible.empty else table
    # Validation Sharpe is used only to choose an abstention level, never reported as test proof.
    best = chosen.sort_values(["net_sharpe", "active_accuracy"], ascending=False).iloc[0]
    return float(best["threshold"]), table


def _evaluation_metrics(
    actual: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    round_trip_cost: float,
    periods_per_year: int = 52,
) -> dict:
    actions, net = _net_returns(actual, predictions, threshold, round_trip_cost)
    active = actions != 0
    trades = int(active.sum())
    strategy_curve = np.cumprod(1 + net)
    running_peak = np.maximum.accumulate(strategy_curve)
    drawdown = strategy_curve / running_peak - 1
    model_mae = mean_absolute_error(actual, predictions)
    zero_mae = float(np.mean(np.abs(actual)))
    return {
        "rows": len(actual),
        "mae_bps": float(model_mae * 10_000),
        "zero_forecast_mae_bps": float(zero_mae * 10_000),
        "mae_skill_vs_zero": float(1 - model_mae / zero_mae) if zero_mae else 0.0,
        "trades": trades,
        "coverage": float(active.mean()),
        "active_directional_accuracy": (
            float(np.mean(np.sign(actual[active]) == actions[active])) if trades else 0.0
        ),
        "net_sharpe": _sharpe(net, periods_per_year=periods_per_year),
        "net_total_return": float(strategy_curve[-1] - 1),
        "max_drawdown": float(drawdown.min()),
    }


def train_asset(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig = DEFAULT_CONFIG,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    started = perf_counter()
    if len(X) < config.min_training_rows:
        raise ValueError(f"Need at least {config.min_training_rows} rows; received {len(X)}.")

    fraction_total = config.train_fraction + config.validation_fraction + config.test_fraction
    if not np.isclose(fraction_total, 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    train_boundary = int(len(X) * config.train_fraction)
    validation_boundary = int(len(X) * (config.train_fraction + config.validation_fraction))
    purge_gap = max(config.cv_gap, config.horizon_sessions)
    train_stop = train_boundary - purge_gap
    validation_stop = validation_boundary - purge_gap
    if train_stop <= 0 or validation_stop <= train_boundary or validation_boundary >= len(X):
        raise ValueError("Not enough rows for the requested purged 70/15/15 split.")

    X_train, y_train = X.iloc[:train_stop], y.iloc[:train_stop]
    X_validation = X.iloc[train_boundary:validation_stop]
    y_validation = y.iloc[train_boundary:validation_stop]
    X_development = X.iloc[:validation_stop]
    y_development = y.iloc[:validation_stop]
    X_test, y_test = X.iloc[validation_boundary:], y.iloc[validation_boundary:]
    splitter = TimeSeriesSplit(n_splits=config.cv_splits, gap=config.cv_gap)
    candidates = _candidate_models(config)
    total_fits = len(candidates) * (config.cv_splits + 3)
    completed_fits = 0
    oof = {name: np.full(len(X_train), np.nan) for name in candidates}
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_train), start=1):
        for name, template in candidates.items():
            model = clone(template)
            _fit_with_recency(
                model,
                X_train.iloc[train_idx],
                y_train.iloc[train_idx],
                config,
            )
            pred = model.predict(X_train.iloc[valid_idx])
            completed_fits += 1
            if progress_callback:
                progress_callback(f"Validation fold {fold}: {name}", completed_fits, total_fits)
            oof[name][valid_idx] = pred
            fold_rows.append(
                {
                    "fold": fold,
                    "model": name,
                    "train_end": X_train.index[train_idx[-1]].date().isoformat(),
                    "validation_start": X_train.index[valid_idx[0]].date().isoformat(),
                    "validation_end": X_train.index[valid_idx[-1]].date().isoformat(),
                    "purge_gap_sessions": config.cv_gap,
                    "mae_bps": mean_absolute_error(y_train.iloc[valid_idx], pred) * 10_000,
                }
            )

    common = np.logical_and.reduce([np.isfinite(values) for values in oof.values()])
    recent_weights = _recency_weights(int(common.sum()), config.recency_half_life_sessions)
    model_mae = {
        name: float(
            np.average(
                np.abs(y_train.to_numpy()[common] - values[common]),
                weights=recent_weights,
            )
        )
        for name, values in oof.items()
    }
    inverse = {name: 1.0 / max(score, 1e-8) for name, score in model_mae.items()}
    total_inverse = sum(inverse.values())
    weights = {name: value / total_inverse for name, value in inverse.items()}

    validation_predictions = np.zeros(len(X_validation))
    for name, template in candidates.items():
        model = clone(template)
        _fit_with_recency(model, X_train, y_train, config)
        validation_predictions += weights[name] * model.predict(X_validation)
        completed_fits += 1
        if progress_callback:
            progress_callback(f"15% validation model: {name}", completed_fits, total_fits)

    validation_monday = X_validation.index.dayofweek == 0
    if not validation_monday.any():
        raise ValueError("Validation segment contains no Monday observations.")
    validation_actual = y_validation.to_numpy()[validation_monday]
    validation_predictions_monday = validation_predictions[validation_monday]
    round_trip_cost = config.round_trip_cost_bps / 10_000
    threshold_window = min(config.threshold_lookback_sessions, len(validation_actual))
    threshold, threshold_table = _choose_threshold(
        validation_actual[-threshold_window:],
        validation_predictions_monday[-threshold_window:],
        round_trip_cost,
    )
    threshold_row = threshold_table.loc[np.isclose(threshold_table["threshold"], threshold)].iloc[0]
    validation_actual_window = validation_actual[-threshold_window:]
    validation_prediction_window = validation_predictions_monday[-threshold_window:]
    validation_zero_mae = float(np.mean(np.abs(validation_actual_window)))
    validation_model_mae = float(
        np.mean(np.abs(validation_actual_window - validation_prediction_window))
    )
    validation_mae_skill = (
        1 - validation_model_mae / validation_zero_mae if validation_zero_mae else 0.0
    )
    policy_enabled = bool(
        threshold_row["net_sharpe"] > 0
        and threshold_row["active_accuracy"] > 0.50
        and validation_mae_skill > 0
    )

    evaluation_models = {}
    test_predictions = np.zeros(len(X_test))
    for name, template in candidates.items():
        model = clone(template)
        _fit_with_recency(model, X_development, y_development, config)
        evaluation_models[name] = model
        test_predictions += weights[name] * model.predict(X_test)
        completed_fits += 1
        if progress_callback:
            progress_callback(f"15% untouched test model: {name}", completed_fits, total_fits)

    test_monday = X_test.index.dayofweek == 0
    if not test_monday.any():
        raise ValueError("Test segment contains no Monday observations.")
    test_actual_monday = y_test.to_numpy()[test_monday]
    test_predictions_monday = test_predictions[test_monday]
    test_metrics = _evaluation_metrics(
        test_actual_monday,
        test_predictions_monday,
        threshold,
        round_trip_cost,
        periods_per_year=52,
    )
    test_actions, test_net = _net_returns(
        test_actual_monday, test_predictions_monday, threshold, round_trip_cost
    )
    holdout = pd.DataFrame(
        {
            "Date": X_test.index[test_monday],
            "actual_return": test_actual_monday,
            "predicted_return": test_predictions_monday,
            "action": test_actions,
            "absolute_error": np.abs(test_actual_monday - test_predictions_monday),
            "direction_correct": np.where(
                test_actions == 0,
                np.nan,
                np.sign(test_actual_monday) == test_actions,
            ),
            "net_strategy_return": test_net,
            "equity": np.cumprod(1 + test_net),
        }
    )

    # Deployment refit: after the untouched test has supplied an honest score, all
    # labelled rows become training data for the live model.
    final_models = {}
    for name, template in candidates.items():
        model = clone(template)
        _fit_with_recency(model, X, y, config)
        final_models[name] = model
        completed_fits += 1
        if progress_callback:
            progress_callback(f"Final all-data model: {name}", completed_fits, total_fits)

    return {
        "models": final_models,
        "weights": weights,
        "threshold": threshold,
        "policy_enabled": policy_enabled,
        "validation_policy_metrics": threshold_row.to_dict(),
        "validation_mae_skill_vs_zero": validation_mae_skill,
        "feature_names": X.columns.tolist(),
        "training_start": X.index.min().date().isoformat(),
        "trained_through": X.index.max().date().isoformat(),
        "n_rows": len(X),
        "n_monday_rows": int((X.index.dayofweek == 0).sum()),
        "split_rows": {
            "train": len(X_train),
            "validation": len(X_validation),
            "test": len(X_test),
            "purged_at_each_boundary": purge_gap,
        },
        "validation_start": X_validation.index.min().date().isoformat(),
        "validation_end": X_validation.index.max().date().isoformat(),
        "test_start": X_test.index.min().date().isoformat(),
        "test_end": X_test.index.max().date().isoformat(),
        "test_metrics": test_metrics,
        "fold_metrics": fold_rows,
        "threshold_search": threshold_table.to_dict(orient="records"),
        "holdout": holdout,
        "training_seconds": perf_counter() - started,
    }


def weighted_predict(asset_bundle: dict, row: pd.DataFrame) -> float:
    row = row.reindex(columns=asset_bundle["feature_names"])
    return float(
        sum(
            asset_bundle["weights"][name] * model.predict(row)[0]
            for name, model in asset_bundle["models"].items()
        )
    )


def signal_from_prediction(prediction: float, threshold: float, enabled: bool = True) -> str:
    if not enabled:
        return "HOLD"
    if prediction > threshold:
        return "BUY"
    if prediction < -threshold:
        return "SELL"
    return "HOLD"


def bundle_metadata(config: TrainingConfig, **extra: object) -> dict:
    return {
        "schema_version": 4,
        "app_version": "0.3.0",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "training_config": asdict(config),
        **extra,
    }
