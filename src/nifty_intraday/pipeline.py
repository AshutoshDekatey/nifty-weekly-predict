from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from .config import (
    ALL_SYMBOLS,
    DATA_PATH,
    DEFAULT_CONFIG,
    DISPLAY_NAMES,
    METADATA_PATH,
    MODEL_PATH,
    NIFTY,
    TrainingConfig,
)
from .data import refresh_market_data
from .features import build_feature_frame, build_target, make_dataset, select_correlated_stocks
from .modeling import (
    ProgressCallback,
    bundle_metadata,
    signal_from_prediction,
    train_asset,
    weighted_predict,
)
from .morning import IST, load_morning_features, refresh_morning_features
from .tracking import update_prediction_history


def save_bundle(
    bundle: dict, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_model = model_path.with_suffix(".tmp")
    joblib.dump(bundle, temp_model, compress=3)
    temp_model.replace(model_path)

    summary = dict(bundle["metadata"])
    summary["assets"] = {
        symbol: {
            "trained_through": item["trained_through"],
            "test_metrics": item["test_metrics"],
        }
        for symbol, item in bundle["assets"].items()
    }
    temp_meta = metadata_path.with_suffix(".tmp")
    temp_meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temp_meta.replace(metadata_path)


def load_bundle(model_path: Path = MODEL_PATH) -> dict | None:
    return joblib.load(model_path) if model_path.exists() else None


def train_from_data(
    data: pd.DataFrame,
    config: TrainingConfig = DEFAULT_CONFIG,
    save: bool = True,
    progress_callback: ProgressCallback | None = None,
    morning_features: pd.DataFrame | None = None,
) -> dict:
    started = perf_counter()
    nifty_target = build_target(data, NIFTY, config.horizon_sessions).dropna()
    if len(nifty_target) < config.min_training_rows:
        raise ValueError("NIFTY history is too short after feature construction.")
    selection_cutoff = nifty_target.index[int(len(nifty_target) * config.train_fraction) - 1]
    stocks, correlations = select_correlated_stocks(
        data,
        as_of=selection_cutoff,
        count=config.selected_stocks,
        years=config.correlation_years,
    )
    targets = [NIFTY, *stocks]
    assets = {}
    shared_features = build_feature_frame(
        data,
        include_next_session=False,
        selected_stocks=stocks,
        morning_features=morning_features,
    )
    fits_per_asset = 3 * (config.cv_splits + 3)
    total_fits = fits_per_asset * len(targets)
    for asset_index, symbol in enumerate(targets):
        X, y = make_dataset(
            data,
            symbol,
            selected_stocks=stocks,
            feature_frame=shared_features,
            horizon_sessions=config.horizon_sessions,
        )

        def asset_progress(
            stage: str,
            completed: int,
            _: int,
            *,
            name: str = symbol,
            offset: int = asset_index * fits_per_asset,
        ) -> None:
            if progress_callback:
                global_completed = offset + completed
                progress_callback(
                    f"{DISPLAY_NAMES.get(name, name)} — {stage}",
                    global_completed,
                    total_fits,
                )

        assets[symbol] = train_asset(X, y, config, progress_callback=asset_progress)

    latest_dates = data.groupby("Ticker")["Date"].max()
    bundle = {
        "metadata": bundle_metadata(
            config,
            targets=targets,
            target_names={symbol: DISPLAY_NAMES.get(symbol, symbol) for symbol in targets},
            stock_selection_cutoff=selection_cutoff.date().isoformat(),
            stock_correlations={k: float(v) for k, v in correlations.items()},
            data_latest_by_ticker={
                symbol: value.date().isoformat() for symbol, value in latest_dates.items()
            },
            data_latest_nifty=latest_dates[NIFTY].date().isoformat(),
            training_seconds=perf_counter() - started,
        ),
        "assets": assets,
    }
    if save:
        save_bundle(bundle)
    return bundle


def refresh_and_train(
    config: TrainingConfig = DEFAULT_CONFIG,
    force_full: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    data, diagnostics = refresh_market_data(
        ALL_SYMBOLS,
        start_date=config.start_date,
        cache_path=DATA_PATH,
        force_full=force_full,
    )
    nifty_calendar = pd.DatetimeIndex(
        data.loc[data["Ticker"] == NIFTY, "Date"].drop_duplicates().sort_values()
    )
    if len(nifty_calendar):
        extended_calendar = nifty_calendar.append(
            pd.DatetimeIndex([nifty_calendar.max() + pd.offsets.BDay(1)])
        )
        morning_features, morning_diagnostics = refresh_morning_features(
            extended_calendar,
            config=config,
        )
    else:
        morning_features, morning_diagnostics = pd.DataFrame(), {}
    diagnostics["morning"] = morning_diagnostics
    existing = load_bundle()
    latest_nifty = data.loc[data["Ticker"] == NIFTY, "Date"].max().date().isoformat()
    if (
        existing
        and existing.get("metadata", {}).get("schema_version") == 4
        and existing["metadata"].get("data_latest_nifty") == latest_nifty
        and existing["metadata"].get("training_config") == asdict(config)
    ):
        diagnostics["training_seconds"] = 0.0
        diagnostics["training_skipped"] = True
        signals = current_signals(existing, data, morning_features=morning_features)
        update_prediction_history(existing, data, signals, config=config)
        return existing, data, diagnostics

    bundle = train_from_data(
        data,
        config=config,
        save=True,
        progress_callback=progress_callback,
        morning_features=morning_features,
    )
    signals = current_signals(bundle, data)
    update_prediction_history(bundle, data, signals, config=config)
    diagnostics["training_seconds"] = bundle["metadata"]["training_seconds"]
    diagnostics["training_skipped"] = False
    return bundle, data, diagnostics


def current_signals(
    bundle: dict,
    data: pd.DataFrame,
    morning_features: pd.DataFrame | None = None,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    selected_stocks = list(bundle["metadata"]["targets"])[1:]
    if morning_features is None:
        morning_features = load_morning_features()
    feature_frame = build_feature_frame(
        data,
        include_next_session=True,
        selected_stocks=selected_stocks,
        morning_features=morning_features,
    )
    forecast_date = pd.Timestamp(feature_frame.index[-1]).normalize()
    if forecast_date.dayofweek != 0:
        return pd.DataFrame()

    config_values = bundle["metadata"].get("training_config", {})
    min_morning = int(
        config_values.get("min_live_morning_symbols", DEFAULT_CONFIG.min_live_morning_symbols)
    )
    latest_row = feature_frame.iloc[[-1]]
    morning_count_value = latest_row.get("morning_available_symbols", pd.Series([0.0])).iloc[0]
    morning_count = int(morning_count_value) if pd.notna(morning_count_value) else 0
    morning_ready = morning_count >= min_morning
    now = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    now_ist = now.tz_convert(IST)
    today_ist = now_ist.normalize().tz_localize(None)
    before_cutoff_end = (now_ist.hour, now_ist.minute) <= (9, 0)
    recordable = bool(forecast_date == today_ist and morning_ready)
    actionable = bool(recordable and before_cutoff_end)
    rows = []
    for symbol, asset in bundle["assets"].items():
        usable = feature_frame.reindex(columns=asset["feature_names"])
        prediction = weighted_predict(asset, usable.iloc[[-1]])
        threshold = asset["threshold"]
        model_policy_enabled = asset.get("policy_enabled", True)
        live_input_gate = morning_ready if forecast_date == today_ist else True
        rows.append(
            {
                "Ticker": symbol,
                "Asset": DISPLAY_NAMES.get(symbol, symbol),
                "For session": latest_row.index[-1].date().isoformat(),
                "Signal": signal_from_prediction(
                    prediction,
                    threshold,
                    enabled=model_policy_enabled and live_input_gate,
                ),
                "Predicted return": prediction,
                "Decision threshold": threshold,
                "Edge / threshold": abs(prediction) / threshold if threshold else 0.0,
                "Policy enabled": model_policy_enabled,
                "Morning inputs ready": morning_ready,
                "Morning symbols": morning_count,
                "Recordable": recordable,
                "Actionable": actionable and model_policy_enabled,
            }
        )
    return pd.DataFrame(rows)
