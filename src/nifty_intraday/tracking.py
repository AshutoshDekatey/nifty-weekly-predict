from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PREDICTION_HISTORY_PATH, TrainingConfig
from .features import build_target

HISTORY_COLUMNS = [
    "Created UTC",
    "Ticker",
    "Asset",
    "Forecast session estimate",
    "Model trained through",
    "Signal",
    "Predicted return",
    "Decision threshold",
    "Actual session",
    "Actual return",
    "Absolute error",
    "Direction correct",
    "Net strategy return",
    "Settled UTC",
]


def load_prediction_history(path: Path = PREDICTION_HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    history = pd.read_csv(path)
    for column in HISTORY_COLUMNS:
        if column not in history:
            history[column] = pd.NA
    return history[HISTORY_COLUMNS]


def _settle_pending(
    history: pd.DataFrame,
    data: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    if history.empty:
        return history

    for column in ["Actual session", "Direction correct", "Settled UTC"]:
        history[column] = history[column].astype("object")
    settled_at = datetime.now(UTC).isoformat()
    cost = config.round_trip_cost_bps / 10_000
    for index, row in history[history["Actual return"].isna()].iterrows():
        forecast_session = pd.Timestamp(row["Forecast session estimate"])
        symbol = str(row["Ticker"])
        target = build_target(data, symbol, config.horizon_sessions).dropna()
        if forecast_session not in target.index:
            continue

        actual_return = float(target.loc[forecast_session])
        symbol_dates = (
            data.loc[data["Ticker"] == symbol, "Date"].drop_duplicates().sort_values().tolist()
        )
        try:
            start_position = symbol_dates.index(forecast_session)
            actual_session = pd.Timestamp(
                symbol_dates[start_position + config.horizon_sessions - 1]
            )
        except (ValueError, IndexError):
            continue
        signal = str(row["Signal"])
        action = {"BUY": 1, "SELL": -1, "HOLD": 0}.get(signal, 0)
        history.at[index, "Actual session"] = actual_session.date().isoformat()
        history.at[index, "Actual return"] = actual_return
        history.at[index, "Absolute error"] = abs(actual_return - float(row["Predicted return"]))
        history.at[index, "Direction correct"] = (
            bool(np.sign(actual_return) == action) if action else pd.NA
        )
        history.at[index, "Net strategy return"] = action * actual_return - (
            cost if action else 0.0
        )
        history.at[index, "Settled UTC"] = settled_at
    return history


def update_prediction_history(
    bundle: dict,
    data: pd.DataFrame,
    signals: pd.DataFrame,
    path: Path = PREDICTION_HISTORY_PATH,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Settle old live forecasts, then record one forecast from the newly fitted model."""
    history = _settle_pending(load_prediction_history(path), data, config)
    created_at = datetime.now(UTC).isoformat()
    new_rows = []
    for _, signal in signals.iterrows():
        if not bool(signal.get("Recordable", False)):
            continue
        symbol = str(signal["Ticker"])
        trained_through = bundle["assets"][symbol]["trained_through"]
        duplicate = (
            (history["Ticker"] == symbol)
            & (history["Forecast session estimate"].astype(str) == str(signal["For session"]))
        ).any()
        if duplicate:
            continue
        new_rows.append(
            {
                "Created UTC": created_at,
                "Ticker": symbol,
                "Asset": signal["Asset"],
                "Forecast session estimate": signal["For session"],
                "Model trained through": trained_through,
                "Signal": signal["Signal"],
                "Predicted return": float(signal["Predicted return"]),
                "Decision threshold": float(signal["Decision threshold"]),
                "Actual session": pd.NA,
                "Actual return": pd.NA,
                "Absolute error": pd.NA,
                "Direction correct": pd.NA,
                "Net strategy return": pd.NA,
                "Settled UTC": pd.NA,
            }
        )

    if new_rows and history.empty:
        history = pd.DataFrame(new_rows, columns=HISTORY_COLUMNS)
    elif new_rows:
        history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)
    history = history[HISTORY_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    history.to_csv(temp, index=False)
    temp.replace(path)
    return history
