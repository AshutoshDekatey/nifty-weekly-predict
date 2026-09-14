from __future__ import annotations

import pandas as pd

from nifty_intraday.config import NIFTY, TrainingConfig
from nifty_intraday.pipeline import current_signals, train_from_data


class ConstantModel:
    def predict(self, frame):
        return [0.02] * len(frame)


def test_end_to_end_bundle_and_signals(synthetic_market_data: pd.DataFrame) -> None:
    config = TrainingConfig(
        min_training_rows=500,
        cv_splits=3,
        correlation_years=2,
        selected_stocks=3,
        gradient_boosting_estimators=20,
        extra_trees_estimators=20,
    )
    progress = []
    bundle = train_from_data(
        synthetic_market_data,
        config=config,
        save=False,
        progress_callback=lambda stage, completed, total: progress.append(
            (stage, completed, total)
        ),
    )
    signals = current_signals(bundle, synthetic_market_data)

    assert bundle["metadata"]["targets"][0] == NIFTY
    assert len(bundle["assets"]) == 4
    assert len(signals) == 4
    assert set(signals["Signal"]).issubset({"BUY", "SELL", "HOLD"})
    assert signals["For session"].nunique() == 1
    assert progress[-1][1] == progress[-1][2] == 72
    assert bundle["metadata"]["training_seconds"] > 0
    assert bundle["metadata"]["training_config"]["horizon_sessions"] == 5


def test_monday_signal_is_actionable_only_with_fresh_morning_inputs(
    synthetic_market_data: pd.DataFrame,
) -> None:
    monday = pd.Timestamp("2021-08-23")
    morning = pd.DataFrame(
        {"morning_available_symbols": [3.0]},
        index=pd.DatetimeIndex([monday], name="Date"),
    )
    bundle = {
        "metadata": {
            "targets": [NIFTY],
            "training_config": {"min_live_morning_symbols": 3},
        },
        "assets": {
            NIFTY: {
                "models": {"constant": ConstantModel()},
                "weights": {"constant": 1.0},
                "threshold": 0.01,
                "policy_enabled": True,
                "feature_names": ["is_monday", "morning_available_symbols"],
            }
        },
    }

    signals = current_signals(
        bundle,
        synthetic_market_data,
        morning_features=morning,
        now_utc=pd.Timestamp("2021-08-23 03:25:00", tz="UTC"),
    )

    assert signals.iloc[0]["Signal"] == "BUY"
    assert bool(signals.iloc[0]["Actionable"]) is True
    assert bool(signals.iloc[0]["Recordable"]) is True
