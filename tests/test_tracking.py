from __future__ import annotations

import pandas as pd

from nifty_intraday.config import NIFTY
from nifty_intraday.tracking import update_prediction_history


def test_live_prediction_is_recorded_then_settled(
    synthetic_market_data: pd.DataFrame, tmp_path
) -> None:
    dates = synthetic_market_data["Date"].drop_duplicates().sort_values()
    forecast_session = dates.iloc[-5]
    trained_through = dates.iloc[-6]
    first_data = synthetic_market_data[synthetic_market_data["Date"] <= dates.iloc[-2]]
    bundle = {"assets": {NIFTY: {"trained_through": trained_through.date().isoformat()}}}
    signals = pd.DataFrame(
        [
            {
                "Ticker": NIFTY,
                "Asset": "NIFTY 50",
                "For session": forecast_session.date().isoformat(),
                "Signal": "BUY",
                "Predicted return": 0.002,
                "Decision threshold": 0.001,
                "Recordable": True,
            }
        ]
    )
    path = tmp_path / "prediction_history.csv"

    pending = update_prediction_history(bundle, first_data, signals, path=path)
    assert len(pending) == 1
    assert pd.isna(pending.iloc[0]["Actual return"])

    settled = update_prediction_history(bundle, synthetic_market_data, signals, path=path)
    assert len(settled) == 1
    assert pd.notna(settled.iloc[0]["Actual return"])
    assert settled.iloc[0]["Actual session"] == dates.iloc[-1].date().isoformat()
