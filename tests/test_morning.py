from __future__ import annotations

import pandas as pd

from nifty_intraday.config import TrainingConfig
from nifty_intraday.morning import IST, build_morning_feature_history


def test_morning_features_never_read_after_cutoff() -> None:
    calendar = pd.DatetimeIndex(["2026-09-11", "2026-09-14"])
    times = [
        pd.Timestamp("2026-09-11 15:30", tz=IST).tz_convert("UTC"),
        pd.Timestamp("2026-09-14 06:30", tz=IST).tz_convert("UTC"),
        pd.Timestamp("2026-09-14 08:50", tz=IST).tz_convert("UTC"),
        pd.Timestamp("2026-09-14 09:05", tz=IST).tz_convert("UTC"),
    ]
    intraday = pd.DataFrame(
        {
            "Timestamp": times,
            "Ticker": ["BTC-USD"] * 4,
            "Close": [100.0, 102.0, 104.0, 999.0],
        }
    )
    config = TrainingConfig(morning_max_age_minutes=180)
    result = build_morning_feature_history(intraday, calendar, config=config)

    monday = result.loc[pd.Timestamp("2026-09-14")]
    assert monday["BTC_USD__morning_session_return"] == 104.0 / 102.0 - 1
    assert monday["BTC_USD__since_previous_nifty_close"] == 104.0 / 100.0 - 1
    assert monday["morning_available_symbols"] == 1
