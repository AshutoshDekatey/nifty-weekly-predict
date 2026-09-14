from __future__ import annotations

import pandas as pd

from nifty_intraday.config import NIFTY
from nifty_intraday.features import (
    build_feature_frame,
    build_target,
    make_dataset,
    select_correlated_stocks,
)


def test_same_day_close_cannot_change_same_day_features(
    synthetic_market_data: pd.DataFrame,
) -> None:
    original = build_feature_frame(synthetic_market_data)
    changed = synthetic_market_data.copy()
    target_date = changed["Date"].drop_duplicates().iloc[500]
    mask = (changed["Date"] == target_date) & (changed["Ticker"] == NIFTY)
    changed.loc[mask, "Close"] *= 1.25
    rebuilt = build_feature_frame(changed)

    pd.testing.assert_series_equal(original.loc[target_date], rebuilt.loc[target_date])
    next_date = original.index[original.index.get_loc(target_date) + 1]
    assert not original.loc[next_date].equals(rebuilt.loc[next_date])


def test_same_day_nikkei_close_cannot_change_preopen_features(
    synthetic_market_data: pd.DataFrame,
) -> None:
    original = build_feature_frame(synthetic_market_data)
    changed = synthetic_market_data.copy()
    target_date = changed["Date"].drop_duplicates().iloc[500]
    mask = (changed["Date"] == target_date) & (changed["Ticker"] == "^N225")
    changed.loc[mask, "Close"] *= 1.50
    rebuilt = build_feature_frame(changed)

    pd.testing.assert_series_equal(original.loc[target_date], rebuilt.loc[target_date])


def test_next_session_row_is_created(synthetic_market_data: pd.DataFrame) -> None:
    normal = build_feature_frame(synthetic_market_data)
    forecast = build_feature_frame(synthetic_market_data, include_next_session=True)
    assert len(forecast) == len(normal) + 1
    assert forecast.index[-1] > normal.index[-1]


def test_dataset_and_selection(synthetic_market_data: pd.DataFrame) -> None:
    X, y = make_dataset(synthetic_market_data, NIFTY)
    chosen, ranking = select_correlated_stocks(
        synthetic_market_data, as_of=X.index[-1], count=3, years=3
    )
    assert len(X) == len(y)
    assert len(chosen) == 3
    assert ranking.is_monotonic_decreasing
    assert "RELIANCE_NS__intraday_20" in X.columns
    assert "RELIANCE_NS__overnight_gap_1" in X.columns
    assert "RELIANCE_NS__nifty_beta_60" in X.columns
    assert "IDX_GSPC__ret_1" in X.columns
    assert "is_monday" in X.columns
    assert "calendar_gap_days" in X.columns


def test_five_session_target_uses_entry_open_and_fifth_close(
    synthetic_market_data: pd.DataFrame,
) -> None:
    nifty = synthetic_market_data[synthetic_market_data["Ticker"] == NIFTY].set_index("Date")
    target = build_target(synthetic_market_data, NIFTY, horizon_sessions=5)
    expected = nifty.iloc[4]["Close"] / nifty.iloc[0]["Open"] - 1
    assert target.iloc[0] == expected
    assert target.tail(4).isna().all()
