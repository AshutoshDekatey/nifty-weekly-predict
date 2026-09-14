from __future__ import annotations

import pandas as pd

from nifty_intraday import data as data_module
from nifty_intraday.data import _normalise_download, refresh_market_data


def test_normalise_ticker_first_multiindex() -> None:
    dates = pd.date_range("2025-01-01", periods=2)
    columns = pd.MultiIndex.from_product([["^NSEI", "RELIANCE.NS"], ["Open", "Close"]])
    raw = pd.DataFrame([[100, 101, 200, 202], [101, 102, 203, 204]], index=dates, columns=columns)
    result = _normalise_download(raw, ["^NSEI", "RELIANCE.NS"])
    assert set(result["Ticker"]) == {"^NSEI", "RELIANCE.NS"}
    assert len(result) == 4


def test_failed_incremental_refresh_preserves_cache(
    synthetic_market_data: pd.DataFrame, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "market_data.csv.gz"
    cached = synthetic_market_data[synthetic_market_data["Ticker"] == "^NSEI"].copy()
    cached.to_csv(path, index=False, compression="gzip")

    monkeypatch.setattr(
        data_module,
        "_download_once",
        lambda symbols, start, end: pd.DataFrame(columns=cached.columns),
    )
    monkeypatch.setattr(data_module.time, "sleep", lambda _: None)
    result, diagnostics = refresh_market_data(
        symbols=["^NSEI"], cache_path=path, start_date="2018-01-01"
    )

    assert len(result) == len(cached)
    assert diagnostics["used_cache_fallback"] is True
