from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nifty_intraday.config import MARKET_CONTEXT, NIFTY, STOCK_CANDIDATES


@pytest.fixture()
def synthetic_market_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2018-01-01", periods=950)
    market_shock = rng.normal(0.0003, 0.009, len(dates))
    intraday_factor = rng.normal(0.0001, 0.006, len(dates))
    symbols = [NIFTY, *list(STOCK_CANDIDATES)[:5], *list(MARKET_CONTEXT)[:5], "^N225"]
    rows = []
    betas = {symbol: 0.95 - i * 0.10 for i, symbol in enumerate(list(STOCK_CANDIDATES)[:5])}
    for symbol in symbols:
        beta = 1.0 if symbol == NIFTY else betas.get(symbol, 0.45)
        close_returns = beta * market_shock + rng.normal(0, 0.005, len(dates))
        closes = 100 * np.cumprod(1 + close_returns)
        overnight = rng.normal(0, 0.002, len(dates))
        opens = closes / (1 + beta * intraday_factor + rng.normal(0, 0.003, len(dates)))
        opens *= 1 + overnight
        for dt, open_price, close_price in zip(dates, opens, closes):
            rows.append(
                {
                    "Date": dt,
                    "Ticker": symbol,
                    "Open": open_price,
                    "High": max(open_price, close_price) * 1.002,
                    "Low": min(open_price, close_price) * 0.998,
                    "Close": close_price,
                    "Volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows).sort_values(["Date", "Ticker"]).reset_index(drop=True)
