from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DEFAULT_CONFIG, MORNING_CONTEXT, MORNING_FEATURES_PATH, TrainingConfig

LOGGER = logging.getLogger(__name__)
IST = "Asia/Kolkata"


def _safe_name(symbol: str) -> str:
    return symbol.replace("^", "IDX_").replace("=", "_").replace("-", "_").replace(".", "_")


def _normalise_intraday(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Close"])

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        ticker_first = bool(set(raw.columns.get_level_values(0)).intersection(symbols))
        for symbol in symbols:
            try:
                part = raw[symbol] if ticker_first else raw.xs(symbol, axis=1, level=1)
            except (KeyError, ValueError):
                continue
            if "Close" not in part:
                continue
            frame = part[["Close"]].copy()
            frame["Ticker"] = symbol
            frames.append(frame)
    elif len(symbols) == 1 and "Close" in raw:
        frame = raw[["Close"]].copy()
        frame["Ticker"] = symbols[0]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Close"])
    result = pd.concat(frames).reset_index()
    result = result.rename(columns={result.columns[0]: "Timestamp"})
    result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="coerce", utc=True)
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return (
        result[["Timestamp", "Ticker", "Close"]]
        .dropna()
        .sort_values(["Timestamp", "Ticker"])
        .drop_duplicates(["Timestamp", "Ticker"], keep="last")
    )


def download_morning_intraday(symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or list(MORNING_CONTEXT)
    kwargs = {
        "tickers": symbols,
        "period": "60d",
        "interval": "5m",
        "group_by": "ticker",
        "auto_adjust": True,
        "repair": False,
        "keepna": False,
        "progress": False,
        "threads": False,
        "timeout": 20,
    }
    try:
        from curl_cffi import requests as curl_requests

        with curl_requests.Session(impersonate="chrome") as session:
            raw = yf.download(session=session, **kwargs)
    except (ImportError, TypeError):
        raw = yf.download(**kwargs)
    return _normalise_intraday(raw, symbols)


def build_morning_feature_history(
    intraday: pd.DataFrame,
    nifty_calendar: pd.DatetimeIndex,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Create point-in-time features using prices observed no later than 08:55 IST."""
    if intraday.empty or len(nifty_calendar) < 2:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))

    calendar = pd.DatetimeIndex(nifty_calendar).sort_values()
    calendar = calendar[~calendar.duplicated()]
    intraday_start = intraday["Timestamp"].min().tz_convert(IST).tz_localize(None).normalize()
    intraday_end = intraday["Timestamp"].max().tz_convert(IST).tz_localize(None).normalize()
    lower_bound = (intraday_start - timedelta(days=7)).value
    upper_bound = (intraday_end + timedelta(days=1)).value
    calendar = calendar[(calendar.asi8 >= lower_bound) & (calendar.asi8 <= upper_bound)]
    if len(calendar) < 2:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    rows: list[dict[str, float | pd.Timestamp]] = []
    for position in range(1, len(calendar)):
        date = pd.Timestamp(calendar[position]).normalize()
        previous_session = pd.Timestamp(calendar[position - 1]).normalize()
        cutoff = pd.Timestamp(date, tz=IST).replace(
            hour=config.morning_cutoff_hour_ist,
            minute=config.morning_cutoff_minute_ist,
        )
        prior_nifty_close = pd.Timestamp(previous_session, tz=IST).replace(hour=15, minute=30)
        session_start = pd.Timestamp(date, tz=IST)
        row: dict[str, float | pd.Timestamp] = {"Date": date}
        available_symbols = 0
        ages = []

        for symbol in MORNING_CONTEXT:
            prices = intraday.loc[intraday["Ticker"] == symbol, ["Timestamp", "Close"]]
            prices = prices[prices["Timestamp"] <= cutoff.tz_convert("UTC")]
            if prices.empty:
                continue
            current_row = prices.iloc[-1]
            age_minutes = (
                cutoff.tz_convert("UTC") - pd.Timestamp(current_row["Timestamp"])
            ).total_seconds() / 60
            if age_minutes < 0 or age_minutes > config.morning_max_age_minutes:
                continue

            day_prices = prices[prices["Timestamp"] >= session_start.tz_convert("UTC")]
            reference_prices = prices[prices["Timestamp"] <= prior_nifty_close.tz_convert("UTC")]
            safe = _safe_name(symbol)
            current = float(current_row["Close"])
            if not day_prices.empty:
                row[f"{safe}__morning_session_return"] = (
                    current / float(day_prices.iloc[0]["Close"]) - 1
                )
            if not reference_prices.empty:
                row[f"{safe}__since_previous_nifty_close"] = (
                    current / float(reference_prices.iloc[-1]["Close"]) - 1
                )
            row[f"{safe}__morning_age_minutes"] = float(age_minutes)
            available_symbols += 1
            ages.append(age_minutes)

        if available_symbols:
            row["morning_available_symbols"] = float(available_symbols)
            row["morning_oldest_price_minutes"] = float(max(ages))
            rows.append(row)

    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    result = pd.DataFrame(rows).set_index("Date").sort_index()
    result.index = pd.DatetimeIndex(result.index).tz_localize(None)
    result.index.name = "Date"
    return result.replace([np.inf, -np.inf], np.nan)


def load_morning_features(path: Path = MORNING_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(index=pd.DatetimeIndex([], name="Date"))
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return frame


def refresh_morning_features(
    nifty_calendar: pd.DatetimeIndex,
    path: Path = MORNING_FEATURES_PATH,
    config: TrainingConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict]:
    """Refresh the rolling Yahoo intraday window while preserving older snapshots."""
    cached = load_morning_features(path)
    try:
        intraday = download_morning_intraday()
        fresh = build_morning_feature_history(intraday, nifty_calendar, config=config)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Morning intraday refresh failed: %s", exc)
        intraday = pd.DataFrame()
        fresh = pd.DataFrame()

    if fresh.empty:
        combined = cached.copy()
    elif cached.empty:
        combined = fresh.copy()
    else:
        combined = pd.concat([cached, fresh]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

    if not combined.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        combined.reset_index().to_csv(temp, index=False, compression="gzip")
        temp.replace(path)

    diagnostics = {
        "intraday_rows": len(intraday),
        "fresh_morning_dates": len(fresh),
        "cached_morning_dates": len(combined),
        "used_morning_cache_fallback": bool(fresh.empty and not cached.empty),
        "latest_morning_date": (
            combined.index.max().date().isoformat() if not combined.empty else None
        ),
    }
    return combined, diagnostics
