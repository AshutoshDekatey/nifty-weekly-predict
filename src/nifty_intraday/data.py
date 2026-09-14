from __future__ import annotations

import logging
import random
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .config import ALL_SYMBOLS, DATA_PATH, NIFTY, STOCK_CANDIDATES

LOGGER = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]


class MarketDataError(RuntimeError):
    """Raised when there is no safe market-data result to use."""


def _empty_long_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def _normalise_download(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_long_frame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        ticker_first = bool(level0.intersection(symbols))
        for symbol in symbols:
            try:
                part = raw[symbol] if ticker_first else raw.xs(symbol, axis=1, level=1)
            except (KeyError, ValueError):
                continue
            part = part.copy()
            part["Ticker"] = symbol
            frames.append(part)
    elif len(symbols) == 1:
        part = raw.copy()
        part["Ticker"] = symbols[0]
        frames.append(part)

    if not frames:
        return _empty_long_frame()

    result = pd.concat(frames).reset_index()
    date_col = result.columns[0]
    result = result.rename(columns={date_col: "Date"})
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in result:
            result[column] = pd.NA
    result = result[REQUIRED_COLUMNS]
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    numeric = ["Open", "High", "Low", "Close", "Volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["Date", "Ticker", "Open", "Close"])
    return result.sort_values(["Date", "Ticker"]).drop_duplicates(["Date", "Ticker"])


def _download_once(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    kwargs = {
        "tickers": symbols,
        "start": start,
        "end": end,
        "interval": "1d",
        "group_by": "ticker",
        "auto_adjust": True,
        "repair": True,
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
    return _normalise_download(raw, symbols)


def download_with_retries(
    symbols: list[str],
    start: str,
    end: str,
    attempts: int = 4,
) -> tuple[pd.DataFrame, list[str]]:
    """Download in one efficient request, then retry only missing symbols."""
    collected = _empty_long_frame()
    remaining = list(dict.fromkeys(symbols))

    for attempt in range(attempts):
        if not remaining:
            break
        try:
            fetched = _download_once(remaining, start, end)
            if not fetched.empty:
                collected = (
                    fetched.copy()
                    if collected.empty
                    else pd.concat([collected, fetched], ignore_index=True)
                )
                received = set(fetched["Ticker"].unique())
                remaining = [symbol for symbol in remaining if symbol not in received]
        except Exception as exc:  # noqa: BLE001  # yfinance exception types vary by version
            LOGGER.warning("Yahoo download attempt %s failed: %s", attempt + 1, exc)

        if remaining and attempt < attempts - 1:
            delay = min(2**attempt + random.random(), 10)
            time.sleep(delay)

    if not collected.empty:
        collected = (
            collected.sort_values(["Date", "Ticker"])
            .drop_duplicates(["Date", "Ticker"], keep="last")
            .reset_index(drop=True)
        )
    return collected, remaining


def load_cached_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return _empty_long_frame()
    data = pd.read_csv(path, parse_dates=["Date"])
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise MarketDataError(f"Cache is missing columns: {sorted(missing)}")
    return data[REQUIRED_COLUMNS].sort_values(["Date", "Ticker"]).reset_index(drop=True)


def refresh_market_data(
    symbols: list[str] | None = None,
    start_date: str = "2012-01-01",
    cache_path: Path = DATA_PATH,
    force_full: bool = False,
) -> tuple[pd.DataFrame, dict]:
    symbols = symbols or ALL_SYMBOLS
    cached = load_cached_data(cache_path)

    if force_full or cached.empty:
        fetch_start = start_date
    else:
        # Overlap protects against revised recent bars and partial market updates.
        fetch_start = (cached["Date"].max() - timedelta(days=10)).date().isoformat()

    today = datetime.now(UTC).date()
    end = (today + timedelta(days=2)).isoformat()  # yfinance end is exclusive
    fresh, missing = download_with_retries(symbols, fetch_start, end)

    if fresh.empty and cached.empty:
        raise MarketDataError(
            "Yahoo Finance returned no usable rows. Wait a few minutes and retry; "
            "shared/cloud IPs are occasionally rate-limited."
        )

    if fresh.empty:
        combined = cached.copy()
    elif cached.empty:
        combined = fresh.copy()
    else:
        combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined[combined["Ticker"].isin(symbols)]
    # A morning refresh can expose Yahoo's still-forming Indian daily bar. It must
    # never enter the labelled cache before the cash session has completed.
    now_ist = datetime.now(UTC).astimezone(ZoneInfo("Asia/Kolkata"))
    if (now_ist.hour, now_ist.minute) < (16, 0):
        indian_symbols = {NIFTY, *STOCK_CANDIDATES}
        incomplete_indian_bar = combined["Ticker"].isin(indian_symbols) & (
            combined["Date"].dt.date == now_ist.date()
        )
        combined = combined.loc[~incomplete_indian_bar]
    combined = (
        combined.sort_values(["Date", "Ticker"])
        .drop_duplicates(["Date", "Ticker"], keep="last")
        .reset_index(drop=True)
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    combined.to_csv(temp_path, index=False, compression="gzip")
    temp_path.replace(cache_path)

    latest_by_ticker = {
        ticker: group["Date"].max().date().isoformat()
        for ticker, group in combined.groupby("Ticker")
    }
    diagnostics = {
        "refreshed_at_utc": datetime.now(UTC).isoformat(),
        "fetch_start": fetch_start,
        "fresh_rows": len(fresh),
        "total_rows": len(combined),
        "missing_from_latest_request": missing,
        "latest_by_ticker": latest_by_ticker,
        "used_cache_fallback": bool(fresh.empty and not cached.empty),
    }
    return combined, diagnostics


def latest_remote_session(symbol: str = "^NSEI") -> pd.Timestamp | None:
    """Small probe used only for freshness; failure returns unknown, never false-fresh."""
    today = datetime.now(UTC).date()
    end = (today + timedelta(days=2)).isoformat()
    start = (today - timedelta(days=14)).isoformat()
    fresh, _ = download_with_retries([symbol], start, end, attempts=2)
    if fresh.empty:
        return None
    return pd.Timestamp(fresh["Date"].max()).normalize()


def data_diagnostics(data: pd.DataFrame, expected_symbols: list[str]) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    rows = []
    for symbol in expected_symbols:
        subset = data[data["Ticker"] == symbol]
        latest = subset["Date"].max() if not subset.empty else pd.NaT
        age = (today - latest.normalize()).days if pd.notna(latest) else None
        rows.append(
            {
                "Ticker": symbol,
                "Rows": len(subset),
                "Latest bar": latest.date().isoformat() if pd.notna(latest) else "missing",
                "Calendar age (days)": age,
                "Status": "OK" if len(subset) >= 250 else "INSUFFICIENT",
            }
        )
    return pd.DataFrame(rows)
