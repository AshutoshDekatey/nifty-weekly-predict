from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MARKET_CONTEXT, MORNING_CONTEXT, NIFTY, STOCK_CANDIDATES


def _wide(data: pd.DataFrame, field: str) -> pd.DataFrame:
    return data.pivot(index="Date", columns="Ticker", values=field).sort_index()


def _safe_name(symbol: str) -> str:
    return (
        symbol.replace("^", "IDX_")
        .replace("=", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("&", "AND")
    )


def _compounded_return(series: pd.Series, window: int) -> pd.Series:
    return (1 + series).rolling(window).apply(np.prod, raw=True).sub(1)


def select_correlated_stocks(
    data: pd.DataFrame,
    as_of: pd.Timestamp,
    count: int = 3,
    years: int = 3,
) -> tuple[list[str], pd.Series]:
    """Select on data ending before evaluation/live dates to avoid look-ahead bias."""
    close = _wide(data[data["Date"] <= as_of], "Close")
    start = as_of - pd.DateOffset(years=years)
    returns = close.loc[close.index >= start].pct_change(fill_method=None)
    correlations = {}
    for symbol in STOCK_CANDIDATES:
        if symbol not in returns or NIFTY not in returns:
            continue
        pair = returns[[NIFTY, symbol]].dropna()
        if len(pair) >= 250:
            correlations[symbol] = pair[NIFTY].corr(pair[symbol])
    ranked = pd.Series(correlations, dtype=float).sort_values(ascending=False)
    if len(ranked) < count:
        raise ValueError(
            f"Only {len(ranked)} stock candidates have sufficient history; need {count}."
        )
    return ranked.head(count).index.tolist(), ranked


def build_feature_frame(
    data: pd.DataFrame,
    include_next_session: bool = False,
    selected_stocks: list[str] | None = None,
    morning_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build only features available before the Indian market opens on each row date."""
    close = _wide(data, "Close")
    open_price = _wide(data, "Open")
    if NIFTY not in close:
        raise ValueError("NIFTY 50 (^NSEI) is missing from the downloaded data.")

    calendar = close[NIFTY].dropna().index
    if include_next_session and len(calendar):
        next_session = pd.Timestamp(calendar.max()) + pd.offsets.BDay(1)
        calendar = calendar.append(pd.DatetimeIndex([next_session]))
    aligned_close = close.reindex(calendar).ffill(limit=3)
    aligned_open = open_price.reindex(calendar).ffill(limit=3)
    daily_returns = aligned_close.pct_change(fill_method=None)
    intraday_returns = aligned_close.div(aligned_open).sub(1)
    overnight_gaps = aligned_open.div(aligned_close.shift(1)).sub(1)
    features: dict[str, pd.Series] = {}

    # Every source gets explicit lagged price-response features. No value from the
    # target session is available to its own prediction row.
    selected_stocks = selected_stocks or list(STOCK_CANDIDATES)
    model_symbols = [NIFTY, *selected_stocks, *MARKET_CONTEXT]
    available_symbols = [symbol for symbol in model_symbols if symbol in aligned_close]
    for symbol in available_symbols:
        safe = _safe_name(symbol)
        features[f"{safe}__ret_1"] = daily_returns[symbol].shift(1)
        features[f"{safe}__ret_2"] = aligned_close[symbol].pct_change(2, fill_method=None).shift(1)
        features[f"{safe}__mom_5"] = aligned_close[symbol].pct_change(5, fill_method=None).shift(1)
        features[f"{safe}__mom_20"] = (
            aligned_close[symbol].pct_change(20, fill_method=None).shift(1)
        )
        features[f"{safe}__vol_10"] = daily_returns[symbol].rolling(10).std().shift(1)

        if symbol in {NIFTY, *selected_stocks}:
            for window in (1, 2, 5, 20):
                features[f"{safe}__intraday_{window}"] = _compounded_return(
                    intraday_returns[symbol], window
                ).shift(1)
            features[f"{safe}__overnight_gap_1"] = overnight_gaps[symbol].shift(1)
            features[f"{safe}__overnight_gap_mean_5"] = (
                overnight_gaps[symbol].rolling(5).mean().shift(1)
            )

        if symbol in {"^TNX", "^FVX"}:
            features[f"{safe}__yield_change_1"] = aligned_close[symbol].diff().shift(1)
            rolling_mean = aligned_close[symbol].rolling(20).mean()
            rolling_std = aligned_close[symbol].rolling(20).std()
            features[f"{safe}__yield_z_20"] = (
                aligned_close[symbol].sub(rolling_mean).div(rolling_std).shift(1)
            )

    # Asian cash markets have opened before the 08:55 IST decision cutoff. Their
    # same-date opening gaps are therefore causal, unlike their eventual daily closes.
    for symbol in MORNING_CONTEXT:
        if symbol not in open_price or symbol not in close:
            continue
        same_day_open = open_price[symbol].reindex(calendar)
        prior_lookup = pd.DatetimeIndex(calendar.asi8 - 1)
        prior_close = close[symbol].reindex(prior_lookup, method="ffill")
        prior_close.index = calendar
        features[f"{_safe_name(symbol)}__same_day_open_gap"] = same_day_open.div(prior_close).sub(1)

    # Crypto trades over the weekend. A Yahoo daily candle labelled Sunday is
    # complete by 05:30 IST Monday, so it is available before this model's cutoff.
    if "BTC-USD" in close:
        btc = close["BTC-USD"].dropna().sort_index()
        completed_cutoffs = pd.DatetimeIndex(calendar.asi8 - 86_400_000_000_000)
        completed = btc.reindex(completed_cutoffs, method="ffill")
        completed.index = calendar
        previous_sessions = pd.Series(calendar, index=calendar).shift(1)
        previous_reference = pd.Series(index=calendar, dtype=float)
        for date, previous_session in previous_sessions.items():
            if pd.notna(previous_session):
                available = btc[btc.index <= previous_session]
                if not available.empty:
                    previous_reference.loc[date] = available.iloc[-1]
        features["BTC_USD__since_previous_nifty_session"] = completed.div(previous_reference).sub(1)

    stock_symbols = [symbol for symbol in selected_stocks if symbol in daily_returns]
    if stock_symbols:
        lagged_stock_returns = daily_returns[stock_symbols].shift(1)
        features["nifty_breadth_positive"] = (lagged_stock_returns > 0).mean(axis=1)
        features["nifty_breadth_mean"] = lagged_stock_returns.mean(axis=1)
        features["nifty_breadth_dispersion"] = lagged_stock_returns.std(axis=1)

        nifty_returns = daily_returns[NIFTY]
        nifty_variance_60 = nifty_returns.rolling(60).var()
        for symbol in stock_symbols:
            safe = _safe_name(symbol)
            relative = daily_returns[symbol].sub(nifty_returns)
            features[f"{safe}__relative_strength_5"] = relative.rolling(5).sum().shift(1)
            features[f"{safe}__relative_strength_20"] = relative.rolling(20).sum().shift(1)
            features[f"{safe}__nifty_corr_20"] = (
                daily_returns[symbol].rolling(20).corr(nifty_returns).shift(1)
            )
            covariance_60 = daily_returns[symbol].rolling(60).cov(nifty_returns)
            features[f"{safe}__nifty_beta_60"] = covariance_60.div(nifty_variance_60).shift(1)

    index = pd.DatetimeIndex(calendar)
    calendar_gaps = pd.Series(index, index=index).diff().dt.days
    features["is_monday"] = pd.Series((index.dayofweek == 0).astype(float), index=index)
    features["calendar_gap_days"] = calendar_gaps
    features["is_long_weekend"] = calendar_gaps.ge(4).astype(float)
    features["weekday_sin"] = pd.Series(np.sin(2 * np.pi * index.dayofweek / 5), index=index)
    features["weekday_cos"] = pd.Series(np.cos(2 * np.pi * index.dayofweek / 5), index=index)
    feature_frame = pd.DataFrame(features, index=calendar)
    if morning_features is not None and not morning_features.empty:
        feature_frame = feature_frame.join(morning_features, how="left")
    feature_frame.index.name = "Date"
    return feature_frame.replace([np.inf, -np.inf], np.nan)


def build_target(data: pd.DataFrame, symbol: str, horizon_sessions: int = 5) -> pd.Series:
    subset = data[data["Ticker"] == symbol].set_index("Date").sort_index()
    exit_close = subset["Close"].shift(-(horizon_sessions - 1))
    target = exit_close.div(subset["Open"]).sub(1.0)
    target.name = "forward_return"
    return target.replace([np.inf, -np.inf], np.nan)


def make_dataset(
    data: pd.DataFrame,
    symbol: str,
    selected_stocks: list[str] | None = None,
    feature_frame: pd.DataFrame | None = None,
    horizon_sessions: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    features = (
        feature_frame
        if feature_frame is not None
        else build_feature_frame(
            data,
            include_next_session=False,
            selected_stocks=selected_stocks,
        )
    )
    target = build_target(data, symbol, horizon_sessions=horizon_sessions)
    joined = features.join(target, how="inner").dropna(subset=["forward_return"])
    # Permit partial macro history; the model pipeline imputes from training data only.
    joined = joined.dropna(axis=1, thresh=min(20, max(5, int(len(joined) * 0.005))))
    return joined.drop(columns="forward_return"), joined["forward_return"]
