from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from nifty_intraday.config import (
    ALL_SYMBOLS,
    DEFAULT_CONFIG,
    DISPLAY_NAMES,
    NIFTY,
)
from nifty_intraday.data import (
    data_diagnostics,
    latest_remote_session,
    load_cached_data,
    refresh_market_data,
)
from nifty_intraday.morning import refresh_morning_features
from nifty_intraday.pipeline import (
    current_signals,
    load_bundle,
    refresh_and_train,
)
from nifty_intraday.tracking import load_prediction_history

st.set_page_config(page_title="NIFTY Five-Session ML Lab", page_icon="↗", layout="wide")


def _business_days_old(date_text: str) -> int:
    then = np.datetime64(date_text, "D")
    today = np.datetime64(datetime.now(UTC).date(), "D")
    return max(0, int(np.busday_count(then, today)))


def _percent(value: float) -> str:
    return f"{value * 100:.3f}%"


def _comparison_table(frame: pd.DataFrame, live: bool) -> pd.DataFrame:
    if live:
        result = frame.rename(
            columns={
                "Actual session": "Session",
                "Signal": "Prediction",
                "Predicted return": "Predicted return",
                "Actual return": "Actual return",
                "Absolute error": "Absolute error",
                "Direction correct": "Direction correct",
            }
        )
    else:
        result = frame.rename(
            columns={
                "Date": "Session",
                "predicted_return": "Predicted return",
                "actual_return": "Actual return",
                "absolute_error": "Absolute error",
                "direction_correct": "Direction correct",
            }
        ).copy()
        result["Prediction"] = frame["action"].map({1: "BUY", -1: "SELL", 0: "HOLD"})

    result = result[
        [
            "Session",
            "Prediction",
            "Predicted return",
            "Actual return",
            "Absolute error",
            "Direction correct",
        ]
    ].copy()
    for column in ["Predicted return", "Actual return", "Absolute error"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").map(
            lambda value: f"{value * 100:.3f}%" if pd.notna(value) else "—"
        )

    def direction_label(value: object) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, str):
            return "Yes" if value.strip().lower() == "true" else "No"
        return "Yes" if bool(value) else "No"

    result["Direction correct"] = result["Direction correct"].map(direction_label)
    return result.tail(10).sort_values("Session", ascending=False)


@st.cache_resource(show_spinner=False)
def _load_model_cached(modified_ns: int) -> dict | None:
    del modified_ns
    return load_bundle()


def _get_bundle() -> dict | None:
    from nifty_intraday.config import MODEL_PATH

    modified = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.exists() else 0
    return _load_model_cached(modified)


st.title("NIFTY Five-Session ML Lab")
st.caption(
    "A leakage-aware weekly research system: enter Monday at the open and exit after five "
    "sessions. It trains on every rolling five-session outcome but issues live signals only Monday. "
    "No candlestick chart, no broker connection, no automatic order placement."
)

with st.sidebar:
    st.header("Model controls")
    st.write(
        "Yahoo data is cached locally; refreshes overlap recent dates and keep the last good cache."
    )
    refresh_only = st.button("Refresh daily + morning data", width="stretch")
    retrain = st.button("Refresh + retrain / capture", type="primary", width="stretch")
    remote_check = st.button("Check Yahoo freshness", width="stretch")
    st.caption(
        f"Training starts {DEFAULT_CONFIG.start_date}. Assumed round-trip cost: "
        f"{DEFAULT_CONFIG.round_trip_cost_bps:.0f} bps."
    )

if refresh_only:
    with st.spinner("Refreshing Yahoo Finance daily and 08:55 IST inputs…"):
        try:
            refreshed_data, refresh_info = refresh_market_data()
            calendar = pd.DatetimeIndex(
                refreshed_data.loc[refreshed_data["Ticker"] == NIFTY, "Date"]
                .drop_duplicates()
                .sort_values()
            )
            if len(calendar):
                calendar = calendar.append(pd.DatetimeIndex([calendar.max() + pd.offsets.BDay(1)]))
                _, morning_info = refresh_morning_features(calendar)
                refresh_info["morning"] = morning_info
            st.session_state["refresh_info"] = refresh_info
            st.success(f"Cache refreshed: {refresh_info['fresh_rows']:,} rows received.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refresh failed safely; the prior cache was not discarded. {exc}")

if retrain:
    training_status = st.status("Downloading and validating Yahoo data…", expanded=True)
    training_progress = st.progress(0, text="Waiting for model training to begin")

    def show_training_progress(stage: str, completed: int, total: int) -> None:
        percentage = min(100, round(100 * completed / max(1, total)))
        training_progress.progress(percentage, text=f"{stage} ({completed}/{total} fits)")

    try:
        _, _, refresh_info = refresh_and_train(progress_callback=show_training_progress)
        _load_model_cached.clear()
        st.session_state["refresh_info"] = refresh_info
        elapsed = refresh_info["training_seconds"]
        if refresh_info.get("training_skipped"):
            training_progress.progress(100, text="No new NIFTY bar; current models reused")
            training_status.update(
                label="Models are already current", state="complete", expanded=False
            )
            st.success("No new completed NIFTY session was found, so retraining was skipped.")
        else:
            training_progress.progress(100, text=f"Training complete in {elapsed:.1f} seconds")
            training_status.update(label="Training complete", state="complete", expanded=False)
            st.success(f"Models updated successfully in {elapsed:.1f} seconds.")
    except Exception as exc:  # noqa: BLE001
        training_status.update(label="Training failed", state="error", expanded=True)
        st.error(f"Retraining did not complete successfully. {exc}")

if remote_check:
    with st.spinner("Checking the latest NIFTY session available from Yahoo…"):
        st.session_state["remote_latest"] = latest_remote_session()

bundle = _get_bundle()
try:
    data = load_cached_data()
except Exception as exc:  # noqa: BLE001
    data = pd.DataFrame()
    st.error(f"The local market-data cache is unreadable: {exc}")

if bundle is None:
    st.info(
        "No trained model is packaged yet. Click **Refresh + retrain / capture**. The first run downloads "
        "history and can take several minutes; later runs are incremental."
    )
    st.stop()

if bundle.get("metadata", {}).get("schema_version") != 4:
    st.warning(
        "An older daily-strategy model bundle was found. Click **Refresh + retrain / capture** "
        "once to create the V0.3.0 five-session model before using this dashboard."
    )
    st.stop()

metadata = bundle["metadata"]
trained_latest = metadata["data_latest_nifty"]
cached_latest = None
if not data.empty and "^NSEI" in set(data["Ticker"]):
    cached_latest = data.loc[data["Ticker"] == "^NSEI", "Date"].max().date().isoformat()
remote_latest = st.session_state.get("remote_latest")

stale_reasons = []
if cached_latest and cached_latest > trained_latest:
    stale_reasons.append(f"cache has {cached_latest}, model ends {trained_latest}")
if remote_latest is not None and remote_latest.date().isoformat() > trained_latest:
    stale_reasons.append(f"Yahoo has {remote_latest.date().isoformat()}")
if _business_days_old(trained_latest) > 3:
    stale_reasons.append("daily source data is more than three weekdays behind today's date")

status_col, data_col, trained_col = st.columns(3)
status_col.metric("Freshness", "UPDATE NEEDED" if stale_reasons else "CURRENT")
data_col.metric("Latest NIFTY bar", cached_latest or "No cache")
trained_col.metric("Model trained through", trained_latest)
if stale_reasons:
    st.warning(
        "Model may be stale: " + "; ".join(stale_reasons) + ". Use Refresh + retrain / capture."
    )
elif remote_latest is None:
    st.caption(
        "Fresh relative to the local cache. Use “Check Yahoo freshness” for a live source probe."
    )

if data.empty:
    st.warning(
        "A model exists but its market-data cache is absent. Refresh Yahoo data to create signals."
    )
    st.stop()

try:
    signals = current_signals(bundle, data)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not construct the latest signal row: {exc}")
    st.stop()

st.subheader("Monday five-session paper signals")
if signals.empty:
    st.info(
        "No signal is issued for the next session because it is not Monday. The models may still "
        "retrain after every completed session as a new rolling five-session label becomes known."
    )
else:
    signal_columns = st.columns(len(signals))
    for column, (_, row) in zip(signal_columns, signals.iterrows()):
        with column:
            st.markdown(f"#### {row['Asset']}")
            st.metric(row["Signal"], _percent(row["Predicted return"]))
            timing = "ACTIONABLE" if row["Actionable"] else "PREVIEW / PAPER ONLY"
            st.caption(
                f"Entry {row['For session']} · {timing} · "
                f"{row['Morning symbols']} fresh morning sources · "
                f"threshold ±{_percent(row['Decision threshold'])}"
            )
            if not row["Morning inputs ready"]:
                st.caption("Morning-data gate not met → a same-day signal is forced to HOLD")
            if not row["Policy enabled"]:
                st.caption("Validation gate failed → forced HOLD")

selected = st.selectbox(
    "Inspect an asset",
    options=list(bundle["assets"]),
    format_func=lambda symbol: DISPLAY_NAMES.get(symbol, symbol),
)
asset = bundle["assets"][selected]
metrics = asset["test_metrics"]

last_ten_tab, test_tab, learning_tab = st.tabs(
    ["Last 10 Monday predictions", "15% untouched test", "Ongoing learning"]
)

with last_ten_tab:
    history = load_prediction_history()
    live_rows = history[(history["Ticker"] == selected) & history["Actual return"].notna()]
    if not live_rows.empty:
        st.markdown("**Forward-generated live results**")
        st.dataframe(
            _comparison_table(live_rows, live=True),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Live ledger: each row was captured from information available by the Monday cutoff, "
            "then settled after the fifth session close."
        )
        if len(live_rows) < 10:
            st.markdown("**Last 10 untouched-test results**")
            st.dataframe(
                _comparison_table(asset["holdout"], live=False),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                f"Only {len(live_rows)} live result(s) have settled so far; this second table "
                "keeps a complete 10-Monday historical comparison visible."
            )
    else:
        st.dataframe(
            _comparison_table(asset["holdout"], live=False),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "No live predictions have settled yet, so these are the final 10 Monday observations "
            "from the untouched historical test. The live ledger replaces this view as forecasts mature."
        )

with test_tab:
    metric_cols = st.columns(6)
    metric_cols[0].metric("Active accuracy", f"{metrics['active_directional_accuracy']:.1%}")
    metric_cols[1].metric("Trades", f"{metrics['trades']}")
    metric_cols[2].metric("Coverage", f"{metrics['coverage']:.1%}")
    metric_cols[3].metric("Net Sharpe", f"{metrics['net_sharpe']:.2f}")
    metric_cols[4].metric("Max drawdown", f"{metrics['max_drawdown']:.1%}")
    metric_cols[5].metric("MAE skill vs zero", f"{metrics['mae_skill_vs_zero']:.1%}")
    st.caption(
        f"Held-out final 15%: {asset['test_start']} to {asset['test_end']}. Metrics use only "
        "Monday entries; costs are deducted once per active five-session trade. Results are "
        "historical, not a promise of future performance."
    )
    holdout = asset["holdout"].set_index("Date")
    st.line_chart(holdout[["equity"]], y_label="Growth of ₹1 (paper strategy)")

with learning_tab:
    st.metric("Overlapping five-session labels in final refit", f"{asset['n_rows']:,}")
    st.metric("Monday labels in history", f"{asset['n_monday_rows']:,}")
    if metadata.get("training_seconds") is not None:
        st.metric("Last complete training runtime", f"{metadata['training_seconds']:.1f} seconds")
    st.write(
        f"Every successful post-close run appends prices. Once a fifth future close becomes "
        f"available, yesterday's oldest rolling label is completed. The app reruns five purged "
        f"walk-forward folds, updates model weights and the HOLD threshold, and refits production "
        f"models through **{asset['trained_through']}**. All rows remain in "
        f"training, but their influence decays with a "
        f"**{DEFAULT_CONFIG.recency_half_life_sessions}-session half-life**, so recent regimes matter more."
    )
    weights = pd.Series(asset["weights"], name="Current ensemble weight")
    st.dataframe(weights.to_frame(), width="stretch")
    st.caption(
        "This is genuine daily updating of a weekly-horizon model, not a static saved prediction. "
        "It can adapt as data arrives, "
        "although improvement is measured—not guaranteed."
    )

with st.expander("How the model was validated"):
    st.write(
        "History is divided chronologically into 70% training, 15% validation, and a final 15% "
        "untouched test. Inside the first 70%, five expanding walk-forward folds compare Ridge, "
        "shallow gradient boosting, and Extra Trees. Every boundary has a five-session purge. "
        "Weights come from training-fold error; the HOLD threshold is selected only on validation "
        "Mondays. The final test is evaluated only on Mondays. After that honest report, deployment "
        "models are refitted on every labelled row with greater weight on recent regimes."
    )
    fold_table = pd.DataFrame(asset["fold_metrics"])
    st.dataframe(fold_table, hide_index=True, width="stretch")

with st.expander("Exactly what enters the model"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Source": "NIFTY + 3 selected Indian stocks",
                    "Lagged features": "Prior returns, 5/20-day momentum, volatility, open-to-close history and overnight gaps",
                },
                {
                    "Source": "Selected stock relationships",
                    "Lagged features": "NIFTY relative strength, rolling correlation and beta, market breadth",
                },
                {
                    "Source": "World indices and Monday Asian sessions",
                    "Lagged features": "Prior returns/momentum/volatility, same-day Asian opening gaps, and 08:55 IST partial-session moves",
                },
                {
                    "Source": "US Treasury yields and bond futures",
                    "Lagged features": "Returns, yield changes, yield level and volatility",
                },
                {
                    "Source": "Gold, oil, copper, FX, VIX and Bitcoin",
                    "Lagged features": "Returns, momentum, volatility, weekend BTC response and 08:55 IST futures/crypto moves",
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "The predicted outcome is Close after five sessions / entry-session Open − 1. Daily-close "
        "features are lagged; only markets already observable before 08:55 IST may contribute "
        "same-date information."
    )

with st.expander("Selected stocks and correlations"):
    ranking = pd.Series(metadata["stock_correlations"], name="Correlation with NIFTY")
    ranking.index.name = "Ticker"
    st.dataframe(ranking.to_frame(), width="stretch")
    st.caption(f"Selection used only information through {metadata['stock_selection_cutoff']}.")

with st.expander("Data diagnostics"):
    st.dataframe(data_diagnostics(data, ALL_SYMBOLS), hide_index=True, width="stretch")
    if metadata.get("data_latest_by_ticker"):
        st.caption(
            f"Last trained {metadata['trained_at_utc']}. Yahoo refresh diagnostics are retained "
            "without silently substituting synthetic data."
        )

st.divider()
st.caption(
    "Research and education only—not investment advice. NIFTY 50 (^NSEI) is an index, not a "
    "tradable security; live implementation would require an ETF or derivative with its own costs, "
    "tracking error, liquidity, broker rules, and risk controls. Yahoo/yfinance is an unofficial "
    "personal-use data route and can be delayed or rate-limited."
)
