# NIFTY Five-Session ML Lab — V0.3.0

A GitHub-ready Streamlit research app that uses Yahoo Finance prices to predict a five-session
return for NIFTY 50 and three dynamically selected NIFTY-correlated stocks. It trains on every
rolling five-session window—roughly five times as many labels as Monday-only training—but produces
paper BUY/SELL/HOLD signals only for Monday entry and exit after the fifth session close.

> Research and education only. This is not investment advice or an automated trading system.
> Historical performance does not guarantee future results.

## Decision and target

For a feature row at session `T`:

```text
target[T] = adjusted Close[T+4] / adjusted Open[T] - 1
```

The app learns from every possible `T`. Live predictions are restricted to Mondays. In a normal
week that means Monday open to Friday close. In a holiday week it means the first five available
sessions, which can cross the calendar-week boundary; use an exchange calendar before any real
execution experiment.

- **BUY**: predicted return clears the positive validation threshold.
- **SELL**: predicted return clears the negative threshold; this assumes shorting is possible.
- **HOLD**: the edge or reliability gate is insufficient.

NIFTY (`^NSEI`) is a research proxy, not a directly tradable security.

## Inputs

Each target model sees the same point-in-time feature set:

- NIFTY and three liquid correlated stocks: prior returns, momentum, volatility, open-to-close
  behaviour, overnight gaps, breadth, relative strength, rolling correlation and beta;
- S&P 500, Nasdaq, Dow, FTSE, DAX, Nikkei, Hang Seng, KOSPI and ASX 200;
- US and India VIX, US 5Y/10Y yields and Treasury futures;
- gold, WTI, Brent, copper, USD/INR, Dollar Index and Bitcoin;
- day-of-week, calendar-gap and long-weekend indicators;
- Bitcoin's move since the previous NIFTY session using the latest completed pre-cutoff crypto
  daily candle;
- same-day Asian opening gaps, which are observable before India opens; and
- when Yahoo intraday data is available, 08:55 IST point-in-time moves for Asian indices, gold,
  oil, USD/INR and Bitcoin.

The app never uses the eventual same-day Asian close to predict the Indian open. Yahoo intraday
history is limited to a rolling recent window, so every retrieved 08:55 snapshot is retained in
`artifacts/morning_features.csv.gz`; its influence becomes more measurable as this ledger grows.

## Validation and all-data refit

The split is chronological—not shuffled:

1. **70% training**: five expanding `TimeSeriesSplit` folds compare Ridge, shallow Gradient
   Boosting and Extra Trees.
2. **Five-session purge**: applied inside cross-validation and at the train/validation/test
   boundaries because adjacent targets overlap.
3. **15% validation**: selects the BUY/SELL/HOLD threshold and checks whether the policy beats a
   zero-return forecast. Only Monday rows determine the trading threshold.
4. **15% untouched test**: opened once after model and threshold selection; reported strategy
   metrics use Monday rows only and deduct the configured round-trip cost.
5. **100% production refit**: after honest evaluation, all labelled rows train the deployed model
   with exponentially decaying recency weights.

Model weights are the normalized inverse out-of-fold MAE. Test results never set model weights or
thresholds. The UI shows the last 10 held-out Monday predictions against actual five-session
returns, then replaces them with genuinely forward-captured paper results as they settle.

## Why no deep learning or reinforcement learning

Overlapping windows increase row count, not independent information. Daily data since 2012 still
contains only a few major market regimes. A neural network or deep-RL agent would introduce far
more degrees of freedom than the evidence supports.

This task has a known supervised label and a fixed single decision, so Ridge plus restrained tree
models are more sample-efficient and auditable. RL becomes relevant only if later versions manage
inventory, position size, cash, stop/exit choices and portfolio constraints—and only if they beat
this baseline on rolling unseen periods after realistic costs.

## How it updates

The weekday post-close workflow:

1. appends/revises Yahoo daily bars using an overlapping refresh window;
2. settles any Monday paper trade whose fifth close is now available;
3. creates the newest rolling five-session label;
4. reruns purged cross-validation and 70/15/15 evaluation;
5. updates ensemble weights and the abstention threshold; and
6. refits production models on all labels.

A second Monday workflow runs at 08:55 IST to capture same-morning global data and write the paper
forecast. GitHub scheduled jobs can start late, so the ledger records the signal but the UI labels
it actionable only when generated no later than 09:00 IST. The Streamlit button is the manual
fallback. Hosted Streamlit files can be ephemeral; committed GitHub Action artifacts are the
persistence mechanism.

If no new NIFTY daily bar exists and the schema/configuration matches, model fitting is skipped.
Morning data can still refresh and a Monday prediction can still be recorded.

## Reliability controls

- browser-like Yahoo session, bulk downloads, bounded retries and exponential backoff;
- last-good-cache fallback and atomic artifact replacement;
- 10-day overlap for revised daily bars;
- intraday feature age limit and minimum fresh-source gate;
- median imputation fitted inside each training fold;
- forced HOLD when validation fails or same-day morning coverage is insufficient;
- data/model freshness diagnostics in Streamlit; and
- no synthetic replacement when Yahoo is unavailable.

## Run locally

### WSL / Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first **Refresh + retrain / capture** run can take several minutes. Later runs download only
the overlap and skip redundant fitting when appropriate. Command-line training is also available:

```bash
python scripts/train_latest.py
python scripts/train_latest.py --full
```

## GitHub and Streamlit deployment

1. Extract the ZIP and upload the project contents to a new GitHub repository.
2. Run **Actions → Daily Yahoo refresh and five-session retrain → Run workflow** once.
3. In Streamlit Community Cloud select the repository, `main` branch and `streamlit_app.py`.
4. Keep GitHub Actions enabled. The daily workflow persists models; the Monday workflow persists
   the point-in-time snapshot and forecast.

No API key is required. Yahoo/yfinance is an unofficial personal-use source and may be delayed or
rate-limited, especially on shared cloud IPs.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

Offline tests cover Yahoo response normalization, feature leakage, five-session targets, morning
snapshot cutoffs, correlated-stock selection, purged 70/15/15 training, Monday-only evaluation,
ensemble prediction, HOLD logic and five-session forecast settlement.

## Repository structure

```text
.
├── .github/workflows/daily_train.yml
├── .github/workflows/monday_signal.yml
├── artifacts/
├── docs/MODEL_CARD.md
├── scripts/train_latest.py
├── src/nifty_intraday/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── modeling.py
│   ├── morning.py
│   ├── pipeline.py
│   └── tracking.py
├── tests/
├── streamlit_app.py
├── requirements.txt
└── pyproject.toml
```

## Before real money

Add a point-in-time NSE calendar, a tradable ETF/future target, broker-specific auction execution,
slippage, brokerage and taxes, borrow constraints, portfolio limits, an independent reconciled
data source, drift alerts, paper trading and a kill switch. Define acceptance criteria before
looking at forward results; do not infer reliability from one backtest metric.

## References

- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [GitHub scheduled workflows](https://docs.github.com/actions/using-workflows/events-that-trigger-workflows#schedule)
