# Model card — V0.3.0

## Intended use

Educational paper-trading research for five-session returns in NIFTY 50 and three dynamically
selected NIFTY-correlated stocks. It does not place orders or provide investment advice.

## Prediction contract

For entry session `T`, the supervised target is adjusted `Close[T+4] / Open[T] - 1`. Training uses
every rolling start date; live signals are generated only for Mondays. A BUY/SELL/HOLD threshold
is selected on validation Mondays. The test report contains only Monday entries.

## Models

- Ridge: stable regularized linear baseline.
- Shallow Gradient Boosting: restrained nonlinear interactions.
- Extra Trees: a variance-reducing nonlinear alternative.

Ensemble weights are inverse recency-weighted MAE from purged expanding-window folds inside the
training segment. Separate models are retained per asset because index and stock risks differ.

## Split and leakage controls

- chronological 70% training / 15% validation / 15% untouched test;
- five expanding folds inside training;
- five-session purge inside folds and at both outer boundaries;
- preprocessing is fitted inside each model pipeline;
- stock selection ends at the training boundary;
- daily-close features are shifted one completed bar;
- same-day values are limited to Asian opening gaps and intraday prices timestamped no later than
  08:55 IST;
- thresholds and policy gates use validation Mondays only; and
- the untouched test does not select models, weights, features or thresholds.

After evaluation, production estimators are refitted on all labels with a 756-session recency
half-life. This uses all available labelled rows without contaminating the reported test.

## Why overlapping rows are not independent data

Adjacent labels share four of five sessions. They provide more feature/target alignments but not
five times as many independent market regimes. The purge prevents direct outcome overlap across
fold boundaries; Monday-only scoring prevents the denser training grid from exaggerating the
actual weekly trading frequency.

## Why no deep learning or RL

The dataset remains small in regime count and has a directly observed supervised target. Deep
models are unnecessarily flexible. RL is a poor match to one fixed entry/exit decision because the
action does not influence future market state. Reconsider RL only for sequential position sizing,
inventory, exits and portfolio constraints, then require superiority over this baseline on unseen
rolling periods after costs.

## Adaptation and monitoring

Each completed NIFTY session can finalize one new rolling five-session label, so the weekly-horizon
model can retrain daily. A Monday 08:55 IST job separately captures current Asian, crypto,
commodity and currency movement. A persistent ledger settles Monday predictions after their fifth
close and powers the last-10 comparison.

## Limitations

- Yahoo intraday history is short and can be delayed or unavailable.
- The current stock-candidate universe creates survivorship bias.
- Yahoo's adjusted daily bars can be revised.
- A five-session target can cross into the next calendar week during holidays.
- Reported opens/closes omit real auction uncertainty, slippage, taxes, liquidity and borrow rules.
- NIFTY itself is not directly tradable.
- Overlapping labels, regime change and repeated experimentation can inflate apparent skill.
- Daily refitting does not guarantee improving accuracy or profitability.

## Promotion criteria

Require months of timestamped forward paper results, independent data reconciliation, a tradable
instrument and execution simulator, predefined acceptance thresholds, drift monitoring, risk
limits and a kill switch before considering capital.
