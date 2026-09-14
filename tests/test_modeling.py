from __future__ import annotations

import numpy as np
import pandas as pd

from nifty_intraday.config import TrainingConfig
from nifty_intraday.modeling import signal_from_prediction, train_asset, weighted_predict


def test_walk_forward_training_and_prediction() -> None:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2019-01-01", periods=820)
    X = pd.DataFrame(
        {
            "lagged_market": rng.normal(0, 0.01, len(index)),
            "risk": rng.normal(0, 1, len(index)),
        },
        index=index,
    )
    X["late_history"] = np.nan
    X.loc[index[600:], "late_history"] = rng.normal(0, 1, len(index) - 600)
    y = pd.Series(0.12 * X["lagged_market"] + rng.normal(0, 0.004, len(index)), index=index)
    config = TrainingConfig(
        min_training_rows=500,
        cv_splits=3,
        gradient_boosting_estimators=20,
        extra_trees_estimators=20,
    )
    result = train_asset(X, y, config)
    prediction = weighted_predict(result, X.iloc[[-1]])

    assert np.isfinite(prediction)
    assert result["test_start"] > result["training_start"]
    assert result["test_metrics"]["rows"] > 0
    assert 0.14 <= result["split_rows"]["test"] / len(X) <= 0.16
    assert (pd.DatetimeIndex(result["holdout"]["Date"]).dayofweek == 0).all()
    assert result["split_rows"]["purged_at_each_boundary"] == 5
    assert abs(sum(result["weights"].values()) - 1) < 1e-9


def test_signal_has_abstention_zone() -> None:
    assert signal_from_prediction(0.02, 0.01) == "BUY"
    assert signal_from_prediction(-0.02, 0.01) == "SELL"
    assert signal_from_prediction(0.005, 0.01) == "HOLD"
    assert signal_from_prediction(0.02, 0.01, enabled=False) == "HOLD"
