from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = ARTIFACT_DIR / "market_data.csv.gz"
MODEL_PATH = ARTIFACT_DIR / "model_bundle.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"
PREDICTION_HISTORY_PATH = ARTIFACT_DIR / "prediction_history.csv"
MORNING_FEATURES_PATH = ARTIFACT_DIR / "morning_features.csv.gz"

NIFTY = "^NSEI"

# A liquid, diversified subset is enough to discover three NIFTY proxies without
# turning one refresh into 50 independent Yahoo requests.
STOCK_CANDIDATES = {
    "RELIANCE.NS": "Reliance Industries",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "INFY.NS": "Infosys",
    "LT.NS": "Larsen & Toubro",
    "SBIN.NS": "State Bank of India",
    "AXISBANK.NS": "Axis Bank",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "ITC.NS": "ITC",
    "TCS.NS": "Tata Consultancy Services",
    "M&M.NS": "Mahindra & Mahindra",
}

MARKET_CONTEXT = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^KS11": "KOSPI",
    "^AXJO": "S&P/ASX 200",
    "^VIX": "US VIX",
    "^INDIAVIX": "India VIX",
    "^TNX": "US 10Y yield",
    "^FVX": "US 5Y yield",
    "ZN=F": "US 10Y Treasury-note futures",
    "ZB=F": "US Treasury-bond futures",
    "GC=F": "Gold",
    "CL=F": "WTI crude oil",
    "BZ=F": "Brent crude oil",
    "HG=F": "Copper",
    "INR=X": "USD/INR",
    "DX-Y.NYB": "US Dollar Index",
    "BTC-USD": "Bitcoin",
}

DISPLAY_NAMES = {NIFTY: "NIFTY 50", **STOCK_CANDIDATES, **MARKET_CONTEXT}
ALL_SYMBOLS = [NIFTY, *STOCK_CANDIDATES, *MARKET_CONTEXT]

MORNING_CONTEXT = {
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^KS11": "KOSPI",
    "^AXJO": "S&P/ASX 200",
    "GC=F": "Gold futures",
    "CL=F": "WTI crude futures",
    "INR=X": "USD/INR",
    "BTC-USD": "Bitcoin",
}


@dataclass(frozen=True)
class TrainingConfig:
    start_date: str = "2012-01-01"
    correlation_years: int = 3
    selected_stocks: int = 3
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    horizon_sessions: int = 5
    cv_splits: int = 5
    cv_gap: int = 5
    min_training_rows: int = 750
    round_trip_cost_bps: float = 10.0
    recency_half_life_sessions: int = 756
    threshold_lookback_sessions: int = 756
    morning_cutoff_hour_ist: int = 8
    morning_cutoff_minute_ist: int = 55
    morning_max_age_minutes: int = 180
    min_live_morning_symbols: int = 3
    gradient_boosting_estimators: int = 100
    extra_trees_estimators: int = 100
    random_state: int = 42


DEFAULT_CONFIG = TrainingConfig()
