
from src.utils.ticker_loader import load_nse_tickers

# ✅ Features used in training & prediction
FEATURE_COLS = [
    "rsi", "macd", "ema20", "ema10", "ema_crossover",
    "adx", "bb_width", "obv",
    "stochrsi_k", "stochrsi_d",
    "cci", "roc"
]

# ✅ Thresholds
CONFIDENCE_BASE = 0.6  # Base threshold before adjusting with VIX
CONFIDENCE_EXIT_THRESHOLD = 0.5
PROFIT_TARGET = 0.04
HOLD_DAYS = 5

# ✅ Macro filter toggle
MACRO_FILTER_ENABLED = True

# ✅ Compute dynamic threshold from VIX
def get_confidence_threshold(vix):
    return min(0.9, CONFIDENCE_BASE + (vix / 40))

# ✅ Model paths and data structure
MODEL_PATH = "models/xgb_model.pkl"
STACKED_MODEL_PATH = "models/stacked_model.pkl"
TICKERS = load_nse_tickers()
BATCH_SIZE = 300
PROCESSED_LOG_PATH = "data/processed_tickers.json"
RAW_DATA_FOLDER = "data/training_batches"

# ✅ Intelligence Layer Toggles
USE_KELLY_SIZING = True
CIRCUIT_BREAKER_ENABLED = True
MAX_CONSECUTIVE_LOSSES = 5  # Trigger circuit breaker after N bad trades
CONFIDENCE_THRESHOLD = 0.7

