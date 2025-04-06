# src/config.py

from src.utils.ticker_loader import load_nse_tickers

# Features used in training & prediction
FEATURE_COLS = [
    "rsi", "macd", "ema20", "ema10", "ema_crossover",
    "adx", "bb_width", "obv",
    "stochrsi_k", "stochrsi_d",
    "cci", "roc"
]

# Thresholds
CONFIDENCE_BASE = 0.6  # used as base threshold before adjusting with VIX
CONFIDENCE_EXIT_THRESHOLD = 0.5
PROFIT_TARGET = 0.04
HOLD_DAYS = 5

# Macro filter control
MACRO_FILTER_ENABLED = True

# Compute dynamic confidence threshold
def get_confidence_threshold(vix):
    return min(0.9, CONFIDENCE_BASE + (vix / 40))

# Model & training
MODEL_PATH = "models/xgb_model.pkl"
TICKERS = load_nse_tickers()
BATCH_SIZE = 300
PROCESSED_LOG_PATH = "data/processed_tickers.json"
RAW_DATA_FOLDER = "data/training_batches"
