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
CONFIDENCE_THRESHOLD = 0.7            # Screener buy threshold
CONFIDENCE_EXIT_THRESHOLD = 0.5       # Backtest: exit if confidence drops below this
PROFIT_TARGET = 0.04                  # 4% profit target
HOLD_DAYS = 5                         # Max hold duration (days)

# Model path
MODEL_PATH = "models/xgb_model.pkl"

TICKERS = load_nse_tickers()

BATCH_SIZE = 300
PROCESSED_LOG_PATH = "data/processed_tickers.json"
RAW_DATA_FOLDER = "data/training_batches"


