# src/utils/path_manager.py

from pathlib import Path

# Base directory: two levels above this file
BASE_DIR = Path(__file__).resolve().parents[2]

# Core project folders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MACRO_DIR = DATA_DIR / "macro"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_BATCH_DIR = DATA_DIR / "training_batches"

LOGS_DIR = BASE_DIR / "logs"
SCREEN_LOGS_DIR = LOGS_DIR / "screener_runs"
BACKTEST_LOGS_DIR = LOGS_DIR / "backtest_runs"

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Common file paths
MACRO_CACHE = MACRO_DIR / "macro_cache.csv"
MARKET_REGIME = MACRO_DIR / "market_regime.csv"
NSE_TICKER_LIST = RAW_DIR / "nse_equity_list.csv"
META_TRAINING = PROCESSED_DIR / "meta_training.csv"
PROCESSED_TICKERS = PROCESSED_DIR / "processed_tickers.json"
SCREENED_SIGNALS = OUTPUT_DIR / "screened_signals.csv"
TOP_TICKERS = OUTPUT_DIR / "top_tickers.csv"
STACKED_PREDICTIONS = OUTPUT_DIR / "stacked_preds_20250407_055319.csv"
