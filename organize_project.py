from src.utils.path_manager import *
from pathlib import Path
import shutil

# Define your project base
BASE_DIR = Path(__file__).resolve().parent

# Define paths
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# New folder structure
MACRO_DIR = DATA_DIR / "macro"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_BATCHES_DIR = DATA_DIR / "training_batches"
SCREEN_LOGS_DIR = LOGS_DIR / "screener_runs"
BACKTEST_LOGS_DIR = LOGS_DIR / "backtest_runs"

# Create folders if not exist
for folder in [MACRO_DIR, RAW_DIR, PROCESSED_DIR, SCREEN_LOGS_DIR, BACKTEST_LOGS_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Safe move function
def safe_move(src, dst_folder):
    if src.exists():
        dst = dst_folder / src.name
        shutil.move(str(src), str(dst))
        print(f"[Moved] {src} -> {dst}")
    else:
        print(f"[Skipped] {src.name} not found")

# Move files to new locations
file_moves = [
    (DATA_DIR / "macro_cache.csv", MACRO_DIR),
    (DATA_DIR / "market_regime.csv", MACRO_DIR),
    (DATA_DIR / "nse_equity_list.csv", RAW_DIR),
    (DATA_DIR / "meta_training.csv", PROCESSED_DIR),
    (DATA_DIR / "processed_tickers.json", PROCESSED_DIR),
    (BASE_DIR / SCREENED_SIGNALS, OUTPUT_DIR),
    (MODELS_DIR / "stacked_preds_20250407_055319.csv", OUTPUT_DIR),
]

for src, dst in file_moves:
    safe_move(src, dst)

# Move screener and backtest logs
for screener_log in LOGS_DIR.glob("screener_*.csv"):
    safe_move(screener_log, SCREEN_LOGS_DIR)

for backtest_log in LOGS_DIR.glob("backtest_*.csv"):
    safe_move(backtest_log, BACKTEST_LOGS_DIR)
