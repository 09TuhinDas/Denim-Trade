import shutil
from src.utils.path_manager import OUTPUT_DIR, SCREEN_LOGS_DIR, BACKTEST_LOGS_DIR
from src.utils.status_manager import update_status
import os

def clear_directory(dir_path):
    for file in dir_path.glob("*.csv"):
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def reset_project():
    print("\n🔁 Resetting project state...")
    clear_directory(OUTPUT_DIR)
    clear_directory(SCREEN_LOGS_DIR)
    clear_directory(BACKTEST_LOGS_DIR)

    # Optionally clear top tickers
    top_ticker_file = OUTPUT_DIR / "top_tickers.csv"
    if top_ticker_file.exists():
        top_ticker_file.unlink()

    update_status("last_reset")
    print("✅ Logs, outputs, top tickers cleared.\n")
