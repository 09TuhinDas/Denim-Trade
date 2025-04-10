import os
import re
from pathlib import Path
import shutil

# === REPLACEMENTS BASED ON PATH_MANAGER ===
REPLACEMENTS = {
    r"['\"]data/macro_cache.csv['\"]": "MACRO_CACHE",
    r"['\"]data/market_regime.csv['\"]": "MARKET_REGIME",
    r"['\"]data/nse_equity_list.csv['\"]": "NSE_TICKER_LIST",
    r"['\"]data/meta_training.csv['\"]": "META_TRAINING",
    r"['\"]data/processed_tickers.json['\"]": "PROCESSED_TICKERS",
    r"['\"]models/stacked_preds_20250407_055319.csv['\"]": "STACKED_PREDICTIONS",
    r"['\"]market_screener_signals.csv['\"]": "SCREENED_SIGNALS",
    r"['\"]top_tickers.txt['\"]": "TOP_TICKERS"
}

IMPORT_LINE = "from src.utils.path_manager import *\n"
BACKUP_DIR = Path("backup_before_refactor")
BACKUP_DIR.mkdir(exist_ok=True)

# === FUNCTION TO PROCESS FILE ===
def process_file(py_file):
    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        updated_text = text
        file_changed = False

        for pattern, replacement in REPLACEMENTS.items():
            if re.search(pattern, updated_text):
                updated_text = re.sub(pattern, replacement, updated_text)
                file_changed = True

        if file_changed:
            if "path_manager" not in updated_text:
                updated_text = IMPORT_LINE + updated_text

            backup_path = BACKUP_DIR / py_file.name
            shutil.copy2(py_file, backup_path)

            py_file.write_text(updated_text, encoding="utf-8")
            print(f"[UPDATED] {py_file}")
            return str(py_file)
    except Exception as e:
        print(f"[ERROR] {py_file} -> {e}")
    return None

# === SCAN ALL .py FILES ===
changed_files = []
for py_file in Path.cwd().rglob("*.py"):
    if any(excl in str(py_file) for excl in ["venv", "archive", "backup_before_refactor", "__pycache__"]):
        continue
    result = process_file(py_file)
    if result:
        changed_files.append(result)

# === SUMMARY ===
print("\n✅ Auto Replacement Complete!")
print("Files Updated:")
for f in changed_files:
    print(" -", f)
