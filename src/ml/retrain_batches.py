from src.utils.path_manager import *

import os
import json
import time
import joblib
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.labeling import compute_swing_label_v2
from src.config import FEATURE_COLS, BATCH_SIZE, PROCESSED_LOG_PATH, RAW_DATA_FOLDER

def load_all_tickers():
    df = pd.read_csv(NSE_TICKER_LIST)
    return df['SYMBOL'].astype(str).str.upper().str.strip().apply(lambda x: f"{x}.NS").tolist()

def load_processed_tickers():
    if os.path.exists(PROCESSED_LOG_PATH):
        with open(PROCESSED_LOG_PATH, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_tickers(tickers):
    with open(PROCESSED_LOG_PATH, "w") as f:
        json.dump(list(tickers), f)

def fetch_and_process(ticker):
    try:
        df = yf.download(ticker, period="2y", progress=False)
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = add_indicators(df).dropna()

        # Apply enhanced labeling logic
        df = compute_swing_label_v2(df, profit_target=0.03, stop_loss=0.02, hold_days=5)
        df["ticker"] = ticker
        df = df.dropna(subset=["label"])

        if len(df) < 50:
            return None, None

        # Build features and target
        X = df[FEATURE_COLS + ["ticker"]].copy()
        X["volatility"] = df["volatility"]
        X["kelly_fraction"] = df["kelly_fraction"]
        y = df["label"]

        return X, y

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")
        return None, None

def save_batch_atomically(batch_data):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = f"{RAW_DATA_FOLDER}/batch_{ts}.pkl"
    temp_path = final_path + ".tmp"
    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

    joblib.dump(batch_data, temp_path)
    os.rename(temp_path, final_path)
    print(f"✅ Batch saved to {final_path}")

def main():
    all_tickers = load_all_tickers()
    processed = load_processed_tickers()
    tickers_to_process = [t for t in all_tickers if t not in processed]

    print(f"📦 Remaining tickers: {len(tickers_to_process)}")

    batch = []
    for ticker in tqdm(tickers_to_process):
        X, y = fetch_and_process(ticker)
        if X is None or y is None:
            continue

        batch.append({"X": X, "y": y})
        processed.add(ticker)

        if len(batch) >= BATCH_SIZE:
            combined_X = pd.concat([item["X"] for item in batch])
            combined_y = pd.concat([item["y"] for item in batch])
            combined_tickers = combined_X["ticker"].tolist()

            batch_data = {
                "X": combined_X.drop(columns=["ticker"]),
                "y": combined_y,
                "tickers": combined_tickers
            }
            save_batch_atomically(batch_data)
            batch = []

        save_processed_tickers(processed)
        time.sleep(1.2)  # avoid rate limits

    if batch:
        combined_X = pd.concat([item["X"] for item in batch])
        combined_y = pd.concat([item["y"] for item in batch])
        combined_tickers = combined_X["ticker"].tolist()

        batch_data = {
            "X": combined_X.drop(columns=["ticker"]),
            "y": combined_y,
            "tickers": combined_tickers
        }
        save_batch_atomically(batch_data)
        save_processed_tickers(processed)

if __name__ == "__main__":
    main()
