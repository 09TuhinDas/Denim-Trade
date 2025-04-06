import os
import json
import time
import joblib
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.config import FEATURE_COLS, BATCH_SIZE, PROCESSED_LOG_PATH, RAW_DATA_FOLDER

def get_swing_label(prices, profit=0.03, stop=0.02, window=5):
    for i in range(1, window + 1):
        change = (prices[i] - prices[0]) / prices[0]
        if change >= profit:
            return 1
        elif change <= -stop:
            return 0
    return None

def load_all_tickers():
    df = pd.read_csv("data/nse_equity_list.csv")
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
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = add_indicators(df).dropna()

        X, y = [], []
        for i in range(len(df) - 5):
            future = df['Close'].iloc[i:i+6].values
            label = get_swing_label(future)
            if label is None:
                continue
            row = df[FEATURE_COLS].iloc[i].copy()
            row["ticker"] = ticker  # ✅ add the ticker name
            X.append(row)
            y.append(label)

        X = pd.DataFrame(X)
        y = pd.Series(y)

        if len(X) < 50:
            return None, None

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
            batch_data = {
                "X": pd.concat([item["X"] for item in batch]),
                "y": pd.concat([item["y"] for item in batch])
            }
            save_batch_atomically(batch_data)
            batch = []

        save_processed_tickers(processed)
        time.sleep(1.2)  # avoid rate limits

    if batch:
        batch_data = {
            "X": pd.concat([item["X"] for item in batch]),
            "y": pd.concat([item["y"] for item in batch])
        }
        save_batch_atomically(batch_data)
        save_processed_tickers(processed)

if __name__ == "__main__":
    main()
