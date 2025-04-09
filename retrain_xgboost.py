import os
import time
import random
import pandas as pd
import joblib
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.config import TICKERS, FEATURE_COLS, MODEL_PATH, HOLD_DAYS
from src.indicators.ta_signals import add_indicators
from src.utils.data_fetcher import fetch_stock_data

LOG_PATH = "logs/retrain_log.txt"


def log_event(message):
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {message}\n")


def train_and_save_model():
    X_all, y_all = [], []

    print("\n📅 Starting XGBoost retraining...")
    with open("data/top_tickers.txt") as f:
        tickers = [line.strip() for line in f if line.strip()]
    for ticker in tqdm(tickers):
        try:
            df = fetch_stock_data(ticker, period="1y")
            if df is None or df.empty:
                continue

            # ✅ Flatten columns in case of MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].strip().lower() for col in df.columns]
            else:
                df.columns = [str(col).strip().lower() for col in df.columns]


            df = add_indicators(df).dropna()

            df['future_close'] = df['close'].shift(-HOLD_DAYS)
            df['target'] = (df['future_close'] > df['close']).astype(int)
            df.dropna(inplace=True)

            if not all(col in df.columns for col in FEATURE_COLS + ['Close', 'future_close']):
                print(f"🧪 {ticker} columns: {df.columns.tolist()}")

                print(f"⚠️ Skipping {ticker}: Missing columns")
                continue

            X = df[FEATURE_COLS].copy()
            X.columns = X.columns.str.strip()
            y = df['target']

            X_all.append(X)
            y_all.append(y)

            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    if not X_all:
        print("❌ No data available for retraining.")
        log_event("Retraining skipped: No data available.")
        return

    X_final = pd.concat(X_all)
    y_final = pd.concat(y_all)

    print("\n🧠 Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n📊 Evaluation Report:\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")
    log_event("XGBoost model retrained and saved successfully.")


if __name__ == "__main__":
    train_and_save_model()
