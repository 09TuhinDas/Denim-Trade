
# src/ml/retrain_filtered_model.py

import os
import json
import joblib
import pandas as pd
from src.config import FEATURE_COLS, MODEL_PATH, RAW_DATA_FOLDER
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_top_tickers():
    with open("data/top_tickers.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def load_batches(tickers_filter):
    X_total, y_total = [], []
    for file in os.listdir(RAW_DATA_FOLDER):
        if file.endswith(".pkl") and not file.startswith("."):
            try:
                batch = joblib.load(os.path.join(RAW_DATA_FOLDER, file))
                X, y = batch["X"], batch["y"]
                X["mask"] = X["ticker"].isin(tickers_filter)
                X_filtered = X[X["mask"]].drop(columns=["ticker", "mask"], errors="ignore")
                y_filtered = y[X["mask"].values]  # align using boolean array
                X_total.append(X_filtered)
                y_total.append(y_filtered)
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")
    if not X_total:
        raise ValueError("❌ No valid training data found.")
    return pd.concat(X_total, ignore_index=True), pd.concat(y_total, ignore_index=True)

def train():
    top_tickers = load_top_tickers()
    X, y = load_batches(top_tickers)
    X = X[FEATURE_COLS]
    y = y.astype(int)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.07,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum()
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n📊 Filtered Model Evaluation:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
