# src/ml/train_lgb_model.py

import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder  # ✅ NEW
from src.config import RAW_DATA_FOLDER, FEATURE_COLS

TOP_TICKERS_PATH = "data/top_tickers.txt"
MODEL_PATH = "models/lgb_model.pkl"

def load_batches(top_tickers):
    data = []
    for file in os.listdir(RAW_DATA_FOLDER):
        if not file.endswith(".pkl"):
            continue
        try:
            batch = joblib.load(os.path.join(RAW_DATA_FOLDER, file))
            X, y = batch["X"], batch["y"]
            tickers = batch.get("tickers", None)

            # Only filter if tickers are present
            if tickers is not None:
                if len(tickers) != len(X):
                    print(f"⚠️ Skipping {file}: ticker length mismatch")
                    continue
                mask = pd.Series(tickers).isin(top_tickers)
                X, y = X[mask.values], y[mask.values]

            if X.empty or y.empty:
                print(f"⚠️ Skipping {file}: empty after filtering")
                continue

            data.append((X, y))

        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")

    if not data:
        raise ValueError("❌ No valid batches loaded.")

    X_all = pd.concat([x for x, _ in data])
    y_all = pd.concat([y for _, y in data])
    return X_all, y_all

def train():
    with open(TOP_TICKERS_PATH) as f:
        top_tickers = [line.strip() for line in f]

    X, y = load_batches(top_tickers)
    X = X[FEATURE_COLS]
    X.columns = X.columns.str.strip()

    # ✅ Encode labels to ensure 0,1,2 format
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42
    )

    model = lgb.LGBMClassifier(
        objective='multiclass',  # ✅ multiclass objective
        n_estimators=100,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        num_class=3  # ✅ optional, makes logic explicit
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print("\n📊 LGBM Model Evaluation:\n")
    print(classification_report(y_val, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, "models/label_encoder.pkl")  # ✅ Save encoder too
    print(f"\n✅ LightGBM model saved to {MODEL_PATH}")
    print(f"✅ Label encoder saved to models/label_encoder.pkl")

if __name__ == "__main__":
    train()
