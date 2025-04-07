# src/ml/calibrate_model.py

import os
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.config import RAW_DATA_FOLDER, FEATURE_COLS

TOP_TICKERS_PATH = "data/top_tickers.txt"
XGB_MODEL_PATH = "models/xgb_model.pkl"
LGB_MODEL_PATH = "models/lgb_model.pkl"
XGB_CALIBRATED_PATH = "models/xgb_calibrated.pkl"
LGB_CALIBRATED_PATH = "models/lgb_calibrated.pkl"

def load_batches(top_tickers):
    data = []
    for file in os.listdir(RAW_DATA_FOLDER):
        if not file.endswith(".pkl"):
            continue
        try:
            batch = joblib.load(os.path.join(RAW_DATA_FOLDER, file))
            X, y = batch["X"], batch["y"]
            tickers = batch.get("tickers", ["UNKNOWN"] * len(X))
            mask = pd.Series(tickers).isin(top_tickers)
            data.append((X[mask.values], y[mask.values]))
        except Exception as e:
            print(f"⚠️ Skipping {file}: {e}")
    if not data:
        raise ValueError("❌ No valid training data found.")
    X_all = pd.concat([x for x, _ in data])
    y_all = pd.concat([y for _, y in data])
    return X_all[FEATURE_COLS], y_all

def calibrate(model_path, X, y):
    print(f"📏 Calibrating {model_path}...")
    base_model = joblib.load(model_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)
    calib_model = CalibratedClassifierCV(estimator=base_model, cv=3, method='sigmoid')
    calib_model.fit(X_train, y_train)
    preds = calib_model.predict(X_val)
    print(classification_report(y_val, preds))
    return calib_model

def main():
    with open(TOP_TICKERS_PATH) as f:
        top_tickers = [line.strip() for line in f]

    X, y = load_batches(top_tickers)

    xgb_calibrated = calibrate(XGB_MODEL_PATH, X, y)
    joblib.dump(xgb_calibrated, XGB_CALIBRATED_PATH)
    print(f"✅ Saved calibrated XGB model to {XGB_CALIBRATED_PATH}")

    lgb_calibrated = calibrate(LGB_MODEL_PATH, X, y)
    joblib.dump(lgb_calibrated, LGB_CALIBRATED_PATH)
    print(f"✅ Saved calibrated LGB model to {LGB_CALIBRATED_PATH}")

if __name__ == "__main__":
    main()
