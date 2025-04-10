from src.utils.path_manager import *

import os
import yfinance as yf
import pandas as pd
import joblib
from tqdm import tqdm
from src.indicators.ta_signals import add_indicators
from src.config import FEATURE_COLS, HOLD_DAYS
from src.utils.labeling import compute_swing_label_v2
from sklearn.exceptions import NotFittedError

XGB_CALIBRATED_PATH = "models/xgb_calibrated.pkl"
LGB_CALIBRATED_PATH = "models/lgb_calibrated.pkl"
TICKERS_FILE = "data/top_tickers.txt"
OUTPUT_FILE = META_TRAINING

def generate_meta_data():
    assert os.path.exists(XGB_CALIBRATED_PATH), "Missing XGB model"
    assert os.path.exists(LGB_CALIBRATED_PATH), "Missing LGB model"
    assert os.path.exists(TICKERS_FILE), "Missing top_tickers.txt"

    xgb_model = joblib.load(XGB_CALIBRATED_PATH)
    lgb_model = joblib.load(LGB_CALIBRATED_PATH)

    with open(TICKERS_FILE) as f:
        tickers = [line.strip() for line in f.readlines()]

    meta_data = []

    for ticker in tqdm(tickers):
        try:
            df = yf.download(ticker, period="6mo", progress=False)
            if df.empty or len(df) < HOLD_DAYS + 10:
                continue

            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
            df = add_indicators(df).dropna()
            df = compute_swing_label_v2(df, hold_days=HOLD_DAYS)
            df["ticker"] = ticker
            df = df.dropna(subset=["label"])

            for i in range(len(df) - HOLD_DAYS):
                window = df.iloc[i:i + HOLD_DAYS + 1]
                if len(window) < HOLD_DAYS + 1:
                    continue

                X_row = df.iloc[i:i+1][FEATURE_COLS]
                xgb_conf = xgb_model.predict_proba(X_row)[0][1]
                lgb_conf = lgb_model.predict_proba(X_row)[0][1]
                avg_conf = (xgb_conf + lgb_conf) / 2
                conf_diff = abs(xgb_conf - lgb_conf)

                meta_data.append({
                    "ticker": ticker,
                    "xgb_conf": xgb_conf,
                    "lgb_conf": lgb_conf,
                    "avg_conf": avg_conf,
                    "conf_diff": conf_diff,
                    "volatility": df.iloc[i]["volatility"],
                    "kelly_fraction": df.iloc[i]["kelly_fraction"],
                    "label": df.iloc[i]["label"]
                })

        except NotFittedError as e:
            print(f"❌ Model not fitted: {e}")
        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    df_meta = pd.DataFrame(meta_data)
    df_meta.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved meta training data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_meta_data()
