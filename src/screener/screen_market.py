# ✅ Updated screen_market.py with stacked model support

import os
import argparse
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import get_macro_snapshot
from src.config import FEATURE_COLS, MODEL_PATH

XGB_CALIBRATED_PATH = "models/xgb_calibrated.pkl"
LGB_CALIBRATED_PATH = "models/lgb_calibrated.pkl"
STACKED_MODEL_PATH = "models/stacked_model.pkl"

def load_models(mode):
    if mode == "xgb-only":
        return joblib.load(XGB_CALIBRATED_PATH), None, None
    elif mode == "lgb-only":
        return None, joblib.load(LGB_CALIBRATED_PATH), None
    elif mode == "stacked":
        return joblib.load(XGB_CALIBRATED_PATH), joblib.load(LGB_CALIBRATED_PATH), joblib.load(STACKED_MODEL_PATH)
    else:  # ensemble
        return joblib.load(XGB_CALIBRATED_PATH), joblib.load(LGB_CALIBRATED_PATH), None

def screen(mode="ensemble"):
    xgb_model, lgb_model, stacked_model = load_models(mode)

    macro = get_macro_snapshot()
    vix = macro.get("vix", 18)
    dynamic_threshold = round(0.65 + 0.15 * (vix / 20), 2)
    print("\n\U0001f310 Macro Snapshot:")
    print(f"VIX       : {macro['vix']}")
    print(f"USDINR    : {macro['usdinr']}")
    print(f"CRUDE     : {macro['crude']}")
    print(f"10Y       : {macro['10y']}")
    print(f"\n\U0001f4c9 VIX-adjusted confidence threshold: {dynamic_threshold}\n")

    with open("data/top_tickers.txt") as f:
        tickers = [line.strip() for line in f]

    results = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2mo", progress=False)
            if df.empty or len(df) < 30:
                raise ValueError("Not enough data")

            df = add_indicators(df).dropna()
            df.columns = [col[0].strip() if isinstance(col, tuple) else str(col).strip() for col in df.columns]
            latest = df[FEATURE_COLS].iloc[-1:]
            latest.columns = latest.columns.str.strip()

            if mode == "xgb-only":
                confidence = xgb_model.predict_proba(latest)[0][1]
            elif mode == "lgb-only":
                confidence = lgb_model.predict_proba(latest)[0][1]
            elif mode == "stacked":
                xgb_conf = xgb_model.predict_proba(latest)[0][1]
                lgb_conf = lgb_model.predict_proba(latest)[0][1]
                meta_features = pd.DataFrame({"xgb_conf": [xgb_conf], "lgb_conf": [lgb_conf]})
                confidence = stacked_model.predict_proba(meta_features)[0][1]
            else:  # ensemble
                xgb_conf = xgb_model.predict_proba(latest)[0][1]
                lgb_conf = lgb_model.predict_proba(latest)[0][1]
                confidence = 0.6 * xgb_conf + 0.4 * lgb_conf

            print(f"\U0001f50d {ticker}: confidence = {confidence:.3f}")

            if confidence >= dynamic_threshold:
                results.append({
                    "ticker": ticker,
                    "confidence": round(confidence, 3),
                    "position_size": round(confidence * 100, 2)
                })

        except Exception as e:
            print(f"\u26a0\ufe0f Error with {ticker}: {e}")

    if not results:
        print("\u274c No valid predictions were made — possibly due to insufficient data or macro filtering.")
        return

    df_out = pd.DataFrame(results).sort_values(by="confidence", ascending=False)
    print("\n\U0001f4c8 Top Predictions:")
    print(df_out.head(10))

    now = datetime.now().strftime("%Y%m%d")
    output_path = f"logs/screener_{now}.csv"
    df_out.to_csv(output_path, index=False)
    print(f"\u2705 Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xgb-only", "lgb-only", "ensemble", "stacked"], default="ensemble")
    args = parser.parse_args()
    screen(mode=args.mode)
