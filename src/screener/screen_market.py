# src/screener/screen_market.py

import os
import argparse
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import get_macro_snapshot
from src.utils.volume_breakout import is_volume_breakout
from src.config import FEATURE_COLS, MODEL_PATH

LGB_MODEL_PATH = "models/lgb_model.pkl"

def load_model(mode):
    if mode == "xgb-only":
        return joblib.load(MODEL_PATH), None
    elif mode == "lgb-only":
        return None, joblib.load(LGB_MODEL_PATH)
    else:
        return joblib.load(MODEL_PATH), joblib.load(LGB_MODEL_PATH)

def screen(mode="ensemble"):
    xgb_model, lgb_model = load_model(mode)

    macro = get_macro_snapshot()
    vix = macro.get("vix", 18)
   # dynamic_threshold = round(0.65 + 0.15 * (vix / 20), 2)
    dynamic_threshold = 0.6  # Temporarily lowered for testing


    print("\n🌐 Macro Snapshot:")
    print(f"VIX       : {macro['vix']}")
    print(f"USDINR    : {macro['usdinr']}")
    print(f"CRUDE     : {macro['crude']}")
    print(f"10Y       : {macro['10y']}")
    print(f"\n📉 VIX-adjusted confidence threshold: {dynamic_threshold}\n")

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

            # ✅ Volume breakout confirmation
           # if not is_volume_breakout(df):
               # continue

            latest = df[FEATURE_COLS].iloc[-1:]
            latest.columns = latest.columns.str.strip()

            if mode == "xgb-only":
                confidence = xgb_model.predict_proba(latest)[0][1]
            elif mode == "lgb-only":
                confidence = lgb_model.predict_proba(latest)[0][1]
            else:
                xgb_conf = xgb_model.predict_proba(latest)[0][1]
                lgb_conf = lgb_model.predict_proba(latest)[0][1]
                confidence = 0.6 * xgb_conf + 0.4 * lgb_conf
                print(f"🔍 {ticker}: confidence = {confidence:.3f}")


            results.append({
                "ticker": ticker,
                "confidence": round(confidence, 3),
                "position_size": round(confidence * 100, 2)
            })

        except Exception as e:
            print(f"⚠️ Error with {ticker}: {e}")

    if not results:
        print("❌ No valid predictions were made — possibly due to insufficient data or macro filtering.")
        return

    df_out = pd.DataFrame(results).sort_values(by="confidence", ascending=False)
    print("\n📊 Top Predictions:")
    print(df_out.head(10))

    now = datetime.now().strftime("%Y%m%d")
    output_path = f"logs/screener_{now}.csv"
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved predictions to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xgb-only", "lgb-only", "ensemble"], default="ensemble")
    args = parser.parse_args()
    screen(mode=args.mode)
