
# src/screener/screen_market.py

import os
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH
from src.indicators.ta_signals import add_indicators

def load_top_tickers():
    with open("data/top_tickers.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

def screen():
    model = joblib.load(MODEL_PATH)
    tickers = load_top_tickers()
    results = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, period="1mo", progress=False)
            df = add_indicators(df).dropna()
            if df.empty: continue

            latest = df.iloc[-1:][FEATURE_COLS]
            latest.columns = [str(c).strip() for c in latest.columns]

            confidence = model.predict_proba(latest)[0][1]

            if confidence >= CONFIDENCE_THRESHOLD:
                size = min(1.0, round((confidence - 0.65) / 0.25, 2))  # Kelly-like position size
                results.append({"ticker": ticker, "confidence": round(confidence, 4), "position_size": size})
        except Exception as e:
            print(f"⚠️ Error with {ticker}: {e}")

    df_out = pd.DataFrame(results).sort_values(by="confidence", ascending=False)
    print("\n📊 High Confidence Signals:")
    print(df_out.head(10))

    date_str = datetime.now().strftime("%Y%m%d")
    os.makedirs("logs", exist_ok=True)
    df_out.to_csv(f"logs/screener_filtered_{date_str}.csv", index=False)
    print(f"✅ Results saved to logs/screener_filtered_{date_str}.csv")

if __name__ == "__main__":
    screen()
