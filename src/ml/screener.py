import joblib
import pandas as pd
from src.indicators.ta_signals import add_indicators
from utils.data_fetcher import fetch_stock_data

STOCK_LIST = [
    "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "WIPRO.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "HINDUNILVR.NS"
]

def load_model(path="models/xgb_model.pkl"):
    return joblib.load(path)

def screen_stocks(stock_list):
    model = load_model()
    results = []

    for ticker in stock_list:
        try:
            df = fetch_stock_data(ticker, period="6mo")
            df = add_indicators(df).dropna()

            latest_data = df[['rsi', 'macd']].iloc[-1:]
            prediction = model.predict(latest_data)[0]
            confidence = model.predict_proba(latest_data)[0][prediction]

            if prediction == 1 and confidence >= 0.6:  # Only show strong BUYs
                results.append((ticker, confidence))
        except Exception as e:
            print(f"❌ Skipping {ticker}: {e}")

    # Sort by confidence
    results = sorted(results, key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    top_signals = screen_stocks(STOCK_LIST)
    
    print("\n📈 High-Confidence BUY Signals:")
    if not top_signals:
        print("No strong BUY signals found today.")
    else:
        for ticker, conf in top_signals:
            print(f"{ticker} → BUY ✅ (Confidence: {conf:.2f})")
