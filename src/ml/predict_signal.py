import joblib
import pandas as pd

from xgboost import XGBClassifier
from src.indicators.ta_signals import add_indicators
from utils.data_fetcher import fetch_stock_data

def load_model(path="models/xgb_model.pkl"):
    return joblib.load(path)

def predict_buy_sell(ticker="RELIANCE.NS"):
    model = load_model()

    df = fetch_stock_data(ticker, period="6mo")
    df = add_indicators(df).dropna()

    latest_data = df[['rsi', 'macd']].iloc[-1:]

    prediction = model.predict(latest_data)[0]
    confidence = model.predict_proba(latest_data)[0][prediction]

    signal = "BUY ✅" if prediction == 1 else "HOLD/SELL ❌"
    print(f"\n📊 {ticker}: {signal} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    predict_buy_sell("RELIANCE.NS")
