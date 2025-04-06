
# src/utils/macro_features.py
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

CACHE_PATH = "data/macro_cache.csv"

def fetch_macro_data():
    now = datetime.today()
    start = now - timedelta(days=365)
    macro = {}

    symbols = {
        "vix": "^INDIAVIX",
        "usdinr": "USDINR=X",
        "crude": "BZ=F",
        "10y": "^IRX"
    }

    for name, symbol in symbols.items():
        try:
            df = yf.download(symbol, period="1y", progress=False)
            value = df['Close'].dropna().iloc[-1]
            macro[name] = round(float(value.iloc[0] if isinstance(value, pd.Series) else value), 2)
        except Exception as e:
            print(f"⚠️ Failed to fetch {name}: {e}")
            macro[name] = float("nan")

    # Save to cache
    df = pd.DataFrame([macro])
    df.index = [datetime.now()]
    df.to_csv(CACHE_PATH)

    return df


def load_macro_data():
    if os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    return fetch_macro_data()

def get_macro_snapshot(date=None):
    df = load_macro_data()
    
    if date:
        df = df[df.index <= pd.to_datetime(date)]

    if df is None or df.empty or df.iloc[-1].isnull().any():
        return {
            "vix": 18,
            "usdinr": 83,
            "crude": 80,
            "10y": 6.5
        }

    latest = df.iloc[-1]
    return {
        "vix": latest.get("vix", 18),
        "usdinr": latest.get("usdinr", 83),
        "crude": latest.get("crude", 80),
        "10y": latest.get("10y", 6.5)
    }
