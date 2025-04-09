import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

MACRO_CACHE_PATH = "data/macro_cache.csv"

# ----------------------
# Generate High-Quality Macro Dataset for Swing Trading
# ----------------------
def generate_macro_cache(path=MACRO_CACHE_PATH):
    print("📊 Fetching macro data for regime modeling...")

    try:
        # Fetch NIFTYBEES as NIFTY proxy
        nifty = yf.download("NIFTYBEES.NS", start="2023-01-01", interval="1d", progress=False)["Close"]
        print("✅ NIFTY fetched")

        # Fetch USDINR
        usdinr = yf.download("INR=X", start="2023-01-01", interval="1d", progress=False)["Close"]
        print("✅ USDINR fetched")

    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        return

    # Compute NIFTY return
    nifty_return = nifty.pct_change()
    vix_proxy = nifty_return.rolling(5).std() * np.sqrt(252)

    # Simulate FII flow
    np.random.seed(42)
    fii_flow = pd.Series(np.random.normal(0, 500, len(nifty)), index=nifty.index)

    # Combine all series with consistent column names
    df = pd.concat([vix_proxy, nifty, usdinr, fii_flow, nifty_return], axis=1)
    df.columns = ["vix", "nifty", "usdinr", "fii_flow", "nifty_return"]

    df.dropna(inplace=True)
    df = df.reset_index(drop=True)

    # Save to cache
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Saved macro regime features to {path}")
    print(df.head())

# ----------------------
# Load Dataset
# ----------------------
def load_macro_cache(path=MACRO_CACHE_PATH):
    return pd.read_csv(path)

# ----------------------
# CLI Entry Point
# ----------------------
if __name__ == "__main__":
    generate_macro_cache()
