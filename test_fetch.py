# test_fetch.py
import yfinance as yf
import pandas as pd

def fetch_macro_series(symbol, name):
    print(f"Fetching {name} from {symbol}")
    if name == "vix":
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
else:
    df = yf.download(symbol, start="2023-01-01", progress=False)
    return df["Close"].to_frame(name)

vix = fetch_macro_series("^INDIAVIX", "vix")
print(vix.tail())
