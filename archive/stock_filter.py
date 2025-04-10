import yfinance as yf
import pandas as pd

def passes_filter(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d", progress=False)

        if df.empty or len(df) < 20:
            return False

        # Basic filters
        latest = df.iloc[-1]
        average_volume = df['Volume'].tail(20).mean()

        # Conditions
        if latest['Volume'] < average_volume: return False
        if latest['Close'] < 20: return False  # Skip penny/illiquid stocks

        # Simple price action filter
        ema20 = df['Close'].rolling(20).mean()
        if latest['Close'] < ema20.iloc[-1]: return False

        return True
    except:
        return False


def filter_tickers(ticker_list):
    filtered = [t for t in ticker_list if passes_filter(t)]
    return filtered


if __name__ == "__main__":
    demo = ["RELIANCE.NS", "XYZPENNY.NS", "TCS.NS"]
    final = filter_tickers(demo)
    print("✅ Filtered:", final)
