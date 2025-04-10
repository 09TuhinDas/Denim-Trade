from src.utils.path_manager import *
import pandas as pd

def load_nse_tickers(csv_path=NSE_TICKER_LIST):
    df = pd.read_csv(csv_path)
    
    # Assume symbol is under "SYMBOL" or similar
    symbol_col = [col for col in df.columns if "symbol" in col.lower()]
    if not symbol_col:
        raise ValueError("Couldn't find SYMBOL column in CSV.")

    symbols = df[symbol_col[0]].dropna().astype(str).str.upper().str.strip()
    
    # Append .NS to match yfinance format
    return [symbol + ".NS" for symbol in symbols if symbol.isalpha()]
