import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data
def fetch_stock_data_by_date(ticker, start, end):
    return yf.download(ticker, start=start, end=end, auto_adjust=True)


