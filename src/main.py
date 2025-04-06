import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_fetcher import fetch_stock_data
from indicators.ta_signals import add_indicators

# Fetch data for Reliance
data = fetch_stock_data("RELIANCE.NS", period="6mo")

# Add RSI and MACD
data = add_indicators(data)

# Show last 5 rows
print(data[['Close', 'rsi', 'macd']].tail())
