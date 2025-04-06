# src/utils/volume_breakout.py

import pandas as pd


def is_volume_breakout(df, vol_mult=2.0, lookback=20):
    """
    Returns True if current volume is significantly higher than previous average
    and today's close breaks out of previous high range.
    """
    if df is None or len(df) < lookback + 1:
        return False

    recent = df[-(lookback + 1):-1]  # exclude today
    avg_vol = recent['Volume'].mean()
    high_range = recent['High'].max()

    today = df.iloc[-1]

    if today['Volume'] > vol_mult * avg_vol and today['Close'] > high_range:
        return True
    return False


def breakout_score(df, vol_mult=2.0, lookback=20):
    """
    Returns a breakout score [0, 1] based on volume and price breakout intensity
    """
    if df is None or len(df) < lookback + 1:
        return 0.0

    recent = df[-(lookback + 1):-1]
    avg_vol = recent['Volume'].mean()
    high_range = recent['High'].max()

    today = df.iloc[-1]
    vol_score = min(today['Volume'] / (avg_vol + 1e-6), 3.0) / 3.0
    price_score = min((today['Close'] - high_range) / (high_range + 1e-6), 0.05) / 0.05
    if today['Close'] <= high_range:
        price_score = 0.0

    return round(0.6 * vol_score + 0.4 * price_score, 3)


# Example usage inside screener:
# if is_volume_breakout(df):
#     print("Breakout candidate!")
