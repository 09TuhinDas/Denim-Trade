
import numpy as np
import pandas as pd

def compute_swing_label_v2(df, profit_target=0.03, stop_loss=0.02, hold_days=5, use_atr=False):
    '''
    Enhanced swing labeler:
    - Checks next N bars to see if stop or target was hit first
    - Computes volatility
    - Calculates Kelly-based position sizing
    Returns:
        df with columns: label, stop_hit, target_hit, volatility, kelly_fraction
    '''
    labels = []
    volatilities = []
    kelly_fractions = []

    # Volatility: ATR or rolling std
    if use_atr and 'atr' in df.columns:
        df['volatility'] = df['atr']
    else:
        df['volatility'] = df['close'].rolling(window=14).std()

    for i in range(len(df) - hold_days):
        entry_price = df.iloc[i]['close']
        target_price = entry_price * (1 + profit_target)
        stop_price = entry_price * (1 - stop_loss)
        window = df.iloc[i+1:i+1+hold_days]

        hit_target = any(window['high'] >= target_price)
        hit_stop = any(window['low'] <= stop_price)

        # Determine which came first (simulate real decision path)
        label = 0
        for j, row in window.iterrows():
            if row['low'] <= stop_price:
                label = 0
                break
            elif row['high'] >= target_price:
                label = 1
                break

        # Volatility & Kelly
        vol = df.iloc[i]['volatility']
        risk = stop_loss
        reward = profit_target
        prob_win = 0.6 if label == 1 else 0.4  # Simple assumption; or use model prob
        edge = (prob_win * reward) - ((1 - prob_win) * risk)
        kelly = edge / (vol ** 2) if vol > 0 else 0

        labels.append(label)
        volatilities.append(vol)
        kelly_fractions.append(kelly)

    # Trim and return result
    result = df.iloc[:len(labels)].copy()
    result['label'] = labels
    result['volatility'] = volatilities
    result['kelly_fraction'] = kelly_fractions
    return result
