import numpy as np
import pandas as pd

def compute_swing_label_with_short(df, profit_target=0.03, stop_loss=0.02, hold_days=5, use_atr=False):
    """
    Enhanced swing labeler with short selling support:
    - Long if price increases enough
    - Short if price drops enough
    - Hold if neither
    Returns:
        df with columns: label (0=hold, 1=long, 2=short), volatility, kelly_fraction
    """
    labels = []
    volatilities = []
    kelly_fractions = []

    # Volatility calculation (ATR or rolling std)
    if use_atr and 'atr' in df.columns:
        df['volatility'] = df['atr']
    else:
        df['volatility'] = df['close'].rolling(window=14).std()

    for i in range(len(df) - hold_days):
        entry_price = df.iloc[i]['close']
        long_tp = entry_price * (1 + profit_target)
        long_sl = entry_price * (1 - stop_loss)

        short_tp = entry_price * (1 - profit_target)
        short_sl = entry_price * (1 + stop_loss)

        window = df.iloc[i+1:i+1+hold_days]

        label = 0  # Default to 'hold'

        for _, row in window.iterrows():
            # Check short stop/target first — symmetry
            if row['high'] >= short_sl:
                break  # Short stop hit → skip short
            elif row['low'] <= short_tp:
                label = 2  # SHORT
                break
            elif row['low'] <= long_sl:
                break  # Long stop hit → skip long
            elif row['high'] >= long_tp:
                label = 1  # LONG
                break

        # Kelly calculation
        vol = df.iloc[i]['volatility']
        prob_win = 0.6 if label in [1, 2] else 0.4
        edge = ((prob_win * profit_target) - ((1 - prob_win) * stop_loss))
        kelly = edge / (vol ** 2) if vol > 0 else 0

        labels.append(label)
        volatilities.append(vol)
        kelly_fractions.append(kelly)

    result = df.iloc[:len(labels)].copy()
    result['label'] = labels
    result['volatility'] = volatilities
    result['kelly_fraction'] = kelly_fractions

    return result
