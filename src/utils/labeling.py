import numpy as np
import pandas as pd

def compute_swing_label_with_short(
    df,
    profit_target=0.03,
    stop_loss=0.02,
    hold_days=5,
    use_atr=False
):
    """
    Time-aware swing labeler that fairly assigns LONG or SHORT
    based on which hits target/stop first within hold window.

    Returns:
        df with:
        - direction_label: 1 = LONG, -1 = SHORT, 0 = HOLD
        - volatility
        - kelly_fraction
    """

    direction_labels = []
    volatilities = []
    kelly_fractions = []

    # Calculate volatility
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

        window = df.iloc[i + 1: i + 1 + hold_days]

        long_hit_day = None
        short_hit_day = None

        for j, row in enumerate(window.itertuples(index=False)):
            if row.high >= long_tp:
                long_hit_day = j
                break
            elif row.low <= long_sl:
                break  # Long stop hit first → discard long

        for j, row in enumerate(window.itertuples(index=False)):
            if row.low <= short_tp:
                short_hit_day = j
                break
            elif row.high >= short_sl:
                break  # Short stop hit first → discard short

        if long_hit_day is not None and short_hit_day is not None:
            label = 1 if long_hit_day < short_hit_day else -1
        elif long_hit_day is not None:
            label = 1
        elif short_hit_day is not None:
            label = -1
        else:
            label = 0

        vol = df.iloc[i]['volatility']
        prob_win = 0.6 if label != 0 else 0.4
        edge = (prob_win * profit_target) - ((1 - prob_win) * stop_loss)
        kelly = edge / (vol ** 2) if vol > 0 else 0

        direction_labels.append(label)
        volatilities.append(vol)
        kelly_fractions.append(kelly)

    result = df.iloc[:len(direction_labels)].copy()
    result['direction_label'] = direction_labels
    result['volatility'] = volatilities
    result['kelly_fraction'] = kelly_fractions

    return result
