# src/indicators/microstructure.py

import pandas as pd
import numpy as np

def compute_imbalance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute order book imbalance using top 2 bid/ask quantities.
    Assumes the DataFrame has the following columns:
    'bid1_qty', 'bid2_qty', 'ask1_qty', 'ask2_qty'
    """
    required_cols = ['bid1_qty', 'bid2_qty', 'ask1_qty', 'ask2_qty']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df['imbalance'] = (
        (df['bid1_qty'] + df['bid2_qty']) - (df['ask1_qty'] + df['ask2_qty'])
    ) / (
        (df['bid1_qty'] + df['bid2_qty'] + df['ask1_qty'] + df['ask2_qty'] + 1e-6)
    )
    return df
