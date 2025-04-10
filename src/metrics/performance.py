# src/metrics/performance.py

import pandas as pd

def tag_market_periods(df: pd.DataFrame, vix_series: pd.Series) -> pd.DataFrame:
    """
    Tag each row with market condition: bull, bear, or high_vol
    Assumes df['returns'] exists.
    """
    df = df.copy()
    df['market_condition'] = 'neutral'

    df.loc[df['returns'] > 0.01, 'market_condition'] = 'bull'
    df.loc[df['returns'] < -0.01, 'market_condition'] = 'bear'
    df.loc[vix_series > 25, 'market_condition'] = 'high_vol'

    return df

def factor_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label each trade with factor exposure: momentum, size, liquidity
    Assumes df has the following:
    - 'momentum_score', 'market_cap', 'adv' (average daily volume)
    """
    df = df.copy()
    df['factor'] = 'unknown'

    df.loc[df['momentum_score'] > 0.7, 'factor'] = 'momentum'
    df.loc[df['market_cap'] < 500e6, 'factor'] = 'size'  # Small cap
    df.loc[df['adv'] > 5e7, 'factor'] = 'liquidity'

    return df

def attribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by market_condition and factor, calculate avg return and hit rate
    """
    summary = df.groupby(['market_condition', 'factor']).agg({
        'pnl': ['mean', 'count'],
        'success': 'mean'
    })
    summary.columns = ['avg_pnl', 'n_trades', 'hit_rate']
    return summary.reset_index()

def analyze(df: pd.DataFrame, vix_series: pd.Series) -> pd.DataFrame:
    """
    Run full attribution pipeline.
    """
    df = tag_market_periods(df, vix_series)
    df = factor_labeling(df)
    return attribution_summary(df)
