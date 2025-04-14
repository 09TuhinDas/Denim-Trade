# src/indicators/short_features.py

import numpy as np
import pandas as pd

def gamma_radiation_index(df):
    """
    Approximate synthetic Gamma Radiation Index
    (usually: put/call ratio * second derivative of VIX or implied vol)
    """
    df['gamma_index'] = (df['put_volume'] / (df['call_volume'] + 1e-6)) * df['vix'].diff().diff()
    df['gamma_index'].replace([np.inf, -np.inf], 0, inplace=True)
    return df

def liquidity_black_hole(df):
    """
    Liquidity depth imbalance approximation (L3 data not available, so synthetic)
    """
    df['blackhole'] = 1 - df['bid_volume'] / (df['ask_volume'] + 1e-6)
    df['blackhole'].replace([np.inf, -np.inf], 0, inplace=True)
    return df

def synthetic_panic_entanglement(df):
    """
    Synthetic put/call panic index - proxy sentiment entanglement signal
    """
    df['panic_score'] = np.tanh(df['news_buzz'] - df['buy_intent'])  # scaled sentiment proxy
    return df

def add_short_features(df):
    """
    Main call to inject short-side alpha signals into feature set.
    Assumes required columns: put_volume, call_volume, vix, bid_volume, ask_volume, news_buzz, buy_intent
    """
    df = gamma_radiation_index(df)
    df = liquidity_black_hole(df)
    df = synthetic_panic_entanglement(df)
    return df