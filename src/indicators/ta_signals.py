import pandas as pd
import ta


def add_indicators(df):
    df = df.copy()

    # Flatten all series to 1D
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()

    # MACD
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd_diff()

    # EMA (20)
    df['ema20'] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df['ema10'] = ta.trend.EMAIndicator(close, window=10).ema_indicator()
    df['ema_crossover'] = (df['ema10'] > df['ema20']).astype(int)

    # ADX (14)
    df['adx'] = ta.trend.ADXIndicator(high, low, close).adx()

    # Bollinger Band Width
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df['bb_width'] = bb.bollinger_wband()

    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    # Stochastic RSI
    stoch_rsi = ta.momentum.StochRSIIndicator(close)
    df['stochrsi_k'] = stoch_rsi.stochrsi_k()
    df['stochrsi_d'] = stoch_rsi.stochrsi_d()

    # CCI
    df['cci'] = ta.trend.CCIIndicator(high, low, close).cci()

    # ROC
    df['roc'] = ta.momentum.ROCIndicator(close).roc()

    return df
