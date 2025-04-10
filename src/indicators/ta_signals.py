import pandas as pd
import numpy as np
import warnings
from src.indicators.microstructure import compute_imbalance


warnings.filterwarnings("ignore")

# ============================
# ✅ Standard Technical Indicators
# ============================
def add_indicators(df):
    df = df.copy()

    # Ensure proper column formatting
    df.columns = [str(col).strip().lower() for col in df.columns]


    # === Momentum Indicators
    df["rsi"] = compute_rsi(df["close"], window=14)
    df["macd"] = compute_macd(df["close"])
    df["roc"] = df["close"].pct_change(periods=10)
    df["cci"] = compute_cci(df, window=20)

    # === Trend Indicators
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema_crossover"] = (df["ema10"] > df["ema20"]).astype(int)

    # === Volatility Indicators
    df["bb_width"] = compute_bollinger_band_width(df)
    df["atr"] = compute_atr(df)

    # === Volume Indicators
    df["obv"] = compute_obv(df)

    # === Stochastic RSI
    stoch_k, stoch_d = compute_stochrsi(df["close"])
    df["stochrsi_k"] = stoch_k
    df["stochrsi_d"] = stoch_d

    # === ADX
    df["adx"] = compute_adx(df)

    # ============================
    # 🔮 Nonlinear Features (AlphaPulse etc.)
    # ============================
    df["alpha_pulse"] = np.tanh(df["rsi"] * df["macd"] / (df["atr"] + 1e-6))

    try:
        from src.indicators.microstructure import compute_imbalance
        df = compute_imbalance(df)
    except ValueError:
        pass  # skip if microstructure columns aren't present


    return df


# ============================
# 📦 Helper Indicator Functions
# ============================
def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=window).mean()
    avg_loss = down.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_bollinger_band_width(df, window=20):
    sma = df["close"].rolling(window).mean()
    std = df["close"].rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma

def compute_cci(df, window=20):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window=window).mean()
    md = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (tp - ma) / (0.015 * md + 1e-6)

def compute_atr(df, window=14):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

def compute_obv(df):
    direction = np.sign(df["close"].diff()).fillna(0)
    volume = df["volume"]
    obv = (volume * direction).cumsum()
    return obv

def compute_stochrsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(window).min()
    max_rsi = rsi.rolling(window).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi + 1e-6)
    k = stoch_rsi.rolling(3).mean()
    d = k.rolling(3).mean()
    return k, d

def compute_adx(df, window=14):
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    tr = compute_atr(df, window)
    plus_di = 100 * (plus_dm.rolling(window).mean() / (tr + 1e-6))
    minus_di = 100 * (minus_dm.rolling(window).mean() / (tr + 1e-6))
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)
    return dx.rolling(window).mean()

# ============================
# 🔍 Test Mode
# ============================
if __name__ == "__main__":
    print("[test] Loading test ticker...")
    df = pd.read_csv("data/sample_ticker.csv")
    df = add_indicators(df)
    print(df[["close", "rsi", "macd", "atr", "alpha_pulse"]].tail())
