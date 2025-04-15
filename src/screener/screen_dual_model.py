
import os
import joblib
import yfinance as yf
import pandas as pd
import time
import random
from datetime import datetime

from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import load_macro_cache
from src.ml.regime_detector import RegimeEngine
from src.utils.risk_management import garch_volatility, dynamic_kelly_size
from src.arbitrage_balancer import decide_trade
from src.utils.ticker_loader import load_nse_tickers
from src.config import FEATURE_COLS
from src.utils.path_manager import SCREEN_LOGS_DIR
from src.utils.status_manager import update_status

global_long_confidences = []
global_short_confidences = []

XGB_MODEL_PATH = "models/xgb_long_trained.pkl"
SHORT_MODEL_PATH = "models/short_model.pkl"

CHUNK_INDEX = 0
CHUNK_SIZE = 300

def load_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    short_model = joblib.load(SHORT_MODEL_PATH)
    return xgb_model, short_model

def safe_download(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, period="3mo", progress=False)
            if not df.empty:
                return df
        except Exception as e:
            print(f"⚠️ {ticker} download failed (attempt {attempt+1}): {e}")
        time.sleep(random.uniform(2, 4))
    return None

def screen_ticker_dual(ticker, xgb_model, short_model, threshold=0.6, margin=0.05, hold_days=5):
    try:
        df = safe_download(ticker)
        if df is None or df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).strip().lower() for col in df.columns]

        if 'close' not in df.columns or len(df) < hold_days:
            return None

        df = add_indicators(df).dropna().copy()
        if len(df) < 1:
            return None

        row = df.iloc[-1:].copy()
        row["volatility"] = df["close"].rolling(window=14).std().iloc[-1]
        row["kelly_fraction"] = 0.01

        long_X = row[FEATURE_COLS].to_numpy()
        print("🔍 long_X shape:", long_X.shape)

        short_features = FEATURE_COLS.copy() + ["volatility", "kelly_fraction"]
        short_X = row[short_features].to_numpy()
        print("🎯 SHORT model expects:", short_model.n_features_in_, "features")
        print("🎯 short_X shape:", short_X.shape)

        long_conf = xgb_model.predict_proba(long_X)[0][1]
        short_conf = short_model.predict_proba(short_X)[0][1]

        if short_conf == 1.0:
            print(f"⚠️ {ticker} — SHORT_CONF is 1.0 (possible overfit or miscalibration)")

        global_long_confidences.append(long_conf)
        global_short_confidences.append(short_conf)

        label = decide_trade(long_conf, short_conf, threshold=threshold, margin=margin)
        if label == 0:
            print(f"❌ {ticker} — No confident signal | LONG: {round(long_conf, 3)} | SHORT: {round(short_conf, 3)}")
            return None

        print(f"{ticker} | LONG_CONF: {round(long_conf, 4)} | SHORT_CONF: {round(short_conf, 4)} | Decision: {label}")

        direction = "long" if label == 1 else "short"

        regime = RegimeEngine()
        macro_df = load_macro_cache()
        regime_state = regime.get_latest_regime(macro_df)
        regime_boost = regime.get_regime_config(regime_state)["max_size"]

        returns = df["close"].pct_change().dropna().iloc[-60:]
        sigma = garch_volatility(returns)
        mu = 0.03 if direction == "long" else -0.03
        position_size = dynamic_kelly_size(max(long_conf, short_conf), mu, sigma, regime_boost)

        return {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "direction": direction,
            "confidence": round(max(long_conf, short_conf), 4),
            "volatility": round(sigma, 4),
            "position_size": round(position_size, 4),
            "regime": regime_state,
        }

    except Exception as e:
        print(f"⚠️ Error for {ticker}: {e}")
        return None

def run(threshold=0.6, margin=0.05, save_path_override=None):
    xgb_model, short_model = load_models()
    all_tickers = load_nse_tickers()
    tickers = all_tickers[CHUNK_INDEX * CHUNK_SIZE : (CHUNK_INDEX + 1) * CHUNK_SIZE]

    print(f"📊 Running dual-model screener... Threshold: {threshold}, Margin: {margin}\n")
    signals = []

    for ticker in tickers:
        print(f"🔍 Checking: {ticker}")
        result = screen_ticker_dual(ticker, xgb_model, short_model, threshold, margin)
        if result:
            print(f"✅ {result['ticker']} | {result['direction'].upper()} | Conf: {result['confidence']} | Size: {result['position_size']} | Regime: {result['regime']}")
            signals.append(result)
        time.sleep(random.uniform(1.5, 2.8))  # jitter to avoid YF rate limits

    if global_long_confidences and global_short_confidences:
        import numpy as np
        print("\n📊 Confidence Summary Across All Tickers:")
        print(f"LONG — Max: {max(global_long_confidences):.4f}, Avg: {np.mean(global_long_confidences):.4f}, Min: {min(global_long_confidences):.4f}")
        print(f"SHORT — Max: {max(global_short_confidences):.4f}, Avg: {np.mean(global_short_confidences):.4f}, Min: {min(global_short_confidences):.4f}")

    if not signals:
        print("❌ No valid signals found.")
        return

    df = pd.DataFrame(signals)
    if save_path_override:
        filename = save_path_override
    else:
        filename = SCREEN_LOGS_DIR / f"screener_dual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    df.to_csv(filename, index=False)
    print(f"📈 Screener results saved to {filename}")
    update_status("last_screen")

if __name__ == "__main__":
    run()
