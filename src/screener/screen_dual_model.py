import os
import joblib
import yfinance as yf
import pandas as pd
import time
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import load_macro_cache
from src.ml.regime_detector import RegimeEngine
from src.utils.risk_management import garch_volatility, dynamic_kelly_size
from src.ml.quantum_short_model import QuantumShortSignal
from src.arbitrage_balancer import decide_trade
from src.utils.ticker_loader import load_nse_tickers
from src.config import FEATURE_COLS
from src.utils.path_manager import SCREEN_LOGS_DIR
from src.utils.status_manager import update_status

XGB_MODEL_PATH = "models/xgb_calibrated.pkl"
SHORT_MODEL_PATH = "models/short_model.pkl"

def load_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    short_model = QuantumShortSignal()
    short_model.model = joblib.load(SHORT_MODEL_PATH)
    return xgb_model, short_model

def screen_ticker_dual(ticker, xgb_model, short_model, threshold=0.6, margin=0.05, hold_days=5):
    try:
        df = yf.download(ticker, period="3mo", progress=False)

        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).strip().lower() for col in df.columns]

        if 'close' not in df.columns or len(df) < hold_days:
            return None

        df = add_indicators(df).dropna().copy()

        row = df.iloc[-1:].copy()
        row["volatility"] = df["close"].rolling(window=14).std().iloc[-1]
        row["kelly_fraction"] = 0.01

        # Pass correct input shape to each model
        long_X = row[FEATURE_COLS].to_numpy()
        short_features = FEATURE_COLS.copy() + ["volatility", "kelly_fraction"]
        short_X = row[short_features].to_numpy()


        


        long_conf = xgb_model.predict_proba(long_X)[0][1]
        short_conf = short_model.predict_proba(short_X)[0]

        label = decide_trade(long_conf, short_conf, threshold=threshold, margin=margin)
        if label == 0:
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
    tickers = load_nse_tickers()

    print(f"📊 Running dual-model screener... Threshold: {threshold}, Margin: {margin}\n")
    signals = []
    for ticker in tickers:
        result = screen_ticker_dual(ticker, xgb_model, short_model, threshold, margin)
        if result:
            print(f"✅ {result['ticker']} | {result['direction'].upper()} | Conf: {result['confidence']} | Size: {result['position_size']} | Regime: {result['regime']}")
            signals.append(result)

            time.sleep(1.2)

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