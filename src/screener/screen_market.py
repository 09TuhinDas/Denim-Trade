import os
import argparse
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import load_macro_cache
from src.ml.regime_detector import RegimeEngine
from src.utils.risk_management import garch_volatility, dynamic_kelly_size
from src.config import FEATURE_COLS

XGB_CALIBRATED_PATH = "models/xgb_calibrated.pkl"
LGB_CALIBRATED_PATH = "models/lgb_calibrated.pkl"
STACKED_MODEL_PATH = "models/stacked_model.pkl"
OUTPUT_DIR = "logs"

def load_models(mode):
    if mode == "xgb-only":
        return joblib.load(XGB_CALIBRATED_PATH), None, None
    elif mode == "lgb-only":
        return None, joblib.load(LGB_CALIBRATED_PATH), None
    elif mode == "stacked":
        return (
            joblib.load(XGB_CALIBRATED_PATH),
            joblib.load(LGB_CALIBRATED_PATH),
            joblib.load(STACKED_MODEL_PATH),
        )
    else:  # ensemble
        return (
            joblib.load(XGB_CALIBRATED_PATH),
            joblib.load(LGB_CALIBRATED_PATH),
            None,
        )

def screen_ticker(ticker, xgb_model, lgb_model, stacked_model, mode, threshold=0.55, hold_days=5):
    try:
        df = yf.download(ticker, period="3mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [str(col).strip().lower() for col in df.columns]

        df = add_indicators(df).dropna().copy()
        if df.empty or len(df) < hold_days:
            return None

        row = df.iloc[-1:]
        latest = row[FEATURE_COLS]

        xgb_conf = xgb_model.predict_proba(latest)[0][1] if xgb_model else None
        lgb_conf = lgb_model.predict_proba(latest)[0][1] if lgb_model else None

        if mode == "xgb-only":
            confidence = xgb_conf
        elif mode == "lgb-only":
            confidence = lgb_conf
        elif mode == "stacked":
            avg_conf = (xgb_conf + lgb_conf) / 2
            conf_diff = abs(xgb_conf - lgb_conf)
            meta = pd.DataFrame({
                "xgb_conf": [xgb_conf],
                "lgb_conf": [lgb_conf],
                "avg_conf": [avg_conf],
                "conf_diff": [conf_diff],
            })
            stacked_conf = stacked_model.predict_proba(meta)[0][1]
            ensemble_conf = 0.6 * xgb_conf + 0.4 * lgb_conf
            confidence = max(stacked_conf, ensemble_conf)
        else:
            confidence = 0.6 * xgb_conf + 0.4 * lgb_conf

        if confidence < threshold:
            return None

        # 🧠 Regime detection + adaptive sizing
        regime = RegimeEngine()
        macro_df = load_macro_cache()
        regime_state = regime.get_latest_regime(macro_df)
        regime_boost = regime.get_regime_config(regime_state)["max_size"]

        returns = df["close"].pct_change().dropna().iloc[-60:]
        sigma = garch_volatility(returns)
        mu = 0.03
        position_size = dynamic_kelly_size(confidence, mu, sigma, regime_boost)

        return {
            "ticker": ticker,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "confidence": round(confidence, 4),
            "volatility": round(sigma, 4),
            "position_size": round(position_size, 4),
            "regime": regime_state,
        }

    except Exception as e:
        print(f"⚠️ Screener error for {ticker}: {e}")
        return None

def run(mode):
    xgb_model, lgb_model, stacked_model = load_models(mode)

    with open("data/top_tickers.txt") as f:
        tickers = [line.strip() for line in f]

    macro_df = load_macro_cache()
    latest_vix = macro_df["vix"].iloc[-1]
    dynamic_threshold = round(0.65 + 0.15 * (latest_vix / 20), 2)

    print(f"\n📉 VIX-adjusted threshold: {dynamic_threshold} (VIX = {latest_vix:.2f})\n")

    signals = []
    for ticker in tickers:
        result = screen_ticker(ticker, xgb_model, lgb_model, stacked_model, mode, threshold=dynamic_threshold)
        if result:
            print(f"✅ {result['ticker']} | Conf: {result['confidence']} | Size: {result['position_size']} | Regime: {result['regime']}")
            signals.append(result)

    if not signals:
        print("❌ No signals met the threshold.")
        return

    df = pd.DataFrame(signals)
    filename = f"{OUTPUT_DIR}/screener_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"📈 Screener results saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xgb-only", "lgb-only", "ensemble", "stacked"], default="stacked")
    args = parser.parse_args()
    run(mode=args.mode)
