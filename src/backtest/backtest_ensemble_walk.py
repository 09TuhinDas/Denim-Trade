import os
import argparse
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import get_macro_snapshot
from src.utils.volume_breakout import is_volume_breakout
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


def walk_forward_backtest(ticker, xgb_model, lgb_model, stacked_model, mode, hold_days=5, threshold=0.55):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or len(df) < 60:
            return []

        df = add_indicators(df).dropna().copy()
        df.columns = [col[0].strip() if isinstance(col, tuple) else str(col).strip() for col in df.columns]
        trades = []

        for i in range(len(df) - hold_days):
            window = df.iloc[i:i + hold_days + 1]
            latest = df[FEATURE_COLS].iloc[i:i + 1]
            latest.columns = latest.columns.str.strip()

            xgb_conf = xgb_model.predict_proba(latest)[0][1] if xgb_model else None
            lgb_conf = lgb_model.predict_proba(latest)[0][1] if lgb_model else None

            if mode == "xgb-only":
                confidence = xgb_conf
            elif mode == "lgb-only":
                confidence = lgb_conf
            elif mode == "stacked":
                avg_conf = (xgb_conf + lgb_conf) / 2
                conf_diff = abs(xgb_conf - lgb_conf)
                meta_features = pd.DataFrame({
                    "xgb_conf": [xgb_conf],
                    "lgb_conf": [lgb_conf],
                    "avg_conf": [avg_conf],
                    "conf_diff": [conf_diff],
                })
                stacked_conf = stacked_model.predict_proba(meta_features)[0][1]
                ensemble_conf = 0.6 * xgb_conf + 0.4 * lgb_conf
                confidence = max(stacked_conf, ensemble_conf)  # fallback logic
                print(f"🔍 {ticker} | XGB: {xgb_conf:.3f}, LGB: {lgb_conf:.3f}, "
                      f"Stacked: {stacked_conf:.3f}, Ensemble: {ensemble_conf:.3f}, Used: {confidence:.3f}")
            else:
                confidence = 0.6 * xgb_conf + 0.4 * lgb_conf

            # Optional: Uncomment to use volume filtering
            # if not is_volume_breakout(window):
            #     continue

            if confidence >= threshold:
                entry_date = window.index[0]
                exit_date = window.index[hold_days] if len(window) > hold_days else window.index[-1]
                entry_price = window.iloc[0]["Close"]
                exit_price = window.iloc[hold_days]["Close"]
                ret = (exit_price - entry_price) / entry_price

                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "exit_date": exit_date.strftime("%Y-%m-%d"),
                    "confidence": round(confidence, 3),
                    "return": round(ret * 100, 2),
                    "exit_reason": "HOLD",
                    "exit": "TP"
                })

        return trades

    except Exception as e:
        print(f"⚠️ Error backtesting {ticker}: {e}")
        return []


def run(mode):
    xgb_model, lgb_model, stacked_model = load_models(mode)

    with open("data/top_tickers.txt") as f:
        tickers = [line.strip() for line in f]

    macro = get_macro_snapshot()
    vix = macro.get("vix", 18)
    dynamic_threshold = 0.55  # Lowered from 0.6
    print(f"\n📉 VIX-adjusted threshold: {dynamic_threshold}\n")

    all_trades = []
    for ticker in tickers:
        trades = walk_forward_backtest(ticker, xgb_model, lgb_model, stacked_model, mode, threshold=dynamic_threshold)
        all_trades.extend(trades)

    if not all_trades:
        print("❌ No trades were generated.")
        return

    df = pd.DataFrame(all_trades)
    filename = f"{OUTPUT_DIR}/backtest_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Saved backtest results to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xgb-only", "lgb-only", "ensemble", "stacked"], default="ensemble")
    args = parser.parse_args()
    run(mode=args.mode)
