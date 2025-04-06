# src/backtest/backtest_ensemble_walk.py

import os
import argparse
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import get_macro_snapshot
from src.utils.volume_breakout import is_volume_breakout
from src.config import FEATURE_COLS, MODEL_PATH

LGB_MODEL_PATH = "models/lgb_model.pkl"
OUTPUT_DIR = "logs"

def load_model(mode):
    if mode == "xgb-only":
        return joblib.load(MODEL_PATH), None
    elif mode == "lgb-only":
        return None, joblib.load(LGB_MODEL_PATH)
    else:
        return joblib.load(MODEL_PATH), joblib.load(LGB_MODEL_PATH)

def walk_forward_backtest(ticker, xgb_model, lgb_model, mode, hold_days=5, threshold=0.6):
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

            if mode == "xgb-only":
                confidence = xgb_model.predict_proba(latest)[0][1]
            elif mode == "lgb-only":
                confidence = lgb_model.predict_proba(latest)[0][1]
            else:
                c1 = xgb_model.predict_proba(latest)[0][1]
                c2 = lgb_model.predict_proba(latest)[0][1]
                confidence = 0.6 * c1 + 0.4 * c2

            # Volume breakout filtering
           # if not is_volume_breakout(window):
              #  continue

            if confidence >= threshold:
                entry_date = window.index[0]
                exit_date = window.index[hold_days] if len(window) > hold_days else window.index[-1]
                entry_price = window.iloc[0]["Close"]
                exit_price = window.iloc[hold_days]["Close"]
                ret = (exit_price - entry_price) / entry_price

                exit_reason = "TP"

                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_date.strftime("%Y-%m-%d"),
                    "exit_date": exit_date.strftime("%Y-%m-%d"),
                    "confidence": round(confidence, 3),
                    "return": round(ret * 100, 2),
                    "exit_reason": "HOLD",
                    "exit": exit_reason 
                })

        return trades

    except Exception as e:
        print(f"⚠️ Error backtesting {ticker}: {e}")
        return []

def run(mode):
    xgb_model, lgb_model = load_model(mode)
    with open("data/top_tickers.txt") as f:
        tickers = [line.strip() for line in f]

    macro = get_macro_snapshot()
    vix = macro.get("vix", 18)
    # dynamic_threshold = round(0.65 + 0.15 * (vix / 20), 2)
    dynamic_threshold = 0.6  # for testing
    print(f"\n📉 VIX-adjusted threshold: {dynamic_threshold}\n")

    all_trades = []
    for ticker in tickers:
        trades = walk_forward_backtest(ticker, xgb_model, lgb_model, mode, threshold=dynamic_threshold)
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
    parser.add_argument("--mode", choices=["xgb-only", "lgb-only", "ensemble"], default="ensemble")
    args = parser.parse_args()
    run(mode=args.mode)
