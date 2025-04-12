import os
import argparse
import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.utils.macro_features import load_macro_cache
from src.utils.volume_breakout import is_volume_breakout
from src.utils.risk_management import garch_volatility, dynamic_kelly_size
from src.ml.regime_detector import RegimeEngine
from src.config import FEATURE_COLS

XGB_CALIBRATED_PATH = "models/xgb_calibrated.pkl"
LGB_CALIBRATED_PATH = "models/lgb_calibrated.pkl"
STACKED_MODEL_PATH = "models/stacked_model.pkl"
OUTPUT_DIR = "logs"
ALLOW_SHORT = True

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
    else:
        return (
            joblib.load(XGB_CALIBRATED_PATH),
            joblib.load(LGB_CALIBRATED_PATH),
            None,
        )

def apply_slippage(entry_price, order_size_value, adv_value):
    try:
        impact = 0.001 + 0.003 * ((order_size_value / (adv_value + 1e-6)) ** 1.5)
        return entry_price * (1 + impact)
    except:
        return entry_price

def walk_forward_backtest(ticker, xgb_model, lgb_model, stacked_model, mode, hold_days=5, threshold=0.55):
    try:
        df = yf.download(ticker, period="6mo", progress=False)
        df.columns = [str(col).strip().lower() for col in df.columns]
        if df.empty or len(df) < 60:
            return []

        df = add_indicators(df).dropna().copy()
        trades = []

        macro_df = load_macro_cache()
        regime = RegimeEngine()
        regime_state = regime.get_latest_regime(macro_df)
        regime_boost = regime.get_regime_config(regime_state)["max_size"]

        capital = 1_000_000

        for i in range(len(df) - hold_days):
            window = df.iloc[i:i + hold_days + 1]
            latest = df[FEATURE_COLS].iloc[i:i + 1]
            latest.columns = latest.columns.str.strip()

            # Predict label
            xgb_probs = xgb_model.predict_proba(latest)[0] if xgb_model else [0, 0, 0]
            lgb_probs = lgb_model.predict_proba(latest)[0] if lgb_model else [0, 0, 0]

            if mode == "xgb-only":
                final_label = xgb_probs.argmax()
                confidence = max(xgb_probs)
            elif mode == "lgb-only":
                final_label = lgb_probs.argmax()
                confidence = max(lgb_probs)
            elif mode == "stacked":
                avg_conf = (xgb_probs[1] + lgb_probs[1]) / 2
                conf_diff = abs(xgb_probs[1] - lgb_probs[1])
                meta = pd.DataFrame({
                    "xgb_conf": [xgb_probs[1]],
                    "lgb_conf": [lgb_probs[1]],
                    "avg_conf": [avg_conf],
                    "conf_diff": [conf_diff],
                })
                stacked_conf = stacked_model.predict_proba(meta)[0]
                final_label = stacked_conf.argmax()
                confidence = max(stacked_conf)
            else:
                ensemble_conf = 0.6 * xgb_probs[1] + 0.4 * lgb_probs[1]
                final_label = 1 if ensemble_conf >= threshold else 0
                confidence = ensemble_conf

            if final_label == 0 or (final_label == 2 and not ALLOW_SHORT):
                continue

            direction = "long" if final_label == 1 else "short"
            entry_date = window.index[0]
            exit_date = window.index[hold_days] if len(window) > hold_days else window.index[-1]
            entry_price_raw = window.iloc[0]["close"]
            adv = df["volume"].rolling(20).mean().iloc[i]
            returns = df["close"].pct_change().dropna().iloc[max(0, i - 60):i]
            sigma = garch_volatility(returns)
            mu = 0.03 if direction == "long" else -0.03
            position_size = dynamic_kelly_size(confidence, mu, sigma, regime_boost)
            order_value = capital * position_size
            entry_price = apply_slippage(entry_price_raw, order_value, adv)
            exit_price = window.iloc[hold_days]["close"]
            pnl = (exit_price - entry_price) / entry_price if direction == "long" else (entry_price - exit_price) / entry_price

            trades.append({
                "ticker": ticker,
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": exit_date.strftime("%Y-%m-%d"),
                "direction": direction,
                "confidence": round(confidence, 3),
                "return": round(pnl * 100, 2),
                "exit_reason": "HOLD",
                "exit": "TP",
                "volatility": round(sigma, 4) if sigma else None,
                "position_size": round(position_size, 4)
            })

        return trades

    except Exception as e:
        print(f"⚠️ Error backtesting {ticker}: {e}")
        return []

def run(mode):
    xgb_model, lgb_model, stacked_model = load_models(mode)

    import glob

    top_path = "data/top_tickers.txt"
    if not os.path.exists(top_path):
        print("⚠️ data/top_tickers.txt not found. Trying to recover from latest screener log...")
        screener_files = sorted(glob.glob("logs/screener_stacked_*.csv") + glob.glob("logs/screener_runs/screener_stacked_*.csv"))
        if screener_files:
            latest = screener_files[-1]
            df = pd.read_csv(latest)
            tickers = df["ticker"].drop_duplicates().tolist()
            os.makedirs("data", exist_ok=True)
            with open(top_path, "w") as f:
                for t in tickers:
                    f.write(t + "\n")
            print(f"✅ Recovered top tickers from {latest} → data/top_tickers.txt")
        else:
            raise FileNotFoundError("❌ Could not find screener fallback or top_tickers.txt")

    with open(top_path) as f:
        tickers = [line.strip() for line in f]
    dynamic_threshold = 0.55
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