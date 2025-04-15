
import os
import glob
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import FEATURE_COLS, PROFIT_TARGET, HOLD_DAYS, RAW_DATA_FOLDER
from src.config import SHORT_MODEL_PATH
from src.utils.risk_management import garch_volatility, dynamic_kelly_size
from src.utils.macro_features import load_macro_cache
from src.ml.regime_detector import RegimeEngine
from src.arbitrage_balancer import decide_trade

XGB_MODEL_PATH = "models/xgb_calibrated.pkl"
LOG_PATH = "logs/backtest_runs/dual_trades_log.csv"

def evaluate_trade_window(prices, direction, profit_target=0.03, stop_loss=0.02):
    entry_price = prices.iloc[0]
    for i, price in enumerate(prices[1:]):
        if direction == 1:  # long
            if price >= entry_price * (1 + profit_target):
                return profit_target
            elif price <= entry_price * (1 - stop_loss):
                return -stop_loss
        elif direction == -1:  # short
            if price <= entry_price * (1 - profit_target):
                return profit_target
            elif price >= entry_price * (1 + stop_loss):
                return -stop_loss
    return prices.iloc[-1] / entry_price - 1 if direction == 1 else entry_price / prices.iloc[-1] - 1

def load_models():
    xgb_model = joblib.load(XGB_MODEL_PATH)
    short_model = joblib.load(SHORT_MODEL_PATH)
    return xgb_model, short_model

def run_backtest():
    batches = glob.glob(os.path.join(RAW_DATA_FOLDER, "*.pkl"))
    xgb_model, short_model = load_models()
    macro_df = load_macro_cache()
    regime_engine = RegimeEngine()
    regime_state = regime_engine.get_latest_regime(macro_df)
    regime_boost = regime_engine.get_regime_config(regime_state)["max_size"]

    logs = []

    try:
        for batch_path in batches:
            try:
                data = joblib.load(batch_path)
                X, y, tickers = data["X"], data["y"], data["tickers"]

                full_features = FEATURE_COLS + ["volatility", "kelly_fraction"]
                X = X[full_features].copy().reset_index(drop=True)
                y = y.reset_index(drop=True)
                tickers = pd.Series(tickers).reset_index(drop=True)

                for i in range(len(X)):
                    try:
                        row = X.iloc[i:i+1]
                        ticker = tickers.iloc[i]
                        label_true = y.iloc[i]

                        long_X = row[FEATURE_COLS]
                        short_X = row

                        long_conf = xgb_model.predict_proba(long_X)[0][1]
                        short_conf = short_model.predict_proba(short_X)[0][1]

                        decision = decide_trade(long_conf, short_conf, threshold=0.6, margin=0.05)
                        if decision == 0:
                            continue

                        # Simulate pseudo price data
                        entry_price = 100
                        price_series = pd.Series([entry_price * (1 + np.random.uniform(-0.03, 0.03)) for _ in range(HOLD_DAYS)])
                        pnl = evaluate_trade_window(price_series, direction=decision,
                                                    profit_target=PROFIT_TARGET, stop_loss=0.02)

                        returns = price_series.pct_change().dropna()
                        sigma = returns.std()
                        mu = 0.03 if decision == 1 else -0.03
                        position_size = dynamic_kelly_size(max(long_conf, short_conf), mu, sigma, regime_boost)

                        logs.append({
                            "ticker": ticker,
                            "decision": decision,
                            "confidence_long": long_conf,
                            "confidence_short": short_conf,
                            "pnl": pnl,
                            "position_size": position_size,
                            "regime": regime_state
                        })

                        if len(logs) % 5000 == 0:
                            print(f"💾 Saving intermediate log at {len(logs)} trades")
                            pd.DataFrame(logs).to_csv(LOG_PATH, index=False)

                    except Exception as e:
                        print(f"⚠️ Skipping row {i} in {batch_path}: {e}")
                        continue

            except Exception as e:
                print(f"⚠️ Skipping batch {batch_path}: {e}")
                continue

    except KeyboardInterrupt:
        print("❌ Backtest interrupted by user. Saving partial results...")

    # Final save
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame(logs)
    df.to_csv(LOG_PATH, index=False)

    # 📊 Summary Statistics
    print("📈 Total Trades:", len(df))
    print("✅ Win Rate:", (df['pnl'] > 0).mean())
    print("📊 Avg PnL:", df['pnl'].mean())
    print("📉 Max Drawdown:", df['pnl'].min())
    print(f"✅ Backtest complete. Trades logged to {LOG_PATH}")

    # Optional: Separate LONG and SHORT logs
    df[df["decision"] == 1].to_csv("logs/backtest_runs/long_trades.csv", index=False)
    df[df["decision"] == -1].to_csv("logs/backtest_runs/short_trades.csv", index=False)

if __name__ == "__main__":
    run_backtest()
