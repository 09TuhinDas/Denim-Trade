
import os
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.config import (
    FEATURE_COLS, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET,
    CONFIDENCE_THRESHOLD, CIRCUIT_BREAKER_ENABLED, MAX_CONSECUTIVE_LOSSES
)

def simulate_trade(entry_price, future_prices, profit=PROFIT_TARGET, hold_days=HOLD_DAYS):
    for i, price in enumerate(future_prices[1:hold_days+1]):
        ret = (price - entry_price) / entry_price
        if ret >= profit:
            return profit, i+1, "TP"
        elif ret <= -0.02:
            return ret, i+1, "SL"
    final_ret = (future_prices[min(hold_days, len(future_prices)-1)] - entry_price) / entry_price
    return final_ret, hold_days, "EXP"

def backtest_ticker(ticker, model, window=14):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        df = add_indicators(df)
        df = df.dropna()
        df = df.reset_index()
    except Exception as e:
        print(f"⚠️ {ticker}: {e}")
        return []

    results = []
    loss_streak = 0

    for i in range(len(df) - window - HOLD_DAYS):
        if CIRCUIT_BREAKER_ENABLED and loss_streak >= MAX_CONSECUTIVE_LOSSES:
            print(f"🚫 Circuit breaker triggered for {ticker} after {MAX_CONSECUTIVE_LOSSES} consecutive losses.")
            break

        train = df.iloc[i:i+window]
        test = df.iloc[i+window:i+window+1]
        if test.empty:
            continue

        latest = test[FEATURE_COLS]
        latest.columns = [str(col).strip() for col in latest.columns]
        confidence = model.predict_proba(latest)[0][1]

        if confidence >= CONFIDENCE_THRESHOLD:
            entry_price = test['close'].values[0]
            date = test['Date'].values[0] if 'Date' in test.columns else test.index[i+window]
            future_prices = df['close'].iloc[i+window:i+window+HOLD_DAYS+1].values
            ret, days_held, exit_reason = simulate_trade(entry_price, future_prices)

            if ret < 0:
                loss_streak += 1
            else:
                loss_streak = 0

            results.append({
                "ticker": ticker,
                "date": str(date),
                "entry": entry_price,
                "confidence": confidence,
                "return": round(ret, 4),
                "days_held": days_held,
                "exit": exit_reason
            })

    return results

def run_backtest(tickers):
    model = joblib.load(MODEL_PATH)
    all_results = []

    for ticker in tickers:
        trades = backtest_ticker(ticker, model)
        all_results.extend(trades)

    df = pd.DataFrame(all_results)
    if not df.empty:
        out_path = f"logs/backtest_walkforward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(out_path, index=False)
        print(f"✅ Backtest complete. Logged {len(df)} trades to {out_path}")
    else:
        print("❌ No trades were triggered.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        tickers = sys.argv[1:]
    else:
        df = pd.read_csv("data/nse_equity_list.csv")
        tickers = df['SYMBOL'].astype(str).str.upper().str.strip().apply(lambda x: f"{x}.NS").tolist()[:50]
    run_backtest(tickers)
