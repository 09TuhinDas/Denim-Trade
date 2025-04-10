
import os
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
from src.indicators.ta_signals import add_indicators
from src.config import (
    FEATURE_COLS, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET,
    CONFIDENCE_THRESHOLD, CIRCUIT_BREAKER_ENABLED, MAX_CONSECUTIVE_LOSSES, CAPITAL, RISK_PER_TRADE, CUMULATIVE_DRAWDOWN_LIMIT
)
from src.utils.macro_features import load_macro_cache
from src.ml.regime_detector import RegimeEngine


def simulate_trade(entry_price, future_prices, profit=PROFIT_TARGET, hold_days=HOLD_DAYS):
    for i, price in enumerate(future_prices[1:hold_days+1]):
        sell_price = price * 0.998  # 0.2% exit slippage
        ret = (sell_price - entry_price) / entry_price
        if ret >= profit:
            return profit, i + 1, "TP"
        elif ret <= -0.02:
            return ret, i + 1, "SL"
    # If expiry hit
    final_sell = future_prices[min(hold_days, len(future_prices) - 1)] * 0.998
    final_ret = (final_sell - entry_price) / entry_price
    return final_ret, hold_days, "EXP"


def backtest_ticker(ticker, model, window=14):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]
        df = add_indicators(df)
        df = df.dropna().reset_index()
    except Exception as e:
        print(f"⚠️ {ticker}: {e}")
        return []
    
    macro_df = load_macro_cache()
    regime_engine = RegimeEngine()
    regime_state = regime_engine.get_latest_regime(macro_df)

    results = []
    loss_streak = 0
    cumulative_pnl = 0

    for i in range(len(df) - window - HOLD_DAYS):
        test = df.iloc[i+window:i+window+1]
        if test.empty:
            continue
        # Consecutive loss circuit breaker
        if CIRCUIT_BREAKER_ENABLED and loss_streak >= MAX_CONSECUTIVE_LOSSES:
            print(f"🚫 Circuit breaker: {ticker} hit {MAX_CONSECUTIVE_LOSSES} losses.")
            break

        date = test['Date'].values[0] if 'Date' in test.columns else df.index[i+window]

       

        if regime_state == 2:
            print(f"📉 Skipping {ticker} on {date} — regime {regime_state}")
            continue
        
        if len(results) >= 5:
            print(f"🛑 Trade cap hit for {ticker}")
            break

        # Cumulative loss breaker
        if cumulative_pnl <= CUMULATIVE_DRAWDOWN_LIMIT:
            print(f"🛑 Cumulative loss breaker: {ticker} loss ₹{abs(cumulative_pnl):.0f}")
            break

        train = df.iloc[i:i+window]
        test = df.iloc[i+window:i+window+1]
        if test.empty:
            continue

        latest = test[FEATURE_COLS]
        latest.columns = [str(col).strip() for col in latest.columns]
        confidence = model.predict_proba(latest)[0][1]

        if confidence >= CONFIDENCE_THRESHOLD:
            raw_entry_price = test['close'].values[0]
            entry_price = raw_entry_price * 1.001  # +0.1% slippage
            date = test['Date'].values[0] if 'Date' in test.columns else test.index[i+window]
            future_prices = df['close'].iloc[i+window:i+window+HOLD_DAYS+1].values

            # Apply slippage on exit inside simulate_trade()
            ret, days_held, exit_reason = simulate_trade(entry_price, future_prices)

            # Position sizing
            position_size = (CAPITAL * RISK_PER_TRADE) / entry_price
            trade_pnl = position_size * ret  

            # Update loss counters
            cumulative_pnl += trade_pnl
            if trade_pnl < 0:
                loss_streak += 1
            else:
                loss_streak = 0

            print(f"📊 {ticker} | {exit_reason} | Conf: {confidence:.2f} | Ret: {ret:.2%} | PnL: ₹{trade_pnl:.2f}")

            results.append({
                "ticker": ticker,
                "date": str(date),
                "entry": round(entry_price, 2),
                "confidence": round(confidence, 4),
                "return": round(ret, 4),
                "pnl": round(trade_pnl, 2),
                "position_size": round(position_size, 2),
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
