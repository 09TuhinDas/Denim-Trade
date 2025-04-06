from src.config import TICKERS

from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET, CONFIDENCE_EXIT_THRESHOLD
import pandas as pd
from src.backtest.backtest_model import backtest



all_trades = []

for ticker in TICKERS:
    print(f"\n🔁 Running backtest for {ticker}...")
    try:
        df = backtest(ticker, return_df=True)
        if df is not None and not df.empty:
            df['ticker'] = ticker
            all_trades.append(df)
    except Exception as e:
        print(f"⚠️ Error in backtest for {ticker}: {e}")

if all_trades:
    combined = pd.concat(all_trades, ignore_index=True)
    combined.to_csv("backtest_all_results.csv", index=False)
    print("\n✅ Combined backtest complete.")
    print("📄 Results saved to backtest_all_results.csv")

    print("\n📊 Per-stock PnL Summary:")
    summary = combined.groupby("ticker")["pnl_%"].describe()
    print(summary)
else:
    print("❌ No trades generated for any ticker.")
