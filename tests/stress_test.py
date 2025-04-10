import pandas as pd
import joblib
from src.backtest.backtest_walk_forward import backtest_ticker
from src.utils.macro_features import load_macro_cache

STARTING_CAPITAL = 100_000

def simulate_market_crash(tickers=None):
    if tickers is None:
        tickers = ['YESBANK.NS', 'IBULHSGFIN.NS', 'ADANIPOWER.NS', 'PNB.NS', 'ZEEL.NS']

    model = joblib.load("models/xgb_calibrated.pkl")
    all_trades = []

    for ticker in tickers:
        print(f"\n📉 Running crash test for {ticker}...")

        # Optional: check VIX during crash (skip trades if needed)
        # macro_df = load_macro_cache()
        # vix_now = macro_df["vix"].iloc[-1]
        # if vix_now > 25:
        #     print(f"⚠️ Skipping {ticker} due to high VIX ({vix_now:.2f})")
        #     continue

        trades = backtest_ticker(ticker, model)
        if trades:
            all_trades.extend(trades)

    return pd.DataFrame(all_trades)

def test_black_swan():
    trades_df = simulate_market_crash()

    if trades_df.empty:
        print("❌ No trades found during crash window.")
        return

    # Track cumulative drawdown
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    max_drawdown_raw = (trades_df['cumulative_pnl'].cummax() - trades_df['cumulative_pnl']).max()

    # Normalize to % of capital
    normalized_drawdown = max_drawdown_raw / STARTING_CAPITAL

    print(f"\n✅ Max Drawdown during COVID crash: {normalized_drawdown:.2%}")
    assert normalized_drawdown < 0.25, "❌ Drawdown exceeded safe threshold during black swan!"

if __name__ == "__main__":
    test_black_swan()
