
import pandas as pd
import numpy as np
import sys
import os

def evaluate(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    if df.empty:
        print("❌ No trades found.")
        return

    # Patch: map 'pnl' to 'return' if needed
    if "return" not in df.columns:
        if "pnl" in df.columns:
            df["return"] = df["pnl"]
        else:
            print("❌ 'return' or 'pnl' column missing in backtest file.")
            return

    # Patch: map 'decision' to 'direction'
    if "direction" not in df.columns and "decision" in df.columns:
        df["direction"] = df["decision"].map({1: "long", -1: "short"})

    total_trades = len(df)
    wins = df[df['return'] > 0]
    losses = df[df['return'] <= 0]
    tp = df[df['exit'] == 'TP'] if 'exit' in df.columns else pd.DataFrame()
    sl = df[df['exit'] == 'SL'] if 'exit' in df.columns else pd.DataFrame()
    exp = df[df['exit'] == 'EXP'] if 'exit' in df.columns else pd.DataFrame()

    avg_return = df['return'].mean()
    win_rate = len(wins) / total_trades
    avg_win = wins['return'].mean() if not wins.empty else 0
    avg_loss = losses['return'].mean() if not losses.empty else 0
    sharpe = np.mean(df['return']) / (np.std(df['return']) + 1e-9) * np.sqrt(252 / 5)

    print(f"📊 Trade Evaluation from: {file_path}")
    print(f"----------------------------------")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Return: {avg_return:.2%}")
    print(f"Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}")
    print(f"Sharpe Ratio (approx): {sharpe:.2f}")

    if not tp.empty or not sl.empty or not exp.empty:
        print(f"Exit Reason Breakdown:")
        print(f"  TP  : {len(tp)}")
        print(f"  SL  : {len(sl)}")
        print(f"  EXP : {len(exp)}")

    if "direction" in df.columns:
        print("\n📈 Directional Stats:")
        for direction in ["long", "short"]:
            sub_df = df[df["direction"] == direction]
            if sub_df.empty:
                print(f"  No {direction.upper()} trades")
                continue

            d_win_rate = (sub_df["return"] > 0).mean()
            d_avg_return = sub_df["return"].mean()
            d_sharpe = np.mean(sub_df['return']) / (np.std(sub_df['return']) + 1e-9) * np.sqrt(252 / 5)

            print(f"  {direction.upper()} Trades: {len(sub_df)}")
            print(f"    Win Rate: {d_win_rate:.2%}")
            print(f"    Avg Return: {d_avg_return:.2%}")
            print(f"    Sharpe: {d_sharpe:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.backtest.evaluate_trades <path_to_backtest_csv>")
    else:
        evaluate(sys.argv[1])
