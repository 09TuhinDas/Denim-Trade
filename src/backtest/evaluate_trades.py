
# src/backtest/evaluate_trades.py

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

    total_trades = len(df)
    wins = df[df['return'] > 0]
    losses = df[df['return'] <= 0]
    tp = df[df['exit'] == 'TP']
    sl = df[df['exit'] == 'SL']
    exp = df[df['exit'] == 'EXP']

    avg_return = df['return'].mean()
    win_rate = len(wins) / total_trades
    avg_win = wins['return'].mean() if not wins.empty else 0
    avg_loss = losses['return'].mean() if not losses.empty else 0

    # Sharpe ratio approximation
    sharpe = np.mean(df['return']) / (np.std(df['return']) + 1e-9) * np.sqrt(252 / 5)

    print(f"📊 Trade Evaluation from: {file_path}")
    print(f"----------------------------------")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Return: {avg_return:.2%}")
    print(f"Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}")
    print(f"Sharpe Ratio (approx): {sharpe:.2f}")
    print(f"Exit Reason Breakdown:")
    print(f"  TP  : {len(tp)}")
    print(f"  SL  : {len(sl)}")
    print(f"  EXP : {len(exp)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.backtest.evaluate_trades <path_to_backtest_csv>")
    else:
        evaluate(sys.argv[1])
