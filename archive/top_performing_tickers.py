# src/backtest/top_performing_tickers.py

import pandas as pd
import sys
import os

def top_performers(file_path, min_trades=3, top_n=50):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    grouped = df.groupby("ticker").agg(
        total_trades=('return', 'count'),
        win_rate=('return', lambda x: (x > 0).mean()),
        avg_return=('return', 'mean')
    )

    filtered = grouped[grouped['total_trades'] >= min_trades]
    sorted_df = filtered.sort_values(by=["win_rate", "avg_return"], ascending=False).head(top_n)

    print(f"🎯 Top {len(sorted_df)} Performing Tickers:\n{sorted_df}\n")

    # Save list to file for use in retraining/screening
    tickers = sorted_df.index.tolist()
    out_path = "data/top_tickers.txt"
    with open(out_path, "w") as f:
        for t in tickers:
            f.write(t + "\n")

    print(f"✅ Saved top tickers list to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.backtest.top_performing_tickers <backtest_log.csv>")
    else:
        top_performers(sys.argv[1])
