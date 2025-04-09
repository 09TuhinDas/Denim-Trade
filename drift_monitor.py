import pandas as pd
import joblib
import os
import yfinance as yf
from scipy.stats import wasserstein_distance
from src.indicators.ta_signals import add_indicators
from src.config import FEATURE_COLS

TRAINED_DATA_PATH = "data/training_batches/"
TOP_TICKERS_PATH = "data/top_tickers.txt"


def load_combined_training_data():
    """Concatenate all training batches from joblib dictionaries."""
    dfs = []
    for file in os.listdir(TRAINED_DATA_PATH):
        if file.endswith(".pkl"):
            path = os.path.join(TRAINED_DATA_PATH, file)
            try:
                batch = joblib.load(path)
                df = batch["X"].copy()
                df["label"] = batch["y"]
                df["ticker"] = batch.get("tickers", [None] * len(df))
                dfs.append(df)
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
    if not dfs:
        raise FileNotFoundError("No valid training batches found.")
    return pd.concat(dfs, ignore_index=True)


def load_live_data():
    """Load real-time model-ready features using top tickers and volume filtering."""
    if not os.path.exists(TOP_TICKERS_PATH):
        raise FileNotFoundError("top_tickers.txt not found")

    with open(TOP_TICKERS_PATH) as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    dfs = []
    for t in tickers:
        try:
            df = yf.download(t, period="3mo", progress=False)
            df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).strip().lower() for col in df.columns]
            df = add_indicators(df).dropna()
            df["ticker"] = t
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {t}: {e}")

    if not dfs:
        raise ValueError("No valid tickers downloaded.")

    full_df = pd.concat(dfs, ignore_index=True)

    # ✅ Remove extreme OBV outliers
    obv_cap = full_df["obv"].quantile(0.99)
    full_df = full_df[full_df["obv"] < obv_cap]

    # ✅ Keep top 30% volume rows
    vol_cutoff = full_df["volume"].quantile(0.7)
    return full_df[full_df["volume"] >= vol_cutoff]


def compute_drift_scores(train_df, live_df):
    scores = {}
    for col in FEATURE_COLS:
        if col in train_df.columns and col in live_df.columns:
            try:
                scores[col] = wasserstein_distance(train_df[col].dropna(), live_df[col].dropna())
            except Exception as e:
                scores[col] = f"Error: {e}"
        else:
            scores[col] = "Missing in one dataset"
    return scores


def report_drift():
    print("\n📊 Drift Monitoring Report")
    train_df = load_combined_training_data()
    live_df = load_live_data()
    drift_scores = compute_drift_scores(train_df, live_df)

    print("\nFeature Drift (Wasserstein Distance):")
    for k, v in sorted(drift_scores.items(), key=lambda x: -float(x[1]) if isinstance(x[1], float) else 0):
        print(f"{k:<20}: {v}")


if __name__ == "__main__":
    report_drift()
