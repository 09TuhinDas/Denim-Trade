import os
import glob
import joblib
import pandas as pd
from src.ml.quantum_short_model import QuantumShortSignal
from src.config import FEATURE_COLS

BATCH_FOLDER = "data/training_batches"
MODEL_SAVE_PATH = "models/short_model.pkl"

def load_all_batches():
    batch_files = glob.glob(os.path.join(BATCH_FOLDER, "*.pkl"))
    all_X, all_y = [], []
    for file in batch_files:
        data = joblib.load(file)
        all_X.append(data["X"])
        all_y.append(data["y"])
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    return X, y

def main():
    print("📦 Loading all training batches...")
    X, y = load_all_batches()
    print(f"🧾 Total samples: {len(X)}")
    print("🔢 Label distribution:", y.value_counts().to_dict())

    short_count = (y == -1).sum()
    if short_count == 0:
        print("❌ No short signals found in dataset. Model not trained.")
        return

    print(f"🔍 Found {short_count} short signals. Training QuantumShortSignal model...")
    selected_cols = FEATURE_COLS.copy() + ["volatility", "kelly_fraction"]
    selected_cols = list(dict.fromkeys(selected_cols))  # ✅ removes duplicates
    X = X[selected_cols].copy()
    X = X.loc[:, selected_cols].astype("float32")
    print("📋 X columns before cleaning:", list(X.columns))
    print("🧮 Shape before .fit():", X.shape)
    print("✅ Using columns:", selected_cols)

    model = QuantumShortSignal()
    model.fit(X, y)
    print(f"💾 Saving trained short model to {MODEL_SAVE_PATH}")
    joblib.dump(model.model, MODEL_SAVE_PATH)
    print("✅ Short model training complete!")

if __name__ == "__main__":
    main()