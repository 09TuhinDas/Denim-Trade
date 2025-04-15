
import os
import glob
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.config import FEATURE_COLS, RAW_DATA_FOLDER

LONG_MODEL_PATH = "models/xgb_long_trained.pkl"

def load_long_batches():
    X_all, y_all = [], []
    files = glob.glob(os.path.join(RAW_DATA_FOLDER, "batch_*.pkl"))
    print(f"📦 Loading {len(files)} batches...")

    for f in files:
        data = joblib.load(open(f, "rb"))
        df_X = data["X"].copy().reset_index(drop=True)
        df_y = data["y"].copy().reset_index(drop=True)

        df = df_X.copy()
        df["label"] = df_y

        # ✅ Filter LONG (1) and HOLD (0)
        df_filtered = df[df["label"].isin([0, 1])]
        if len(df_filtered) == 0:
            continue

        available_features = [col for col in FEATURE_COLS if col in df_filtered.columns]
        missing = set(FEATURE_COLS) - set(df_filtered.columns)
        if missing:
            print(f"⚠️ Missing features in batch {f}: {missing}")

        X_filtered = df_filtered[available_features]
        y_filtered = df_filtered["label"].replace({1: 1, 0: 0})  # 1 = LONG, 0 = HOLD

        X_all.append(X_filtered)
        y_all.append(y_filtered)

    return pd.concat(X_all), pd.concat(y_all)

def train_and_save(X, y):
    print(f"🧮 Final dataset size: {X.shape}, Labels: {y.value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.8,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    joblib.dump(model, LONG_MODEL_PATH)
    print(f"✅ LONG model saved to {LONG_MODEL_PATH}")
    print("🔍 Sample prediction:", model.predict_proba([X_test.iloc[0].tolist()]))

if __name__ == "__main__":
    X, y = load_long_batches()
    train_and_save(X, y)
