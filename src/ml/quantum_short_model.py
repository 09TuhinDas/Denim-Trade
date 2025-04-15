
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from src.config import FEATURE_COLS, RAW_DATA_FOLDER, SHORT_MODEL_PATH

def load_short_batches():
    X_all, y_all = [], []
    files = [f for f in os.listdir(RAW_DATA_FOLDER) if f.endswith(".pkl")]
    print(f"📦 Loading {len(files)} batches...")

    for f in files:
        data = joblib.load(open(os.path.join(RAW_DATA_FOLDER, f), "rb"))

        df_X = data["X"].copy().reset_index(drop=True)
        df_y = data["y"].copy().reset_index(drop=True)

        df = df_X.copy()
        df["label"] = df_y

        # ✅ Filter SHORT (-1) and HOLD (0)
        df_filtered = df[df["label"].isin([-1, 0])]
        if len(df_filtered) == 0:
            continue

        extended_features = FEATURE_COLS + ["volatility", "kelly_fraction"]
        available_features = [col for col in extended_features if col in df_filtered.columns]
        missing = set(FEATURE_COLS) - set(df_filtered.columns)
        if missing:
            print(f"⚠️ Missing features in batch {f}: {missing}")

        X_filtered = df_filtered[available_features]
        y_filtered = df_filtered["label"].replace({-1: 1, 0: 0})  # 1 = SHORT, 0 = HOLD

        X_all.append(X_filtered)
        y_all.append(y_filtered)

    return pd.concat(X_all), pd.concat(y_all)

def train_short_model(X, y):
    # Split before calibration to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Balance short (1) and hold (0) with resampling
    X_pos = X_train[y_train == 1]
    y_pos = y_train[y_train == 1]
    X_neg = X_train[y_train == 0]
    y_neg = y_train[y_train == 0]

    # 🛡️ Safety checks
    if len(y_pos) == 0:
        print("❌ No SHORT (label=1) samples found in training set. Cannot train.")
        return
    if len(y_neg) == 0:
        print("❌ No HOLD (label=0) samples found in training set. Cannot train.")
        return

    X_pos_resampled, y_pos_resampled = resample(
        X_pos, y_pos, replace=True, n_samples=len(y_neg), random_state=42
    )

    X_train_final = pd.concat([X_neg, X_pos_resampled])
    y_train_final = pd.concat([y_neg, y_pos_resampled])

    # ✅ Label smoothing
    #y_train_final = y_train_final.replace({1: 0.9, 0: 0.1})

    # Define base XGBoost model with regularization + dropout
    base_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # ✅ Platt scaling for calibration
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    calibrated_model.fit(X_train_final, y_train_final)

    # Save the model
    joblib.dump(calibrated_model, SHORT_MODEL_PATH)
    print(f"✅ Short model saved to {SHORT_MODEL_PATH}")

    # Diagnostics
    print("🔍 Zero input test:", calibrated_model.predict_proba([[0]*X.shape[1]]))
    print("🔍 Real sample test:", calibrated_model.predict_proba([X_test.iloc[0].tolist()]))

if __name__ == "__main__":
    X, y = load_short_batches()
    print("🔍 Global label counts before training:", y.value_counts().to_dict())
    train_short_model(X, y)
