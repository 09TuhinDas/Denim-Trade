from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET, CONFIDENCE_EXIT_THRESHOLD
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.indicators.ta_signals import add_indicators
from src.utils.data_fetcher import fetch_stock_data
from src.utils.labeling import compute_swing_label_with_short 

def prepare_dataset(df):
    # Add indicators
    df = add_indicators(df)
    df = compute_swing_label_with_short(df, profit_target=PROFIT_TARGET, hold_days=HOLD_DAYS)

    required_cols = FEATURE_COLS + ["label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=required_cols)
    X = df[FEATURE_COLS]
    y = df["label"]
    return X, y

def train_xgboost_model(X, y):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    if preds.ndim > 1:  # Convert softprob to class predictions
        preds = preds.argmax(axis=1)
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, "models/label_encoder.pkl")
    print(f"✅ Model saved to {MODEL_PATH}")
    print("✅ Label encoder saved to models/label_encoder.pkl")

# Entry point
if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS", period="1y")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    X, y = prepare_dataset(df)
    X.columns = X.columns.str.strip()
    train_xgboost_model(X, y)
