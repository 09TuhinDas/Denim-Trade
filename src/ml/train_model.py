from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET, CONFIDENCE_EXIT_THRESHOLD
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.indicators.ta_signals import add_indicators
from utils.data_fetcher import fetch_stock_data

def prepare_dataset(df):
    # Flatten MultiIndex if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    required_cols = ['Close', 'rsi', 'macd', 'ema20', 'adx', 'bb_width', 'obv']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.dropna(subset=required_cols)
    df['future_close'] = df['Close'].shift(-5)
    df = df.dropna(subset=['future_close'])

    # ✅ THIS LINE FIXES the alignment issue
    df['target'] = (df['future_close'].values > df['Close'].values).astype(int)

    X = df[required_cols]
    y = df['target']
    return X, y



def train_xgboost_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    # Save the model
    joblib.dump(model, "models/xgb_model.pkl")
    print("✅ Model saved to models/xgb_model.pkl")

# Entry point
if __name__ == "__main__":
    df = fetch_stock_data("RELIANCE.NS", period="1y")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = add_indicators(df)
    print(df.columns)  # Check what's coming back
    X, y = prepare_dataset(df)
    X.columns = X.columns.str.strip()
    train_xgboost_model(X, y)
