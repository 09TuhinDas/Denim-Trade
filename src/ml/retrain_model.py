from src.config import TICKERS
from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET, CONFIDENCE_EXIT_THRESHOLD
import os
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.indicators.ta_signals import add_indicators
from utils.data_fetcher import fetch_stock_data





def prepare_stock_data(ticker, period="1y"):
    df = fetch_stock_data(ticker, period=period)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = add_indicators(df).dropna()

    if df.empty or len(df) < HOLD_DAYS + 1:
        return None

    # Label creation
    df['future_close'] = df['Close'].shift(-HOLD_DAYS)
    df.dropna(subset=['future_close'], inplace=True)
    df['target'] = (df['future_close'] > df['Close']).astype(int)

    features = df[FEATURE_COLS]
    labels = df['target']
    return features, labels

def build_dataset(tickers):
    all_X, all_y = [], []

    print("\U0001F4C8 Preparing training dataset...")

    for ticker in tqdm(tickers):
        try:
            result = prepare_stock_data(ticker)
            if result:
                X, y = result
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"\u26A0\uFE0F Skipping {ticker} due to error: {e}")

    if not all_X:
        raise ValueError("\u274C No data collected for training. Check tickers or data source.")

    X_final = pd.concat(all_X)
    y_final = pd.concat(all_y)

    return X_final, y_final

def train_and_save_model(X, y):
    print("\U0001F9E0 Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\n\U0001F4CA Evaluation Report:\n")
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    X, y = build_dataset(TICKERS)
    train_and_save_model(X, y)
