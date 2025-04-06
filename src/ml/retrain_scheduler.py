import time
import random
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.config import TICKERS, FEATURE_COLS, MODEL_PATH, HOLD_DAYS
from src.indicators.ta_signals import add_indicators
from src.utils.data_fetcher import fetch_stock_data

X_all, y_all = [], []

print("\n📅 Starting scheduled retraining...")
for ticker in tqdm(TICKERS[:100]):  # limit for test; remove [:100] for full market
    try:
        df = fetch_stock_data(ticker, period="1y")
        if df is None or df.empty:
            continue

        df = add_indicators(df).dropna()

        df['future_close'] = df['Close'].shift(-HOLD_DAYS)
        df['target'] = (df['future_close'] > df['Close']).astype(int)
        df.dropna(inplace=True)

        if not all(col in df.columns for col in FEATURE_COLS + ['Close', 'future_close']):
            print(f"⚠️ Error processing {ticker}: Missing columns")
            continue

        X = df[FEATURE_COLS].copy()
        X.columns = X.columns.str.strip()
        y = df['target']

        X_all.append(X)
        y_all.append(y)

        time.sleep(random.uniform(0.5, 1.5))  # avoid rate-limiting

    except Exception as e:
        print(f"⚠️ Error processing {ticker}: {e}")

if not X_all:
    print("❌ No data available for training.")
    exit()

X_final = pd.concat(X_all)
y_final = pd.concat(y_all)

print("\n🧠 Training refined XGBoost model...")
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("\n📊 Evaluation Report:\n")
print(classification_report(y_test, preds))

joblib.dump(model, MODEL_PATH)
print(f"\n✅ Refined model saved to {MODEL_PATH}")
