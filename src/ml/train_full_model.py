# src/ml/train_full_model.py

import os
import glob
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from src.config import FEATURE_COLS, RAW_DATA_FOLDER, MODEL_PATH


def load_batches():
    X_all, y_all = [], []
    files = glob.glob(os.path.join(RAW_DATA_FOLDER, "batch_*.pkl"))
    print(f"📦 Loading {len(files)} batches...")

    for f in files:
        data = joblib.load(open(f, "rb"))
        X_all.append(data["X"][FEATURE_COLS])
        y_all.append(data["y"])

    X_final = pd.concat(X_all)
    y_final = pd.concat(y_all)
    return X_final, y_final


def train_and_save(X, y):
    print("🧠 Training XGBoost model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Basic GridSearch (can be extended)
    pos_weight = len(y_train[y_train==0]) / (len(y_train[y_train==1]) + 1e-6)
    model = XGBClassifier(
    scale_pos_weight=pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.05],
    }

    clf = GridSearchCV(model, grid, cv=3, scoring="f1", verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    preds = best_model.predict(X_test)

    print("\n📊 Final Model Evaluation:\n")
    print(classification_report(y_test, preds))

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    X, y = load_batches()
    train_and_save(X, y)
