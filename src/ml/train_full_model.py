# src/ml/train_full_model.py

import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.preprocessing import LabelEncoder
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

    # ✅ Encode labels (0 = hold, 1 = long, 2 = short)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = XGBClassifier(
        objective="multi:softprob",  # ✅ multiclass mode
        num_class=3,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.05],
    }

    clf = GridSearchCV(
        model,
        grid,
        cv=3,
        scoring=make_scorer(f1_score, average="macro"),
        verbose=1,
        n_jobs=-1
    )

    try:
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        preds = best_model.predict(X_test)
        preds = np.asarray(preds)
        y_test = np.asarray(y_test)

        if preds.ndim > 1:
            preds = preds.argmax(axis=1)
        if y_test.ndim > 1:
            y_test = y_test.argmax(axis=1)

        print("\n📊 Final Model Evaluation:\n")
        print(classification_report(y_test, preds))

        joblib.dump(best_model, MODEL_PATH)
        joblib.dump(le, "models/label_encoder.pkl")  # ✅ Save encoder for screener use
        print(f"\n✅ Final model saved to {MODEL_PATH}")
        print(f"✅ Label encoder saved to models/label_encoder.pkl")

    except Exception as e:
        print(f"❌ Training or evaluation failed: {e}")

if __name__ == "__main__":
    X, y = load_batches()
    train_and_save(X, y)
