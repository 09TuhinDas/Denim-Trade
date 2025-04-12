from src.utils.path_manager import *

import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder  # ✅ NEW

def main():
    meta_data_path = META_TRAINING
    assert os.path.exists(meta_data_path), f"Meta training file not found: {meta_data_path}"

    meta_df = pd.read_csv(meta_data_path)

    # Features and label (label = 0: hold, 1: long, 2: short)
    features = ["xgb_conf", "lgb_conf", "avg_conf", "conf_diff"]
    X = meta_df[features]
    y = meta_df["label"]

    # ✅ Label encode (if not already integers)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y_encoded):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # Multi-class logistic regression with softmax
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(estimator=LogisticRegression(max_iter=1000), method="sigmoid", cv=5))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate classification performance
    preds = pipe.predict(X_test)
    print("\n✅ Stacked Meta-Model Classification Report:\n")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs("models", exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/stacked_model_{version}.pkl"
    joblib.dump(pipe, model_path)
    print(f"💾 Saved: {model_path}")

    # Save prediction CSV
    pred_df = X_test.copy()
    pred_df["true_label"] = y_test
    pred_df["pred_label"] = preds
    pred_df.to_csv(f"models/stacked_preds_{version}.csv", index=False)
    print(f"📄 Saved predictions: models/stacked_preds_{version}.csv")

if __name__ == "__main__":
    main()
