
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main():
    meta_data_path = "data/meta_training.csv"
    assert os.path.exists(meta_data_path), f"Meta training file not found: {meta_data_path}"

    meta_df = pd.read_csv(meta_data_path)

    # Define features and target
    features = ["xgb_conf", "lgb_conf", "avg_conf", "conf_diff"]
    X = meta_df[features]
    y = meta_df["label"]

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Logistic Regression + Calibration in a pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="sigmoid", cv=5))
    ])

    pipe.fit(X_train, y_train)

    # Evaluate AUC
    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"✅ Stacked model AUC: {auc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/stacked_model_{version}.pkl"
    joblib.dump(pipe, model_path)
    print(f"💾 Saved: {model_path}")

    # Save prediction CSV (optional)
    pred_df = X_test.copy()
    pred_df["true_label"] = y_test.values
    pred_df["pred_prob"] = probs
    pred_df.to_csv(f"models/stacked_preds_{version}.csv", index=False)
    print(f"📄 Saved predictions: models/stacked_preds_{version}.csv")

if __name__ == "__main__":
    main()
