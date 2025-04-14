import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class QuantumShortSignal:
    """
    Trains on short signals only (label -1), remapped to 1 internally.
    """

    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def fit(self, X, y):
        short_idx = y == -1
        if not any(short_idx):
            raise ValueError("No short signals found in dataset.")

        X_short = X.loc[short_idx].copy()
        y_short = pd.Series(1, index=X_short.index)

    # Select numeric columns only
        X_short = X_short.select_dtypes(include=["number"]).astype("float32")

    # Add dummy 0-class sample
        X_dummy = X_short.head(1).copy()
        y_dummy = pd.Series([0])

    # Concatenate and convert to NumPy (XGBoost-safe)
        X_final_df = pd.concat([X_short, X_dummy], ignore_index=True)
        y_final = pd.concat([y_short, y_dummy], ignore_index=True)

    # ✅ Final fix: convert to NumPy array before passing to fit
        X_final = X_final_df.to_numpy()

        self.model.fit(X_final, y_final)


    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.6):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int) * -1  # 1 → -1 = SHORT