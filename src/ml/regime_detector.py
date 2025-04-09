import pandas as pd
import numpy as np
from hmmlearn import hmm
import joblib
import os

class RegimeEngine:
    def __init__(self, model_path="models/regime_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.features = ['vix', 'nifty_return', 'fii_flow', 'usdinr']

    def fit(self, df):
        df = df.dropna(subset=self.features)
        X = df[self.features].values
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=1000, random_state=42)
        self.model.fit(X)
        joblib.dump(self.model, self.model_path)
        print(f"[RegimeEngine] Model trained and saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No saved regime model found at {self.model_path}")

    def predict_states(self, df):
        if self.model is None:
            self.load()
        df = df.dropna(subset=self.features)
        X = df[self.features].values
        states = self.model.predict(X)
        df['regime'] = states
        return df

    def get_latest_regime(self, df):
        df = df.dropna(subset=self.features)
        X = df[self.features].values
        if self.model is None:
            self.load()
        latest_state = self.model.predict(X[-1].reshape(1, -1))[0]
        return latest_state

    def get_regime_config(self, regime):
        REGIME_ADAPTIVE = {
            0: {"confidence_boost": 0.00, "max_size": 0.03},  # Normal
            1: {"confidence_boost": 0.15, "max_size": 0.02},  # High Volatility
            2: {"confidence_boost": -0.10, "max_size": 0.01}, # Crash
            3: {"confidence_boost": 0.05, "max_size": 0.04},  # Recovery
        }
        return REGIME_ADAPTIVE.get(regime, REGIME_ADAPTIVE[0])
