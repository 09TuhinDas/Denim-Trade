# src/utils/risk_management.py

import numpy as np
from arch import arch_model

# ------------------------------
# Volatility Forecast (GARCH)
# ------------------------------
def garch_volatility(returns):
    """
    Forecast next-period volatility using GARCH(1,1).
    Returns annualized volatility (daily scale * sqrt(252))
    """
    try:
        am = arch_model(returns * 100, vol='Garch', p=1, q=1)
        res = am.fit(update_freq=0, disp='off')
        daily_vol = res.conditional_volatility[-1] / 100  # Convert back to decimal
        return daily_vol * np.sqrt(252)  # Annualized
    except Exception as e:
        print(f"⚠️ GARCH failed: {e}")
        return np.nan

# ------------------------------
# Adaptive Kelly Sizing
# ------------------------------
def dynamic_kelly_size(confidence, mu, sigma, regime_multiplier):
    """
    Kelly = (mu / sigma^2) * confidence * regime boost
    - mu = expected return (3% for swing trades)
    - sigma = volatility
    - confidence = model's predict_proba()
    - regime_multiplier = from config per regime
    """
    try:
        raw_kelly = (mu / (sigma ** 2 + 1e-6)) * confidence
        return min(raw_kelly * regime_multiplier, 0.02)  # Max 2% sizing
    except:
        return 0.0
