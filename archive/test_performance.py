from src.metrics.performance import analyze
import pandas as pd

# Sample trades
df = pd.DataFrame({
    'returns': [0.03, -0.02, 0.01],
    'pnl': [300, -200, 100],
    'success': [1, 0, 1],
    'market_cap': [400e6, 700e6, 300e6],
    'adv': [10e6, 50e6, 60e6],
    'momentum_score': [0.8, 0.4, 0.9],
})

# Dummy VIX series
vix = pd.Series([10, 12, 30])

# Run analysis
summary = analyze(df, vix)
print(summary)
