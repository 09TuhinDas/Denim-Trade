from src.indicators.microstructure import compute_imbalance
import pandas as pd

# Sample order book data
df = pd.DataFrame({
    'bid1_qty': [100, 200],
    'bid2_qty': [50, 60],
    'ask1_qty': [80, 190],
    'ask2_qty': [70, 50],
})

# Run imbalance
df = compute_imbalance(df)
print(df[['bid1_qty', 'ask1_qty', 'imbalance']])
