import pandas as pd
from src.ml.regime_detector import RegimeEngine

def main():
    df = pd.read_csv("data/macro_cache.csv")
    regime = RegimeEngine()
    regime.fit(df)

if __name__ == "__main__":
    main()
