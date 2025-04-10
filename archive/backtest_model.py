from src.config import FEATURE_COLS, CONFIDENCE_THRESHOLD, MODEL_PATH, HOLD_DAYS, PROFIT_TARGET, CONFIDENCE_EXIT_THRESHOLD
import pandas as pd
import joblib
from src.indicators.ta_signals import add_indicators
from src.utils.data_fetcher import fetch_stock_data





def load_model(path="models/xgb_model.pkl"):
    return joblib.load(path)

def backtest(ticker="RELIANCE.NS", model_path="models/xgb_model.pkl", return_df=False):
    model = load_model(model_path)
    df = fetch_stock_data(ticker, period="2y")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = add_indicators(df).dropna()

    trades = []
    i = 0

    while i < len(df) - HOLD_DAYS:
        features = df[FEATURE_COLS].iloc[i:i+1]
        features.columns = pd.Index([c[0].strip() if isinstance(c, tuple) else c.strip() for c in features.columns])
        probs = model.predict_proba(features)[0]
        confidence = probs[1]

        if confidence >= 0.7:
            entry_date = df.index[i]
            entry_price = df['Close'].iloc[i]
            max_conf = confidence
            hold = 1

            for j in range(1, HOLD_DAYS + 1):
                if i + j >= len(df):
                    break

                next_features = df[FEATURE_COLS].iloc[i + j:i + j + 1]
                next_features.columns = pd.Index([c[0].strip() if isinstance(c, tuple) else c.strip() for c in next_features.columns])
                next_conf = model.predict_proba(next_features)[0][1]
                max_conf = max(max_conf, next_conf)

                pnl = (df['Close'].iloc[i + j] - entry_price) / entry_price

                if next_conf < CONFIDENCE_EXIT_THRESHOLD or pnl >= PROFIT_TARGET:
                    break

                hold += 1

            exit_date = df.index[i + hold]
            exit_price = df['Close'].iloc[i + hold]
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl_%": round(pnl_pct, 2),
                "hold_days": hold,
                "max_confidence": round(max_conf, 3)
            })

            i += hold
        else:
            i += 1

    results = pd.DataFrame(trades)

    if return_df:
        return results if not results.empty else None

    if not results.empty:
        results.to_csv(f"backtest_{ticker}.csv", index=False)
        print(f"✅ Backtest complete. {len(results)} trades found.")
        print(f"📄 Results saved to backtest_{ticker}.csv")

        print("\n📊 PnL Summary:")
        print(results['pnl_%'].describe())

        print("\n🔍 Confidence Summary:")
        print(results['max_confidence'].describe())
    else:
        print("❌ No BUY trades triggered during backtest.")

# For standalone testing
if __name__ == "__main__":
    backtest("RELIANCE.NS")
