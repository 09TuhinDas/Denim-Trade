from src.utils.macro_features import fetch_macro_series

vix = fetch_macro_series("^INDIAVIX", "vix")
nifty = fetch_macro_series("^NSEI", "nifty")
usdinr = fetch_macro_series("USDINR=X", "usdinr")

print("\n✅ Final Checks")
for name, series in {"vix": vix, "nifty": nifty, "usdinr": usdinr}.items():
    print(f"{name}:")
    print("  ➤ Type:", type(series))
    print("  ➤ Empty:", series.empty if isinstance(series, pd.Series) else "❌ Not a Series")
    print("  ➤ Head:")
    print(series.head() if isinstance(series, pd.Series) else series)
    print("-" * 40)
