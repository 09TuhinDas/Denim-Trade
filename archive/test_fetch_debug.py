# test_fetch_debug.py

from src.utils.macro_features import fetch_macro_series

vix = fetch_macro_series("^INDIAVIX", "vix")
nifty = fetch_macro_series("^NSEI", "nifty", fallback_symbol="NIFTYBEES.NS")
usdinr = fetch_macro_series("USDINR=X", "usdinr")

print("\n✅ Type Check & Empty Status")
for name, series in [("vix", vix), ("nifty", nifty), ("usdinr", usdinr)]:
    print(f"{name}: type={type(series)}, empty={series is None or series.empty}")

print("\n📈 Last few rows (if available):")
if vix is not None and not vix.empty:
    print("\nVIX:")
    print(vix.tail())

if nifty is not None and not nifty.empty:
    print("\nNIFTY:")
    print(nifty.tail())

if usdinr is not None and not usdinr.empty:
    print("\nUSDINR:")
    print(usdinr.tail())

