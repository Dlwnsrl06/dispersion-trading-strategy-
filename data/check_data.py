import pandas as pd

# Re-run the pipeline steps to get top150_by_qtr in memory
# (or save/load it from build_basket_universe.py if you persisted it earlier)

INPUT_FILE = "data/Historical_SPY_Components.csv"

df = load_sp500_membership(INPUT_FILE)
top150_by_qtr = rank_top_n_per_quarter(df, n=150)

# The 15 tickers in question
candidates = [
    'APP', 'BNY', 'BRK-B', 'CMI', 'DDOG', 'DELL', 'GLW', 'HOOD',
    'MRSH', 'MRVL', 'PWR', 'SNDK', 'STX', 'VRT', 'WDC'
]

# Note: BRK-B needs to be checked as BRKB (no hyphen) since that's
# likely how CRSP/your ticker field stores it, ticker formatting
# differs across yfinance, CRSP, and IvyDB
candidates_normalized = [t.replace('-', '') for t in candidates]

for original, normalized in zip(candidates, candidates_normalized):
    matches = top150_by_qtr[top150_by_qtr["Ticker"] == normalized]
    if matches.empty:
        # try the original un-normalized form too, in case it's stored with the hyphen
        matches = top150_by_qtr[top150_by_qtr["Ticker"] == original]

    if matches.empty:
        print(f"{original:8s} -> NOT in top150 at any point 2015-2025 (skip, don't query)")
    else:
        quarters = sorted(matches["quarter"].astype(str).unique())
        print(f"{original:8s} -> IN top150 during: {quarters}")