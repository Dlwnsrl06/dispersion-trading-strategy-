import pandas as pd

df = pd.read_csv("data/options_historical_data.csv")

# duplicate check
print(f"Duplicate rows: {df.duplicated().sum()}")

# missing ticker check
superset = set(pd.read_csv("superset_tickers.csv")["ticker"])
present = set(df["ticker"].unique())
missing = superset - present
print(f"Tickers missing entirely: {len(missing)}")
print(sorted(missing))