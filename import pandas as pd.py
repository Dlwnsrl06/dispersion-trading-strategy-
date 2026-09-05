import pandas as pd
df = pd.read_csv("data/correlation_history.csv", parse_dates=["date"])

print(df["num_components"].describe())
print()

df["year"] = df["date"].dt.year
print(df.groupby("year")["num_components"].agg(["mean", "std", "min", "max"]))
print()

df["component_change"] = df["num_components"].diff()
big_jumps = df[df["component_change"].abs() > 15]
print(f"Days with jump > 15: {len(big_jumps)}")
print(big_jumps[["date", "num_components", "component_change"]].to_string(index=False))