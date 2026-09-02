import pandas as pd


def load_sp500_membership(filepath: str) -> pd.DataFrame:
    """
    Loads the CRSP daily S&P 500 constituents file.
    Uses actual columns from the WRDS export.
    """
    usecols = ["PERMNO", "Ticker", "DlyCalDt", "DlyCap"]
    df = pd.read_csv(
        filepath,
        usecols=usecols,
        dtype={"PERMNO": "int64", "Ticker": "string"},
        parse_dates=["DlyCalDt"],
    )
    return df


def rank_top_n_per_quarter(df: pd.DataFrame, n: int = 150) -> pd.DataFrame:
    """
    Ranks by DlyCap (already computed by CRSP) on the last trading
    day of each quarter, keeps the top N names per quarter.
    """
    df = df.copy()
    df["quarter"] = df["DlyCalDt"].dt.to_period("Q")

    last_day_per_quarter = (
        df.sort_values("DlyCalDt")
        .groupby(["PERMNO", "quarter"])
        .tail(1)
    )

    top_n_by_quarter = (
        last_day_per_quarter
        .groupby("quarter", group_keys=False)
        .apply(lambda g: g.nlargest(n, "DlyCap"))
    )

    return top_n_by_quarter


def get_superset_tickers(top_n_by_quarter: pd.DataFrame) -> list:
    """
    Unions the top-N tickers across every quarter, this is your
    full basket universe for the options pull. Using Ticker directly
    since it's already in this file, no separate crosswalk needed.
    """
    return sorted(top_n_by_quarter["Ticker"].dropna().unique().tolist())

def compute_quarterly_weights(top_n_by_quarter: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarter, computes each ticker's weight as its share of
    total market cap among that quarter's top-N basket. This gives
    point-in-time weights, avoiding the look-ahead bias of applying
    today's SPY weights to a historical backtest.
    """
    df = top_n_by_quarter.copy()

    # Reconstruct quarter from DlyCalDt since it's not already present
    df["quarter"] = df["DlyCalDt"].dt.to_period("Q")

    quarter_totals = df.groupby("quarter")["DlyCap"].transform("sum")
    df["weight"] = df["DlyCap"] / quarter_totals

    return df[["quarter", "Ticker", "DlyCap", "weight"]].rename(
        columns={"Ticker": "ticker"}
    )

if __name__ == "__main__":
    INPUT_FILE = "data/Historical_SPY_Components.csv"

    df = load_sp500_membership(INPUT_FILE)
    top150_by_qtr = rank_top_n_per_quarter(df, n=150)
    superset_tickers = get_superset_tickers(top150_by_qtr)

    print(f"Total unique tickers across all quarters: {len(superset_tickers)}")

    pd.Series(superset_tickers, name="ticker").to_csv(
        "data/superset_tickers.csv", index=False
    )

    # NEW: point-in-time quarterly weights table
    quarterly_weights = compute_quarterly_weights(top150_by_qtr)
    quarterly_weights.to_csv("data/quarterly_weights.csv", index=False)
    print(f"Saved {len(quarterly_weights):,} quarter-ticker weight rows to data/quarterly_weights.csv")

    # sanity check: weights within each quarter should sum to ~1.0
    check = quarterly_weights.groupby("quarter")["weight"].sum()
    print(f"\nWeight sum sanity check (should all be ~1.0):")
    print(check.describe())