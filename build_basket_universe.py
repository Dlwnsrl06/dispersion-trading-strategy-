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


if __name__ == "__main__":
    INPUT_FILE = "data/Historical_SPY_Components.csv"

    df = load_sp500_membership(INPUT_FILE)
    top150_by_qtr = rank_top_n_per_quarter(df, n=150)
    superset_tickers = get_superset_tickers(top150_by_qtr)

    print(f"Total unique tickers across all quarters: {len(superset_tickers)}")
    print(superset_tickers)

    pd.Series(superset_tickers, name="ticker").to_csv(
        "superset_tickers.csv", index=False
    )