"""
Automatically fetches SPY's daily holdings file directly from State
Street (SSGA) and rebuilds spy_weights.csv, replacing the manual
download-then-run-build_weights.py workflow.

Run this directly to refresh weights on demand:
    python fetch_spy_weights.py --top 50

Or import refresh_weights_if_stale() and call it from main.py so the
weights file refreshes itself automatically without you having to
remember to run this separately.

Known unknowns, worth verifying once you run this locally: SSGA's xlsx
has a few metadata rows (fund name, as-of date) before the actual
holdings table starts. skiprows=4 is the commonly reported value for
this file's layout, but SSGA has changed this file's structure before
without notice (see the URL history in tools like tidyquant), so if the
column names below come back wrong, print the raw dataframe with
skiprows adjusted up or down by 1-2 to find the real header row.
"""

import argparse
import csv
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

SSGA_SPY_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/institutional/etfs/library-content/"
    "products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)

# SSGA's server has been known to reject requests that don't look like
# they're coming from a real browser. A standard browser User-Agent
# avoids that.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "data", "spy_weights.csv")


def fetch_holdings_dataframe():
    """
    Downloads the raw SSGA holdings file and returns it as a cleaned
    DataFrame with 'ticker' and 'weight' columns, sorted by weight
    descending.
    """
    response = requests.get(SSGA_SPY_HOLDINGS_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    raw_path = os.path.join(os.path.dirname(__file__), "spy_holdings_raw.xlsx")
    with open(raw_path, "wb") as f:
        f.write(response.content)

    # Dump a raw, unprocessed CSV of the whole sheet, no header
    # assumption, no filtering. This is purely for you to open and
    # visually inspect in a normal text editor, since xlsx files show
    # up as binary garbage in VS Code. Use this to confirm the real
    # header row number if the skiprows value below ever needs updating.
    debug_csv_path = os.path.join(os.path.dirname(__file__), "spy_holdings_raw_debug.csv")
    pd.read_excel(raw_path, header=None).to_csv(debug_csv_path, index=False, header=False)
    print(f"Saved a readable debug copy to {debug_csv_path}, open that to inspect the raw layout.")

    # SSGA's file typically has a few rows of fund metadata before the
    # real header row. Adjust skiprows if the columns below don't show
    # up, check spy_holdings_raw_debug.csv above to find the real
    # header row number.
    df = pd.read_excel(raw_path, skiprows=4)

    # Column names in SSGA's file are typically "Name", "Ticker",
    # "Identifier", "SEDOL", "Weight", "Sector", "Shares Held",
    # "Local Currency". Normalize to lowercase to make matching robust
    # to minor naming changes.
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "ticker" not in df.columns or "weight" not in df.columns:
        raise ValueError(
            f"Expected 'ticker' and 'weight' columns, got: {list(df.columns)}. "
            "SSGA likely changed their file layout, inspect the raw file "
            "manually to find the correct column names and skiprows."
        )

    df = df[["ticker", "weight"]].dropna()

    # SSGA writes share-class tickers with a period (e.g. "BRK.B"), but
    # yfinance and most other free data sources expect a hyphen instead
    # (e.g. "BRK-B"). Without this, any share-class ticker silently fails
    # to fetch later and just gets dropped, no crash, just a quietly
    # missing component every run.
    df["ticker"] = df["ticker"].astype(str).str.replace(".", "-", regex=False)

    # Weight is usually already a percentage number (e.g. 7.02 meaning
    # 7.02%), convert to decimal to match what config.py/correlation.py
    # expect (weights as decimals, e.g. 0.0702).
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce") / 100
    df = df.dropna(subset=["weight"])

    # Drop the cash/other line SSGA sometimes includes at the bottom.
    df = df[df["ticker"].astype(str).str.match(r"^[A-Z.\-]+$")]

    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def build_weights_csv(top_n=50, output_path=WEIGHTS_PATH):
    """
    Fetches fresh holdings, takes the top N by weight, and writes
    spy_weights.csv in the format config.py expects.
    """
    df = fetch_holdings_dataframe()
    top = df.head(top_n)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "weight"])
        for _, row in top.iterrows():
            writer.writerow([row["ticker"], row["weight"]])

    coverage = top["weight"].sum()
    print(f"Wrote {len(top)} tickers to {output_path}")
    print(f"Basket coverage: {coverage:.1%} of SPY's total weight")
    return output_path


def refresh_weights_if_stale(max_age_hours=20, top_n=50):
    """
    Checks the age of the existing spy_weights.csv and rebuilds it if
    it's older than max_age_hours (defaults to just under a day, since
    SSGA updates once per trading day). Call this at the start of
    main.py to keep weights current without a separate manual step.

    Safe to call every run: if the file is fresh, this does nothing but
    a cheap file-time check, no network call.
    """
    if not os.path.exists(WEIGHTS_PATH):
        print("No spy_weights.csv found, fetching for the first time...")
        return build_weights_csv(top_n=top_n)

    age = time.time() - os.path.getmtime(WEIGHTS_PATH)
    if age > max_age_hours * 3600:
        print(f"spy_weights.csv is {age / 3600:.1f} hours old, refreshing...")
        return build_weights_csv(top_n=top_n)

    print(f"spy_weights.csv is {age / 3600:.1f} hours old, still fresh, skipping refresh.")
    return WEIGHTS_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50, help="Number of top-weighted tickers to keep")
    parser.add_argument("--force", action="store_true", help="Refresh even if the file looks fresh")
    args = parser.parse_args()

    if args.force:
        build_weights_csv(top_n=args.top)
    else:
        refresh_weights_if_stale(top_n=args.top)
