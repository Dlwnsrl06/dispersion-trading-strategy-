"""
Turns the implied vs. realized correlation spread into a trade signal.

This is intentionally simple (a z-scored spread with fixed thresholds).
Treat it as a starting point, not a finished signal, once you have a
real history of implied correlation values to look at, you'll likely
want to revisit the thresholds and possibly the lookback windows.
"""

import numpy as np
import pandas as pd

import config

DEFAULT_INPUT_PATH = "data/correlation_history.csv"
DEFAULT_OUTPUT_PATH = "data/signal_history.csv"


def compute_spread_series(implied_corr_series, realized_corr_series, zscore_window=None):
    """
    Both inputs are pandas Series indexed by date. Returns a DataFrame
    with the spread and its rolling z-score.

    Positive spread means implied correlation is running above realized,
    which is the condition dispersion trades are designed to monetize
    (sell index vol, buy component vol, betting realized correlation
    stays lower than what's priced in).
    """
    df = pd.DataFrame(
        {"implied": implied_corr_series, "realized": realized_corr_series}
    ).dropna()

    df["spread"] = df["implied"] - df["realized"]

    zscore_window = (
        config.SIGNAL_ZSCORE_WINDOW if zscore_window is None else zscore_window
    )

    rolling_mean = df["spread"].rolling(zscore_window).mean()
    rolling_std = df["spread"].rolling(zscore_window).std()

    df["zscore"] = (df["spread"] - rolling_mean) / rolling_std

    return df


def generate_positions(spread_df, entry_z=None, exit_z=None):
    """
    Simple stateful signal: enter a dispersion trade (short index vol,
    long basket vol) when the z-scored spread crosses above entry_z,
    exit when it falls back below exit_z. Position is 1 (in trade) or
    0 (flat), there's no short-dispersion side implemented here since
    that's a separate, less common trade with its own risk profile.

    Extend this if you want position sizing proportional to z-score
    magnitude rather than a binary in/out signal.
    """
    entry_z = config.ENTRY_ZSCORE if entry_z is None else entry_z
    exit_z = config.EXIT_ZSCORE if exit_z is None else exit_z

    positions = []
    in_position = False

    for z in spread_df["zscore"]:
        if pd.isna(z):
            positions.append(0)
            continue

        if not in_position and z > entry_z:
            in_position = True
        elif in_position and z < exit_z:
            in_position = False

        positions.append(1 if in_position else 0)

    spread_df = spread_df.copy()
    spread_df["position"] = positions
    return spread_df


def build_signal_table(
    correlation_history,
    entry_z=None,
    exit_z=None,
    zscore_window=None,
):
    """
    Takes the output of historical_correlation_series.py and adds
    spread, z-score, and position columns.
    """
    df = correlation_history.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    spread_df = compute_spread_series(
        df["implied_correlation"],
        df["realized_correlation"],
        zscore_window=zscore_window,
    )
    signal_df = generate_positions(spread_df, entry_z=entry_z, exit_z=exit_z)

    passthrough_cols = [
        col
        for col in [
            "quarter",
            "num_components",
            "raw_weight_coverage",
            "index_iv",
            "out_of_bounds",
        ]
        if col in df.columns
    ]
    return df[passthrough_cols].join(signal_df, how="right")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate z-score dispersion signals from correlation history."
    )
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--entry-z", type=float, default=None)
    parser.add_argument("--exit-z", type=float, default=None)
    parser.add_argument("--zscore-window", type=int, default=None)
    args = parser.parse_args()

    correlation_history = pd.read_csv(args.input_path)
    signal_df = build_signal_table(
        correlation_history,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        zscore_window=args.zscore_window,
    )
    signal_df.to_csv(args.output_path, index_label="date")

    print(f"Saved {len(signal_df):,} signal rows to {args.output_path}")
    if not signal_df.empty:
        print(signal_df.tail().to_string())


if __name__ == "__main__":
    main()
