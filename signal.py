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


def compute_spread_series(implied_corr_series, realized_corr_series):
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

    rolling_mean = df["spread"].rolling(config.SIGNAL_ZSCORE_WINDOW).mean()
    rolling_std = df["spread"].rolling(config.SIGNAL_ZSCORE_WINDOW).std()

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
