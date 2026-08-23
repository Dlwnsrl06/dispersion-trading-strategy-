"""
Backtest skeleton for the dispersion trade.

Honest scope note: a fully realistic backtest would track actual option
positions (short index straddles/options, long a vega-weighted basket
of component options), roll them at each expiry, and account for bid-
ask spreads, assignment risk, and margin. That's a substantial amount
of additional work and is the right place for you two to spend real
design time rather than have it handed to you.

What's here instead is a variance-notional approximation: when the
strategy is "in position," it earns/loses based on the spread between
implied and realized correlation converging or diverging, scaled by a
notional. This is a legitimate simplification used for a first-pass
sanity check on whether the correlation signal has any edge at all,
before you invest time building full position-level P&L tracking. Be
upfront in interviews that this is what you started with and describe
what you added on top of it.
"""

import numpy as np
import pandas as pd


def run_backtest(spread_df, notional=1_000_000, transaction_cost_bps=5):
    """
    spread_df: output of signal.generate_positions, must have columns
               'spread', 'position', indexed by date

    notional: dollar notional scaling factor for the P&L approximation

    transaction_cost_bps: round-trip cost charged in basis points of
                           notional each time the position flips, as a
                           placeholder for real bid-ask/commission costs

    Returns spread_df with added 'pnl' and 'cumulative_pnl' columns.

    P&L logic: while in position, you profit as the spread mean-reverts
    (shrinks) from where you entered, since that's the realized outcome
    a genuine short-index/long-basket trade would be profiting from.
    This is a proxy for real option P&L, not a substitute for it.
    """
    df = spread_df.copy()
    df["position_change"] = df["position"].diff().fillna(0).abs()

    daily_spread_change = -df["spread"].diff()  # spread shrinking = profit while in position
    df["pnl"] = df["position"].shift(1).fillna(0) * daily_spread_change * notional

    cost = df["position_change"] * (transaction_cost_bps / 10_000) * notional
    df["pnl"] = df["pnl"] - cost

    df["cumulative_pnl"] = df["pnl"].cumsum()

    return df


def summarize_backtest(backtest_df):
    """
    Prints basic performance stats. Extend with Sharpe, max drawdown,
    hit rate, etc, the same metrics you already computed for the GARCH
    project, once you have real P&L rather than the notional-scaled
    approximation.
    """
    total_pnl = backtest_df["pnl"].sum()
    num_trades = int(backtest_df["position_change"].sum() / 2)  # entry+exit = 2 flips
    win_days = (backtest_df["pnl"] > 0).sum()
    total_active_days = (backtest_df["position"] == 1).sum()

    print(f"Total PnL: {total_pnl:,.2f}")
    print(f"Number of round-trip trades: {num_trades}")
    print(f"Days in position: {total_active_days}")
    if total_active_days > 0:
        print(f"Win rate on active days: {win_days / total_active_days:.2%}")
