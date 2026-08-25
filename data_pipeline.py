"""
Fetches options chain data for the index and each component, and finds
the near-ATM contract at a matching expiry for each ticker.

This is the part most likely to give you trouble in practice: illiquid
components will have wide bid-ask spreads or missing strikes, and
expiries won't always line up perfectly across every ticker. The
functions below raise clear errors rather than silently returning bad
data, so you notice problems instead of debugging weird backtest
results three steps downstream.
"""


"""
given a ticker (from yfinance), hand back the four numbers Black-Scholes needs:
    spot, strike, time to expiry, and a market price.

operate during US market hours.
"""

#this is the original data_pipelin.py file using yfinacne api

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import time

import config


def get_expiry_in_range(ticker_obj, min_days, max_days):
    """
    Picks the first available expiry that falls within [min_days, max_days]
    from today. Returns None if nothing qualifies.
    """
    today = datetime.now().date()
    for expiry_str in ticker_obj.options:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        days_out = (expiry_date - today).days
        if min_days <= days_out <= max_days:
            return expiry_str, days_out
    return None, None


def get_atm_option(ticker_symbol, min_days, max_days, option_type="call", target_expiry=None):
    """
    Fetches the near-the-money option for a single ticker at a matching
    expiry. Returns a dict with the fields needed for IV extraction, or
    raises a ValueError if no suitable contract is found.
    """
    ticker_obj = yf.Ticker(ticker_symbol)

    spot_history = ticker_obj.history(period="1d")
    if spot_history.empty:
        raise ValueError(f"No spot price data for {ticker_symbol}")
    spot = spot_history["Close"].iloc[-1]

    if target_expiry is not None:
        if target_expiry not in ticker_obj.options:
            raise ValueError(f"{ticker_symbol} has no contract expiring {target_expiry}")
        expiry_str = target_expiry
        days_out = (datetime.strptime(expiry_str, "%Y-%m-%d").date()
                    - datetime.now().date()).days
    else:
        expiry_str, days_out = get_expiry_in_range(ticker_obj, min_days, max_days)
        if expiry_str is None:
            raise ValueError(
                f"No expiry between {min_days} and {max_days} days found for {ticker_symbol}"
            )

    chain = ticker_obj.option_chain(expiry_str)
    options_df = chain.calls if option_type == "call" else chain.puts

    if options_df.empty:
        raise ValueError(f"Empty {option_type} chain for {ticker_symbol} at {expiry_str}")

    options_df = options_df.copy()
    options_df["strike_diff"] = (options_df["strike"] - spot).abs()
    candidates = options_df.sort_values("strike_diff")

    atm_row = None
    used_last_price = False
    for _, row in candidates.iterrows():
        if row["bid"] > 0 and row["ask"] > 0 and row["ask"] >= row["bid"]:
            atm_row = row
            break

    if atm_row is None:
        for _, row in candidates.iterrows():
            if row.get("lastPrice", 0) > 0:
                atm_row = row
                used_last_price = True
                break

    if atm_row is None:
        raise ValueError(
            f"No usable quote found for {ticker_symbol} at {expiry_str}, "
            f"checked {len(candidates)} strikes, all had zero, missing "
            f"or crossed bid/ask."
        )

    if used_last_price:
        mid_price = atm_row["lastPrice"]
    else:
        mid_price = (atm_row["bid"] + atm_row["ask"]) / 2


    return {
        "ticker": ticker_symbol,
        "spot": spot,
        "strike": atm_row["strike"],
        "expiry": expiry_str,
        "days_to_expiry": days_out,
        "time_to_expiry_years": days_out / 365,
        "mid_price": mid_price,
        "bid": atm_row["bid"],
        "ask": atm_row["ask"],
    }

def get_basket_options(tickers, min_days, max_days, option_type="call", target_expiry=None):
    """
    Fetches ATM options for a list of tickers. Skips (with a printed
    warning) any ticker that fails, rather than crashing the whole
    pipeline over one bad quote.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_atm_option(
                ticker, min_days, max_days, option_type, target_expiry=target_expiry
            )
        except ValueError as e:
            print(f"Warning: skipping {ticker}: {e}")
        time.sleep(0.3)
    return results

def get_price_history(tickers, lookback_days):
    """
    Fetches daily close prices for realized volatility and correlation
    calculations. Pulls extra calendar days to comfortably cover the
    requested number of trading days after weekends/holidays.
    """
    calendar_days = int(lookback_days * 1.6) + 10
    start = (datetime.now() - timedelta(days=calendar_days)).strftime("%Y-%m-%d")

    data = yf.download(tickers, start=start, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

    return data.tail(lookback_days)

if __name__ == "__main__":
    # Quick manual check. Requires internet access to Yahoo Finance,
    # which will not work inside a sandboxed environment without
    # outbound access, run this locally to verify.
    print(f"Fetching ATM call for {config.INDEX_TICKER}...")
    index_option = get_atm_option(
        config.INDEX_TICKER, config.MIN_DAYS_TO_EXPIRY, config.MAX_DAYS_TO_EXPIRY
    )
    print(index_option)

    from datetime import datetime
    symbols = [config.INDEX_TICKER] + config.COMPONENT_TICKERS
    today = datetime.now().date()
    for s in symbols:
        import yfinance as yf
        opts = yf.Ticker(s).options
        hits = [e for e in opts
                if 25 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 45]
        print(f"{s:6s} {hits}")

