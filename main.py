"""
End-to-end run of the pipeline for a single snapshot in time:
fetch options, extract IVs, back out implied correlation, compute
realized correlation, and print both.

This gives you one data point. To get the time series you need for
signal.py and backtest.py, you'll need to run this (or the underlying
functions) across many historical dates, which means sourcing historical
options chains rather than just the live chain yfinance gives you by
default. That's the next real piece of work once this runs cleanly:
decide on a historical options data source (some free, most paid) and
adapt data_pipeline.py to pull from it instead of live snapshots.

Run this locally, it needs outbound internet access to Yahoo Finance,
which will not work in a sandboxed environment without it.
"""

import config
from black_scholes import implied_volatility
from data_pipeline import get_basket_options, get_atm_option, get_price_history
from correlation import implied_correlation, realized_correlation


def main():
    print(f"Fetching index option for {config.INDEX_TICKER}...")
    index_option = get_atm_option(
        config.INDEX_TICKER, config.MIN_DAYS_TO_EXPIRY, config.MAX_DAYS_TO_EXPIRY
    )
    index_iv = implied_volatility(
        market_price=index_option["mid_price"],
        S=index_option["spot"],
        K=index_option["strike"],
        T=index_option["time_to_expiry_years"],
        r=config.RISK_FREE_RATE,
    )
    print(f"{config.INDEX_TICKER} implied vol: {index_iv:.4f}")

    print(f"\nFetching component options for {len(config.COMPONENT_TICKERS)} tickers...")
    component_options = get_basket_options(
        config.COMPONENT_TICKERS, config.MIN_DAYS_TO_EXPIRY, config.MAX_DAYS_TO_EXPIRY
    )

    component_ivs = {}
    for ticker, option_data in component_options.items():
        iv = implied_volatility(
            market_price=option_data["mid_price"],
            S=option_data["spot"],
            K=option_data["strike"],
            T=option_data["time_to_expiry_years"],
            r=config.RISK_FREE_RATE,
        )
        if iv is not None:
            component_ivs[ticker] = iv
            print(f"  {ticker}: IV = {iv:.4f}")
        else:
            print(f"  {ticker}: IV solver failed, dropping from basket")

    # Only keep weights for tickers that actually produced a valid IV.
    active_weights = {t: config.COMPONENT_WEIGHTS[t] for t in component_ivs}

    print("\nSolving for implied correlation...")
    implied_result = implied_correlation(index_iv, component_ivs, active_weights)
    print(f"Implied correlation: {implied_result['implied_correlation']:.4f}")
    if implied_result["out_of_bounds"]:
        print(
            "Warning: implied correlation is outside [-1, 1]. This usually means "
            "stale/mismatched quotes across tickers rather than a real signal, "
            "check the expiry alignment and bid-ask quality before trusting this number."
        )

    print("\nFetching price history for realized correlation...")
    price_history = get_price_history(
        list(active_weights.keys()), config.REALIZED_LOOKBACK_DAYS
    )
    realized_result = realized_correlation(price_history, active_weights)
    print(f"Realized correlation: {realized_result['realized_correlation']:.4f}")

    spread = implied_result["implied_correlation"] - realized_result["realized_correlation"]
    print(f"\nImplied minus realized correlation spread: {spread:.4f}")
    print(
        "A positive spread is the condition the dispersion trade is designed "
        "to monetize: sell index vol, buy the component basket."
    )


if __name__ == "__main__":
    main()
