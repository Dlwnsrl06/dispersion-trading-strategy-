"""
End-to-end run of the pipeline for a single snapshot in time:
fetch options, extract IVs, back out implied correlation, compute
realized correlation, and print both.

This gives you one data point. The historical path builds the time
series that signal.py and backtest.py need from WRDS/IvyDB ATM straddles
and point-in-time quarterly weights.

Run this locally, it needs outbound internet access to Yahoo Finance,
which will not work in a sandboxed environment without it.
"""

from fetch_spy_weights import refresh_weights_if_stale
import config

# Rebuilds spy_weights.csv automatically if it's more than 20 hours old,
# using config.NUM_COMPONENTS so this stays in sync with any manual
# `python fetch_spy_weights.py --top ...` runs rather than silently
# falling back to a different default basket size.
refresh_weights_if_stale(max_age_hours=20, top_n=config.NUM_COMPONENTS)

from black_scholes import implied_volatility
from data_pipeline import get_basket_options, get_atm_option, get_price_history
from correlation_snapshot import compute_correlation_snapshot


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
    if index_iv is None:
        raise RuntimeError(
            f"IV solver failed for the index option itself ({config.INDEX_TICKER}). "
            f"Quote data: spot={index_option['spot']}, strike={index_option['strike']}, "
            f"mid_price={index_option['mid_price']}, days_to_expiry={index_option['days_to_expiry']}. "
            f"This usually means a stale or crossed quote, check these values before rerunning."
        )
    print(f"{config.INDEX_TICKER} implied vol: {index_iv:.4f}")

    print(f"\nFetching component options for {len(config.COMPONENT_TICKERS)} tickers...")
    component_options = get_basket_options(
        config.COMPONENT_TICKERS,
        config.MIN_DAYS_TO_EXPIRY,
        config.MAX_DAYS_TO_EXPIRY,
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

    active_tickers = [
        ticker for ticker in component_ivs if ticker in config.COMPONENT_WEIGHTS
    ]
    total_weight = sum(config.COMPONENT_WEIGHTS[ticker] for ticker in active_tickers)
    print(
        f"\nBasket coverage: {config.BASKET_COVERAGE:.1%} of index weight "
        f"across all {len(config.COMPONENT_TICKERS)} configured tickers, "
        f"{total_weight:.1%} after dropping tickers with failed IV solves. "
        f"Renormalizing to 100%."
    )

    print(f"\nFetching price history for realized correlation...")
    price_history = get_price_history(
        active_tickers, config.REALIZED_LOOKBACK_DAYS
    )

    print("\nSolving for implied and realized correlation...")
    snapshot = compute_correlation_snapshot(
        index_iv=index_iv,
        component_ivs=component_ivs,
        base_weights=config.COMPONENT_WEIGHTS,
        price_history=price_history,
    )

    implied_result = snapshot["implied"]
    realized_result = snapshot["realized"]

    print(f"Implied correlation: {implied_result['implied_correlation']:.4f}")
    if implied_result["out_of_bounds"]:
        print(
            "Warning: implied correlation is outside [-1, 1]. This usually means "
            "stale/mismatched quotes across tickers rather than a real signal, "
            "check the expiry alignment and bid-ask quality before trusting this number."
        )
    print(f"Realized correlation: {realized_result['realized_correlation']:.4f}")

    spread = implied_result["implied_correlation"] - realized_result["realized_correlation"]
    print(f"\nImplied minus realized correlation spread: {spread:.4f}")
    print(
        "A positive spread is the condition the dispersion trade is designed "
        "to monetize: sell index vol, buy the component basket."
    )


if __name__ == "__main__":
    main()
