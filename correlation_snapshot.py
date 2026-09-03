"""
Shared one-date correlation calculation used by both the live snapshot
and the historical time-series builder.
"""

from correlation import implied_correlation, realized_correlation


def normalize_active_weights(component_ivs, base_weights):
    """
    Keep only tickers with both a usable IV and a base weight, then
    renormalize weights to sum to 1.
    """
    active_weights = {
        ticker: base_weights[ticker]
        for ticker in component_ivs
        if ticker in base_weights and component_ivs[ticker] is not None
    }

    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        raise ValueError("No positive component weights available after filtering.")

    normalized_weights = {
        ticker: weight / total_weight
        for ticker, weight in active_weights.items()
    }

    active_ivs = {
        ticker: component_ivs[ticker]
        for ticker in normalized_weights
    }

    return active_ivs, normalized_weights, total_weight


def compute_correlation_snapshot(
    index_iv,
    component_ivs,
    base_weights,
    price_history=None,
):
    """
    Compute implied correlation for one date, and realized correlation
    too when a price history window is supplied.

    index_iv: index implied volatility for the date.
    component_ivs: dict of {ticker: implied_vol}.
    base_weights: dict of {ticker: point-in-time or live basket weight}.
    price_history: optional close-price DataFrame with ticker columns.
    """
    active_ivs, active_weights, raw_weight_coverage = normalize_active_weights(
        component_ivs, base_weights
    )

    implied_result = implied_correlation(index_iv, active_ivs, active_weights)

    result = {
        "implied": implied_result,
        "realized": None,
        "component_ivs": active_ivs,
        "weights": active_weights,
        "raw_weight_coverage": raw_weight_coverage,
        "num_components": len(active_ivs),
    }

    if price_history is not None:
        result["realized"] = realized_correlation(price_history, active_weights)

    return result
