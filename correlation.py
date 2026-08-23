"""
Core dispersion math: backing out a single "average implied correlation"
from index and component implied vols, and computing the matching
realized correlation from historical returns.

The simplifying assumption used here (a single correlation applied
uniformly across every pair of components) is standard in dispersion
trading and in index providers' own implied correlation indices. Know
why it's used: solving for every pairwise correlation individually from
just the index IV and component IVs is underdetermined, you don't have
enough equations for that many unknowns. Collapsing to one average
correlation makes the problem solvable with the information options
prices actually give you.
"""

import numpy as np
import pandas as pd


def implied_correlation(index_iv, component_ivs, weights):
    """
    Solves for the single average pairwise correlation implied by the
    relationship:

        Var(index) = sum_i w_i^2 * sigma_i^2
                     + rho * [ (sum_i w_i * sigma_i)^2 - sum_i w_i^2 * sigma_i^2 ]

    which comes directly from expanding Var(sum_i w_i * r_i) and
    assuming every pairwise correlation equals the same rho.

    index_iv: index implied volatility (annualized, decimal, e.g. 0.15)
    component_ivs: dict of {ticker: implied_vol}
    weights: dict of {ticker: weight}, same keys as component_ivs

    Returns the implied correlation, along with the two intermediate
    sums so you can sanity-check the calculation.
    """
    tickers = list(component_ivs.keys())
    w = np.array([weights[t] for t in tickers])
    sigma = np.array([component_ivs[t] for t in tickers])

    weighted_var_sum = np.sum((w * sigma) ** 2)          # sum_i w_i^2 sigma_i^2
    weighted_sum_sq = np.sum(w * sigma) ** 2              # (sum_i w_i sigma_i)^2

    numerator = index_iv ** 2 - weighted_var_sum
    denominator = weighted_sum_sq - weighted_var_sum

    if abs(denominator) < 1e-12:
        raise ValueError(
            "Denominator too close to zero, check that weights and vols "
            "aren't degenerate (e.g. only one component)."
        )

    rho = numerator / denominator

    return {
        "implied_correlation": rho,
        "weighted_var_sum": weighted_var_sum,
        "weighted_sum_sq": weighted_sum_sq,
        "out_of_bounds": not (-1.0 <= rho <= 1.0),
    }


def realized_correlation(price_history, weights):
    """
    Computes the same "average implied correlation" quantity, but from
    realized daily returns instead of options prices. This is what you
    compare implied correlation against to generate a signal.

    price_history: DataFrame of daily close prices, columns are tickers
                   (must include all tickers in `weights`)
    weights: dict of {ticker: weight}

    Uses the same algebraic identity as implied_correlation, but with
    realized (annualized) volatilities and the realized index variance
    computed directly from the weighted return series, rather than
    assuming a single rho and solving backwards. This keeps the
    comparison apples-to-apples with the implied side.
    """
    tickers = list(weights.keys())
    returns = price_history[tickers].pct_change().dropna()

    w = np.array([weights[t] for t in tickers])

    # Annualized realized vol per component.
    component_realized_vol = returns.std() * np.sqrt(252)
    sigma = component_realized_vol[tickers].values

    # Realized "index" returns as the weighted sum of component returns.
    # This is an approximation of the real index, using only the basket,
    # not a substitute for pulling the index's own realized vol.
    weighted_index_returns = (returns[tickers] * w).sum(axis=1)
    realized_index_var = weighted_index_returns.var() * 252

    weighted_var_sum = np.sum((w * sigma) ** 2)
    weighted_sum_sq = np.sum(w * sigma) ** 2

    denominator = weighted_sum_sq - weighted_var_sum
    if abs(denominator) < 1e-12:
        raise ValueError("Denominator too close to zero in realized correlation calc.")

    rho = (realized_index_var - weighted_var_sum) / denominator

    return {
        "realized_correlation": rho,
        "realized_index_var": realized_index_var,
        "component_realized_vol": component_realized_vol[tickers].to_dict(),
    }


if __name__ == "__main__":
    # Sanity check with synthetic data where we control the true
    # correlation, to confirm the formula recovers it correctly.
    np.random.seed(0)
    true_rho = 0.4
    n_days = 500
    n_assets = 5

    # Build correlated returns via a simple factor model:
    # each asset return = sqrt(rho)*common_factor + sqrt(1-rho)*idiosyncratic
    common_factor = np.random.normal(0, 0.01, n_days)
    idio = np.random.normal(0, 0.01, (n_days, n_assets))
    asset_returns = np.sqrt(true_rho) * common_factor[:, None] + np.sqrt(1 - true_rho) * idio

    tickers = [f"A{i}" for i in range(n_assets)]
    prices = 100 * np.cumprod(1 + asset_returns, axis=0)
    price_df = pd.DataFrame(prices, columns=tickers)

    weights = {t: 1 / n_assets for t in tickers}

    result = realized_correlation(price_df, weights)
    print(f"True correlation:      {true_rho:.4f}")
    print(f"Recovered correlation: {result['realized_correlation']:.4f}")
