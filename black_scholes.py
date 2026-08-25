"""
Black-Scholes pricing and implied volatility extraction.

Important framing for your own understanding: Black-Scholes is used here
purely as a quoting convention to convert an observed market price into
a comparable volatility number. Nothing downstream assumes the world
actually follows Black-Scholes dynamics (constant vol, lognormal
returns, etc). This distinction is worth being able to state explicitly
in an interview.
"""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes price for a European option.

    S: spot price
    K: strike price
    T: time to expiry in years
    r: risk-free rate (annualized, continuously compounded)
    sigma: volatility (annualized)
    option_type: "call" or "put"
    """
    if T <= 0 or sigma <= 0:
        # Degenerate case, return intrinsic value to avoid division by zero.
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_vega(S, K, T, r, sigma):
    """
    Vega: sensitivity of option price to a change in volatility.
    Used as the derivative term in the Newton-Raphson IV solver.
    Same formula for calls and puts.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    initial_guess=0.3,
    tol=1e-6,
    max_iterations=100,
):
    """
    Solve for implied volatility given an observed market price.

    Uses Newton-Raphson first since it converges fast when vega isn't
    tiny. Falls back to bisection if Newton-Raphson fails to converge
    or vega collapses near zero (which happens for deep ITM/OTM options
    close to expiry, where price is barely sensitive to vol).

    Returns None if no solution is found in a reasonable vol range,
    which usually signals bad input data (stale quote, crossed market)
    rather than a real IV, so don't silently coerce this to a number
    upstream.
    """
    sigma = initial_guess

    for _ in range(max_iterations):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)

        if vega < 1e-8:
            break  # vega too small, Newton-Raphson step would be unstable

        diff = market_price - price
        if abs(diff) < tol:
            return sigma

        sigma = sigma + diff / vega

        if sigma <= 0 or sigma > 5:
            break  # stepped somewhere nonsensical, bail to bisection

    # Bisection just in case Newton-Raphson fails
    lo, hi = 1e-4, 5.0
    price_lo = bs_price(S, K, T, r, lo, option_type) - market_price
    price_hi = bs_price(S, K, T, r, hi, option_type) - market_price

    if price_lo * price_hi > 0:
        # Market price is outside what any vol in [lo, hi] can produce.
        # Almost always bad/stale data rather than a real edge case.
        return None

    for _ in range(200):
        mid = (lo + hi) / 2
        price_mid = bs_price(S, K, T, r, mid, option_type) - market_price

        if abs(price_mid) < tol:
            return mid
        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo = mid
            price_lo = price_mid

    return (lo + hi) / 2


if __name__ == "__main__":
    # Quick sanity check: price an option, then recover the vol we
    # priced it with by feeding the price back through the IV solver.
    S, K, T, r = 100, 100, 30 / 365, 0.045
    true_sigma = 0.22

    price = bs_price(S, K, T, r, true_sigma, "call")
    recovered_sigma = implied_volatility(price, S, K, T, r, "call")

    print(f"True sigma:      {true_sigma:.6f}")
    print(f"Recovered sigma: {recovered_sigma:.6f}")
    assert abs(recovered_sigma - true_sigma) < 1e-4, "IV solver failed round-trip check"
    print("Round-trip check passed.")
