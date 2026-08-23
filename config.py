"""
Configuration for the dispersion trading pipeline.

Edit INDEX_TICKER, COMPONENT_TICKERS, and COMPONENT_WEIGHTS to change
the basket. Weights should roughly reflect each component's index weight
and don't need to sum to exactly 1.0, but should be reasonably close.
"""

# The index whose options you're comparing against the basket.
# SPY is used instead of SPX because SPX options are cash-settled index
# options that are harder to source with free data. SPY is a liquid,
# easily accessible proxy.
INDEX_TICKER = "SPY"

# A subset of the index's largest constituents. Full replication (500
# names) isn't practical for a student project, so this uses the top
# names by weight, which is the same simplification real dispersion
# desks make when full replication is too costly or illiquid.
COMPONENT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "BRK-B", "AVGO", "LLY", "JPM",
]

# Approximate index weights for the tickers above (as of when this was
# written). These will drift over time, refresh them periodically from
# a live source (e.g. State Street's SPY holdings page) rather than
# trusting these numbers indefinitely.
COMPONENT_WEIGHTS = {
    "AAPL": 0.070,
    "MSFT": 0.065,
    "NVDA": 0.060,
    "AMZN": 0.038,
    "GOOGL": 0.020,
    "META": 0.024,
    "BRK-B": 0.017,
    "AVGO": 0.024,
    "LLY": 0.014,
    "JPM": 0.013,
}

# Risk-free rate used in Black-Scholes. Treated as constant for
# simplicity. A more careful version would pull a matching-maturity
# Treasury yield for each option's expiry.
RISK_FREE_RATE = 0.045

# How far out to look for expiries, in days. Options dispersion trades
# are typically done on monthly expiries around 30-60 days out.
MIN_DAYS_TO_EXPIRY = 25
MAX_DAYS_TO_EXPIRY = 45

# Lookback window (trading days) for realized correlation and realized
# volatility calculations.
REALIZED_LOOKBACK_DAYS = 30

# Rolling window (trading days) used when z-scoring the implied minus
# realized correlation spread to generate signals.
SIGNAL_ZSCORE_WINDOW = 60

# Entry/exit thresholds on the z-scored spread. These are placeholders,
# tune them once you have real historical output to look at.
ENTRY_ZSCORE = 1.0
EXIT_ZSCORE = 0.0
