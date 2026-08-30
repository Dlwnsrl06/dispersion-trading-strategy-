# Dispersion Trading Pipeline

An index-vs-component correlation dispersion strategy. 
Sells index implied volatility and buys a basket of component
implied volatility, betting that realized correlation across the basket
comes in lower than what's implied. This strategy is based off of the paper: 
Driessen, J., Maenhout, P. J., & Vilkov, G. (2009). 
"The Price of Correlation Risk: Evidence from Equity Options." Journal of Finance, 64(3), 1377-1406.


## Structure

- `data`: Folder of component stocks and their weights
- `config.py`: index/basket tickers, weights, and strategy parameters
- `black_scholes.py`: Black-Scholes pricer and Newton-Raphson (with
  bisection fallback) implied volatility solver
- `data_pipeline.py`: fetches index and component options chains and
  price history via yfinance
- `correlation.py`: solves for implied correlation from index and
  component IVs, and computes the matching realized correlation from
  historical returns
  `build_basket_universe.py`: Outputs cumulative list of components that were the top 150 weights of SPY from 2015-2025
- `signal.py`: z-scores the implied-minus-realized correlation spread
  and generates a binary in/out position signal
- `backtest.py`: a simplified variance-notional P&L approximation, see
  the module docstring for what this does and does not capture
- `main.py`: runs the full pipeline for a single live snapshot

## Setup

```
pip install -r requirements.txt
python main.py
```

Requires outbound internet access to Yahoo Finance. `main.py` fetches
live data, it does not include historical options data, see below.

Requires to update the config.py's ticker information:
1. update the list of tickers
2. update the list of weights for tickers in spy_weights.csv


## What this gives you, and what it doesn't

Running `main.py` gets you one live data point: today's implied
correlation vs. today's realized correlation. That's enough to sanity
check that the pipeline works end to end.

It does not give you a backtest. For that you need implied correlation
computed at many points in history, which means historical options
chain data, not just the live snapshot yfinance provides by default.
This is the first real design decision to make as a team: find a
historical options data source (several exist, most free ones are
limited in depth or history) and adapt `data_pipeline.py` to pull from
it. -> waiting for WRDS access

## Known simplifications, and where to go deeper

- **Single average correlation assumption.** The math assumes one
  correlation number applies uniformly across every pair of components.
  This is standard practice in dispersion trading and in index
  providers' own implied correlation indices.
- **ATM options only.** Real dispersion desks think about the whole
  skew, not just one strike. Extending to a skew-aware version (e.g.
  variance swap replication using a strip of strikes) is a legitimate
  "further work" extension once the ATM version works.
- **Backtest is a notional-scaled approximation**, not real option
  position P&L. See the docstring in `backtest.py` for the honest scope
  of what it captures. Building out real position-level tracking (entry
  Greeks, rolling positions at expiry, assignment handling) is where
  most of the remaining project's difficulty lives.
- **Constant risk-free rate.** Fine for now, a more careful version
  pulls a maturity-matched Treasury yield per expiry.
- **Basket of 10 names, not full replication.** Same simplification
  real dispersion desks make when full index replication isn't
  practical. Worth stating explicitly rather than treating it as a
  shortcut you took to save time.

## Suggested build order

1. Get `main.py` running end to end locally with live data.
2. Sanity check the implied correlation number against public sources
   (CBOE publishes an implied correlation index you can compare
   against directionally, not exactly, since methodology differs).
3. Source historical options data and extend the pipeline to compute
   implied correlation across a real date range.
4. Build out the signal and backtest against that historical series.
5. Only after 1 to 4 work: consider extending the backtest toward real
   position-level P&L tracking if you have time left.
