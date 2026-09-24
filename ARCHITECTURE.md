# Architecture (MVP)

## Backend vertical slices

1. **Data layer** (`backend/marketdata.py`)
   - `get_price_data(tickers, lookback_years)` via `yfinance`
   - simple CSV cache under `.cache/`

2. **Statistics layer** (`backend/statistics.py`)
   - daily log returns
   - annualized mean return
   - annualized volatility
   - annualized covariance matrix

3. **Monte Carlo engine** (`backend/montecarlo.py`)
   - vectorized NumPy simulation (`num_simulations x num_days`)
   - daily compounding (`value *= 1 + r`)

4. **Risk metrics** (`backend/portfolio.py`)
   - ending value percentiles (5/50/95)
   - probability of loss
   - VaR (95%)
   - Sharpe ratio and expected return

5. **API layer** (`backend/api.py`)
   - `POST /simulate`
   - returns fan-chart bands (daily p5/p50/p95), sampled paths, ending-value histogram, and risk metrics
