# Monte-Carlo-Stock-Simulator

Backend for a Monte Carlo portfolio simulator:

- `backend/marketdata.py` — historical close price fetch with caching and ticker validation.
- `backend/statistics.py` — daily log returns + annualized mean/volatility/covariance.
- `backend/montecarlo.py` — vectorized NumPy Monte Carlo path simulation.
- `backend/portfolio.py` — risk metrics (percentiles, VaR, Sharpe, probability of loss).
- `backend/api.py` — FastAPI `/simulate` endpoint returning fan-chart bands, sampled paths, histogram, and risk metrics.

## Local sanity checks

```bash
python -m unittest discover -s tests -p "test_*.py"
```