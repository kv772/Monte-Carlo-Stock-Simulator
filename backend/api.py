from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.marketdata import get_price_data
from backend.montecarlo import portfolio_mean_vol, simulate_portfolio_paths
from backend.portfolio import risk_metrics
from backend.statistics import annualized_mean_return, compute_daily_log_returns, covariance_matrix

app = FastAPI(title="Monte Carlo Stock Simulator API")


class SimulationRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    weights: list[float] = Field(min_length=1)
    initial_investment: float = Field(gt=0)
    horizon_days: int = Field(default=252, ge=1)
    num_simulations: int = Field(default=10000, ge=100, le=100000)
    lookback_years: int = Field(default=5, ge=1, le=20)
    risk_free_rate: float = Field(default=0.02, ge=-1.0, le=1.0)
    sample_paths: int = Field(default=200, ge=10, le=1000)

    @model_validator(mode="after")
    def validate_lengths(self) -> "SimulationRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("weights length must match tickers length")
        return self


@app.post("/simulate")
def simulate(request: SimulationRequest) -> dict[str, object]:
    try:
        prices = get_price_data(request.tickers, request.lookback_years)
        log_returns = compute_daily_log_returns(prices)
        annual_mean = annualized_mean_return(log_returns)
        annual_cov = covariance_matrix(log_returns)
        mu, sigma = portfolio_mean_vol(annual_mean, annual_cov, np.array(request.weights))

        paths = simulate_portfolio_paths(
            mean_return=mu,
            volatility=sigma,
            initial_investment=request.initial_investment,
            num_days=request.horizon_days,
            num_simulations=request.num_simulations,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail="Simulation failed") from exc

    day_percentiles = np.percentile(paths, [5, 50, 95], axis=0)
    ending_values = paths[:, -1]
    hist_counts, hist_edges = np.histogram(ending_values, bins=20)

    rng = np.random.default_rng(42)
    sample_count = min(request.sample_paths, request.num_simulations)
    sampled_idx = rng.choice(request.num_simulations, size=sample_count, replace=False)

    return {
        "input": request.model_dump(),
        "fan_chart": {
            "p5": day_percentiles[0].tolist(),
            "p50": day_percentiles[1].tolist(),
            "p95": day_percentiles[2].tolist(),
        },
        "sample_paths": paths[sampled_idx].tolist(),
        "histogram": {"bin_edges": hist_edges.tolist(), "counts": hist_counts.tolist()},
        "risk_metrics": risk_metrics(
            paths,
            initial_investment=request.initial_investment,
            risk_free_rate=request.risk_free_rate,
            horizon_days=request.horizon_days,
        ),
    }

