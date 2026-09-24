from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_mean_vol(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Blend asset-level annualized moments into portfolio-level annualized moments."""
    if len(mean_returns) != len(weights):
        raise ValueError("weights length must match number of assets")

    w = np.asarray(weights, dtype=float)
    if np.isclose(w.sum(), 100.0):
        w = w / 100.0
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must sum to 1.0 or 100.0")

    portfolio_mu = float(np.dot(w, mean_returns.to_numpy()))
    portfolio_variance = float(w.T @ covariance.to_numpy() @ w)
    portfolio_sigma = float(np.sqrt(max(portfolio_variance, 0.0)))
    return portfolio_mu, portfolio_sigma


def simulate_portfolio_paths(
    mean_return: float,
    volatility: float,
    initial_investment: float,
    num_days: int,
    num_simulations: int,
    trading_days: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    """Simulate uncorrelated daily-compounded portfolio paths with NumPy vectorization."""
    if initial_investment <= 0:
        raise ValueError("initial_investment must be positive")
    if num_days < 1 or num_simulations < 1:
        raise ValueError("num_days and num_simulations must be positive")

    rng = np.random.default_rng(random_seed)
    daily_mu = mean_return / trading_days
    daily_sigma = volatility / np.sqrt(trading_days)
    daily_returns = rng.normal(daily_mu, daily_sigma, size=(num_simulations, num_days))
    daily_returns = np.clip(daily_returns, -0.999999, None)
    growth = np.cumprod(1.0 + daily_returns, axis=1)
    return initial_investment * growth

