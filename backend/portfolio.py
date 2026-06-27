from __future__ import annotations

import numpy as np


def ending_value_percentiles(paths: np.ndarray, percentiles: tuple[int, int, int] = (5, 50, 95)) -> dict[str, float]:
    """Return selected percentiles of ending portfolio values."""
    ending = paths[:, -1]
    p_values = np.percentile(ending, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, p_values)}


def probability_of_loss(paths: np.ndarray, initial_investment: float) -> float:
    """Return probability that ending value is below initial investment."""
    ending = paths[:, -1]
    return float(np.mean(ending < initial_investment))


def value_at_risk_95(paths: np.ndarray, initial_investment: float) -> float:
    """Return one-period 95% VaR as a currency amount."""
    ending = paths[:, -1]
    p5 = np.percentile(ending, 5)
    return float(max(initial_investment - p5, 0.0))


def sharpe_ratio(
    paths: np.ndarray,
    initial_investment: float,
    risk_free_rate: float = 0.02,
    horizon_days: int | None = None,
    trading_days: int = 252,
) -> float:
    """Estimate Sharpe ratio from simulated ending values using annualized returns."""
    ending = paths[:, -1]
    period_returns = ending / initial_investment - 1.0
    if horizon_days is None:
        horizon_days = paths.shape[1]
    years = max(horizon_days / trading_days, 1e-9)
    annualized_returns = np.power(1.0 + period_returns, 1.0 / years) - 1.0
    volatility = np.std(annualized_returns, ddof=1)
    if np.isclose(volatility, 0.0):
        return 0.0
    return float((np.mean(annualized_returns) - risk_free_rate) / volatility)


def risk_metrics(
    paths: np.ndarray,
    initial_investment: float,
    risk_free_rate: float = 0.02,
    horizon_days: int | None = None,
) -> dict[str, float]:
    """Compute key risk metrics from simulation paths."""
    pct = ending_value_percentiles(paths)
    expected_return = float(np.mean(paths[:, -1]) / initial_investment - 1.0)
    return {
        **pct,
        "probability_of_loss": probability_of_loss(paths, initial_investment),
        "value_at_risk_95": value_at_risk_95(paths, initial_investment),
        "expected_return": expected_return,
        "sharpe_ratio": sharpe_ratio(paths, initial_investment, risk_free_rate, horizon_days),
    }

