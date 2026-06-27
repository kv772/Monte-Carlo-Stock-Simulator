from __future__ import annotations

import numpy as np
import pandas as pd


def compute_daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a DataFrame of closing prices."""
    if prices.empty:
        raise ValueError("prices must not be empty")
    return np.log(prices / prices.shift(1)).dropna(how="any")


def annualized_mean_return(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Compute annualized mean return from daily log returns."""
    if log_returns.empty:
        raise ValueError("log_returns must not be empty")
    return log_returns.mean() * trading_days


def annualized_volatility(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """Compute annualized volatility from daily log returns."""
    if log_returns.empty:
        raise ValueError("log_returns must not be empty")
    return log_returns.std(ddof=1) * np.sqrt(trading_days)


def covariance_matrix(log_returns: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """Compute annualized covariance matrix from daily log returns."""
    if log_returns.empty:
        raise ValueError("log_returns must not be empty")
    return log_returns.cov() * trading_days

