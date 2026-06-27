from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"


def _cache_path(tickers: Iterable[str], lookback_years: int) -> Path:
    normalized = "-".join(sorted(t.upper() for t in tickers))
    return _CACHE_DIR / f"prices_{normalized}_{lookback_years}y.csv"


def get_price_data(tickers: list[str], lookback_years: int) -> pd.DataFrame:
    """Fetch historical adjusted-close prices for tickers with simple file caching."""
    if not tickers:
        raise ValueError("tickers must not be empty")
    if lookback_years < 1:
        raise ValueError("lookback_years must be at least 1")

    normalized_tickers = [ticker.upper().strip() for ticker in tickers]
    cache_file = _cache_path(normalized_tickers, lookback_years)
    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        expected = set(normalized_tickers)
        if expected.issubset(set(cached.columns)):
            return cached[normalized_tickers].sort_index()

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw = yf.download(
        normalized_tickers,
        period=f"{lookback_years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No price data returned for requested tickers.")

    if "Close" in raw.columns:
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=normalized_tickers[0])

    prices = prices.dropna(how="all").sort_index()
    missing = [ticker for ticker in normalized_tickers if ticker not in prices.columns]
    if missing:
        raise ValueError(f"Invalid or unavailable ticker(s): {', '.join(missing)}")

    prices = prices[normalized_tickers].dropna()
    if prices.empty:
        raise ValueError("Price data is empty after cleaning.")

    prices.to_csv(cache_file)
    return prices

