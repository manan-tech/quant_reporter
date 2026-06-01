"""
DataProvider protocol — all network data fetches go through this interface.

This lets users swap yfinance for Bloomberg, Refinitiv, a local CSV, or a mock
in tests without touching any report or optimizer code.

Usage
-----
Default (yfinance)::

    ctx = build_context(portfolio, benchmark, train_start, train_end)

Custom provider::

    from quant_reporter.providers import DataProvider
    import pandas as pd

    class MyProvider:
        def get_prices(self, tickers, start, end) -> pd.DataFrame: ...
        def get_risk_free_rate(self) -> float: ...

    ctx = build_context(portfolio, benchmark, train_start, train_end,
                        data_provider=MyProvider())
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RISK_FREE_RATE = 0.02


@runtime_checkable
class DataProvider(Protocol):
    """Minimal interface any data backend must satisfy."""

    def get_prices(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices, shape (dates × tickers)."""
        ...

    def get_risk_free_rate(self) -> float:
        """Return the annualised risk-free rate as a decimal (e.g. 0.045)."""
        ...


class YFinanceProvider:
    """Default implementation — wraps yfinance.

    yfinance is an unofficial Yahoo Finance scraper with no SLA. It is
    the default because it requires no API key and covers most use-cases,
    but it can break silently when Yahoo restructures its responses. Swap
    it out with a custom DataProvider if you need reliability guarantees.
    """

    def get_prices(self, tickers: List[str], start: str, end: str) -> Optional[pd.DataFrame]:
        import yfinance as yf

        logger.info("YFinanceProvider: fetching %s", ", ".join(tickers))
        try:
            raw = yf.download(tickers, start=start, end=end, threads=False)
            if raw.empty:
                raise ValueError("No data returned. Check tickers and date range.")
            if "Close" not in raw:
                raise ValueError("Downloaded data does not contain 'Close' prices.")
            data = raw["Close"]
            if isinstance(data, pd.Series):
                data = data.to_frame(name=tickers[0])
            data = data.ffill().bfill().dropna()
            return data
        except Exception as exc:
            logger.error("YFinanceProvider.get_prices failed: %s", exc)
            return None

    def get_risk_free_rate(self) -> float:
        import yfinance as yf

        try:
            tbill = yf.download("^IRX", period="5d", threads=False)
            if tbill.empty:
                return DEFAULT_RISK_FREE_RATE
            close = tbill["Close"]
            if hasattr(close, "iloc"):
                val = close.dropna().iloc[-1]
                if hasattr(val, "item"):
                    val = val.item()
                return float(val) / 100.0
        except Exception as exc:
            logger.warning("YFinanceProvider.get_risk_free_rate failed: %s — using %.2f", exc, DEFAULT_RISK_FREE_RATE)
        return DEFAULT_RISK_FREE_RATE


_default_provider: DataProvider = YFinanceProvider()


def get_default_provider() -> DataProvider:
    return _default_provider


def set_default_provider(provider: DataProvider) -> None:
    """Replace the global default provider (useful for testing or air-gapped environments)."""
    global _default_provider
    _default_provider = provider
