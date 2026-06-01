"""Tests for the DataProvider abstraction and its wiring through build_context.

These run fully offline: a MockProvider supplies synthetic prices and a fixed
risk-free rate, proving the library never has to touch yfinance/Yahoo when a
custom backend is supplied.
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import quant_reporter as qr
from quant_reporter.providers import (
    DataProvider,
    YFinanceProvider,
    get_default_provider,
    set_default_provider,
    DEFAULT_RISK_FREE_RATE,
)
from quant_reporter.black_litterman import get_market_caps


class MockProvider:
    """A deterministic, network-free DataProvider for tests."""

    def __init__(self, rfr=0.037):
        self._rfr = rfr
        self._idx = pd.bdate_range("2020-01-01", "2023-01-01")
        self._rng = np.random.default_rng(42)
        self.calls = []

    def get_prices(self, tickers, start, end):
        self.calls.append(("get_prices", tuple(tickers)))
        data = {}
        for t in tickers:
            steps = self._rng.normal(0.0005, 0.01, len(self._idx))
            data[t] = 100 * np.exp(np.cumsum(steps))
        return pd.DataFrame(data, index=self._idx)

    def get_risk_free_rate(self):
        self.calls.append(("get_risk_free_rate", None))
        return self._rfr


@pytest.fixture
def restore_default_provider():
    """Snapshot and restore the global provider so tests don't contaminate each other."""
    original = get_default_provider()
    try:
        yield
    finally:
        set_default_provider(original)


def test_protocol_is_runtime_checkable():
    assert isinstance(YFinanceProvider(), DataProvider)
    assert isinstance(MockProvider(), DataProvider)


def test_object_missing_methods_is_not_a_provider():
    class NotAProvider:
        pass

    assert not isinstance(NotAProvider(), DataProvider)


def test_default_provider_is_yfinance():
    assert isinstance(get_default_provider(), YFinanceProvider)


def test_set_and_get_default_provider(restore_default_provider):
    mp = MockProvider()
    set_default_provider(mp)
    assert get_default_provider() is mp


def test_build_context_uses_injected_provider_for_prices_and_rfr():
    mp = MockProvider(rfr=0.041)
    ctx = qr.build_context(
        {"AAPL": 0.6, "MSFT": 0.4}, "SPY",
        "2020-06-01", "2022-06-01",
        risk_free_rate="auto", data_provider=mp,
    )
    assert ctx.risk_free_rate == 0.041
    assert ctx.data_provider is mp
    assert ("get_risk_free_rate", None) in mp.calls
    fetched = [c[1] for c in mp.calls if c[0] == "get_prices"][0]
    assert set(fetched) == {"AAPL", "MSFT", "SPY"}


def test_build_context_uses_global_default_provider(restore_default_provider):
    mp = MockProvider(rfr=0.029)
    set_default_provider(mp)
    ctx = qr.build_context(
        {"AAPL": 0.5, "MSFT": 0.5}, "SPY",
        "2020-06-01", "2022-06-01", risk_free_rate="auto",
    )
    assert ctx.data_provider is mp
    assert ctx.risk_free_rate == 0.029


def test_numeric_rfr_does_not_call_provider():
    mp = MockProvider()
    ctx = qr.build_context(
        {"AAPL": 0.5, "MSFT": 0.5}, "SPY",
        "2020-06-01", "2022-06-01",
        risk_free_rate=0.05, data_provider=mp,
    )
    assert ctx.risk_free_rate == 0.05
    assert ("get_risk_free_rate", None) not in mp.calls


def test_report_generator_threads_provider_offline():
    mp = MockProvider()
    out = os.path.join(tempfile.mkdtemp(), "p.html")
    qr.create_portfolio_report(
        {"AAPL": 0.5, "MSFT": 0.5}, "SPY",
        "2020-06-01", "2022-06-01",
        filename=out, data_provider=mp,
    )
    assert os.path.exists(out)
    assert any(c[0] == "get_prices" for c in mp.calls)


def test_build_context_from_prices_records_provider():
    mp = MockProvider()
    prices = mp.get_prices(["AAPL", "MSFT", "SPY"], None, None)
    ctx = qr.build_context_from_prices(
        prices, {"AAPL": 0.5, "MSFT": 0.5}, "SPY",
        "2020-06-01", "2022-06-01",
        risk_free_rate=0.03, data_provider=mp,
    )
    assert ctx.data_provider is mp


def test_get_market_caps_uses_provider_capability():
    class CapProvider(MockProvider):
        def get_market_caps(self, tickers):
            return {t: 1_000.0 * (i + 1) for i, t in enumerate(tickers)}

    caps = get_market_caps(["AAPL", "MSFT"], provider=CapProvider())
    assert caps["AAPL"] == 1000.0
    assert caps["MSFT"] == 2000.0


def test_get_risk_free_rate_routes_through_provider():
    from quant_reporter.opt_core import get_risk_free_rate

    mp = MockProvider(rfr=0.055)
    assert get_risk_free_rate(provider=mp) == 0.055


def test_yfinance_provider_rfr_falls_back_on_failure(monkeypatch):
    import yfinance as yf

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(yf, "download", boom)
    assert YFinanceProvider().get_risk_free_rate() == DEFAULT_RISK_FREE_RATE


def test_yfinance_provider_get_prices_returns_none_on_failure(monkeypatch):
    import yfinance as yf

    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(yf, "download", boom)
    assert YFinanceProvider().get_prices(["AAPL"], "2020-01-01", "2020-02-01") is None
