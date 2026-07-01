import pytest

import quant_reporter.providers as prov
from quant_reporter.opt_core import get_risk_free_rate
from quant_reporter.providers import DEFAULT_RISK_FREE_RATE, RiskFreeRateUnavailable


def test_rfr_single_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    import yfinance as yf
    monkeypatch.setattr(yf, "download", boom)
    assert get_risk_free_rate() == DEFAULT_RISK_FREE_RATE
    assert DEFAULT_RISK_FREE_RATE == 0.02


def test_fetch_raises_but_get_swallows(monkeypatch):
    # GH #22: the explicit fetch surfaces the failure so callers can flag it,
    # while the swallowing wrapper keeps the -> float contract for everyone else.
    def boom(*a, **k):
        raise RuntimeError("network down")

    import yfinance as yf
    monkeypatch.setattr(yf, "download", boom)
    p = prov.YFinanceProvider()
    with pytest.raises(RiskFreeRateUnavailable):
        p.fetch_risk_free_rate()
    assert p.get_risk_free_rate() == DEFAULT_RISK_FREE_RATE


def test_fetch_wraps_parsing_errors_as_unavailable(monkeypatch):
    # A malformed response (no 'Close' column) must surface as
    # RiskFreeRateUnavailable, not a raw KeyError that bypasses the fallback
    # path and leaks to callers.
    import pandas as pd
    import yfinance as yf
    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame({"Open": [1.0, 2.0]}))
    p = prov.YFinanceProvider()
    with pytest.raises(RiskFreeRateUnavailable):
        p.fetch_risk_free_rate()
    assert p.get_risk_free_rate() == DEFAULT_RISK_FREE_RATE
