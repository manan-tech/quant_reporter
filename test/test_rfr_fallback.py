import quant_reporter.providers as prov
from quant_reporter.opt_core import get_risk_free_rate
from quant_reporter.providers import DEFAULT_RISK_FREE_RATE


def test_rfr_single_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    import yfinance as yf
    monkeypatch.setattr(yf, "download", boom)
    assert get_risk_free_rate() == DEFAULT_RISK_FREE_RATE
    assert DEFAULT_RISK_FREE_RATE == 0.02
