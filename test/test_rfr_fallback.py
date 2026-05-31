import quant_reporter.opt_core as oc
from quant_reporter.opt_core import DEFAULT_RISK_FREE_RATE, get_risk_free_rate


def test_rfr_single_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(oc.yf, "download", boom)
    assert get_risk_free_rate() == DEFAULT_RISK_FREE_RATE
    assert DEFAULT_RISK_FREE_RATE == 0.02
