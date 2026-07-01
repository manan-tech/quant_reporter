# test/test_recommend_method.py
"""Offline tests for the no-forecast allocation switch on recommend_weights (GH #7).

The dominant source of optimizer error is the expected-return estimate. `method=`
lets the caller pick an allocator that does not forecast returns (min_variance /
risk_parity / max_diversification) — the honest "we don't forecast returns"
stance. The default ("optimize") preserves the prior objective-based behavior.
"""
import pytest

from quant_reporter.recommendation import recommend_weights
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


def _valid_weights(w):
    return sum(w.values()) == pytest.approx(1.0, abs=1e-6) and all(v >= -1e-9 for v in w.values())


def test_default_method_is_optimize_and_preserves_behavior():
    prices = _prices()
    default = recommend_weights(prices)
    explicit = recommend_weights(prices, method="optimize")
    assert default.evidence["method"] == "optimize"
    assert default.evidence["uses_return_forecast"] is True
    for k in default.weights:
        assert default.weights[k] == pytest.approx(explicit.weights[k], abs=1e-9)


@pytest.mark.parametrize("method", ["min_variance", "risk_parity", "max_diversification"])
def test_no_forecast_methods_valid_and_flagged(method):
    rec = recommend_weights(_prices(), method=method)
    assert _valid_weights(rec.weights)
    assert rec.evidence["method"] == method
    assert rec.evidence["uses_return_forecast"] is False


def test_no_forecast_rationale_states_the_assumption():
    rec = recommend_weights(_prices(), method="risk_parity")
    # The rationale must plainly state that returns are not forecast.
    assert "no-forecast" in rec.rationale.lower() or "not forecast" in rec.rationale.lower()


def test_no_forecast_composes_with_robust_covariance():
    # method= and cov_method= are orthogonal — a no-forecast allocator can still
    # consume a robust covariance.
    rec = recommend_weights(_prices(), method="min_variance", cov_method="ledoit_wolf")
    assert _valid_weights(rec.weights)
    assert rec.evidence["method"] == "min_variance"
    assert rec.evidence["cov_method"] == "ledoit_wolf"


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="bogus") as ei:
        recommend_weights(_prices(), method="bogus")
    assert "risk_parity" in str(ei.value)
