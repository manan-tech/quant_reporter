# test/test_recommend_cov_method.py
"""Offline tests for the covariance-estimator switch on recommend_weights (GH #6).

Wires Ledoit-Wolf shrinkage and eigenvalue-clipping denoising into the
recommendation path as an opt-in `cov_method=`. The default ("sample") preserves
the prior behavior exactly, so the change is additive/SemVer-safe.
"""
import pytest

from quant_reporter.recommendation import recommend_weights, walk_forward_recommendation
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


def _valid_weights(w):
    return sum(w.values()) == pytest.approx(1.0, abs=1e-6) and all(v >= -1e-9 for v in w.values())


def test_default_cov_method_is_sample_and_preserves_behavior():
    prices = _prices()
    default = recommend_weights(prices)
    sample = recommend_weights(prices, cov_method="sample")
    assert default.evidence["cov_method"] == "sample"
    # The sample path carries no estimator diagnostics.
    assert "cov_info" not in default.evidence
    # Default must equal the explicit sample path weight-for-weight (no behavior change).
    for k in default.weights:
        assert default.weights[k] == pytest.approx(sample.weights[k], abs=1e-9)


def test_ledoit_wolf_path_valid_and_records_shrinkage():
    rec = recommend_weights(_prices(), cov_method="ledoit_wolf")
    assert _valid_weights(rec.weights)
    assert rec.evidence["cov_method"] == "ledoit_wolf"
    shrink = rec.evidence["cov_info"]["shrinkage"]
    assert 0.0 <= shrink <= 1.0


def test_denoise_path_valid_and_records_method():
    rec = recommend_weights(_prices(), cov_method="denoise", n_components=2)
    assert _valid_weights(rec.weights)
    assert rec.evidence["cov_method"] == "denoise"
    assert rec.evidence["cov_info"]["n_components"] == 2


def test_invalid_cov_method_raises():
    with pytest.raises(ValueError, match="bogus") as ei:
        recommend_weights(_prices(), cov_method="bogus")
    # Error names the valid options so the caller can self-correct.
    assert "ledoit_wolf" in str(ei.value)


def test_walk_forward_accepts_cov_method():
    v = walk_forward_recommendation(_prices(), cov_method="ledoit_wolf")
    assert v.n_windows >= 1
