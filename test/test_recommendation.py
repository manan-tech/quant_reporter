# test/test_recommendation.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend_weights, RecommendedWeights
from quant_reporter.objectives import neg_sharpe
from quant_reporter.opt_core import get_optimization_inputs, get_portfolio_stats
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


def test_recommend_weights_sums_to_one_and_keys():
    rec = recommend_weights(_prices())
    assert isinstance(rec, RecommendedWeights)
    assert sum(rec.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(rec.weights) == {"AAA", "BBB", "CCC"}
    assert all(v >= -1e-9 for v in rec.weights.values())


def test_recommend_weights_evidence_matches_portfolio_stats():
    prices = _prices()
    rec = recommend_weights(prices, objective=neg_sharpe, risk_free_rate=0.02)
    mean, cov, _ = get_optimization_inputs(prices)
    w = np.array([rec.weights[c] for c in cov.columns])
    pr, pv, sh = get_portfolio_stats(w, mean, cov, 0.02)
    assert rec.evidence["sharpe"] == pytest.approx(sh, rel=1e-9)
    assert rec.evidence["expected_vol"] == pytest.approx(pv, rel=1e-9)
    assert rec.evidence["expected_return"] == pytest.approx(pr, rel=1e-9)
    assert rec.objective == "neg_sharpe"


def test_recommend_weights_rationale_nonempty():
    assert recommend_weights(_prices()).rationale
