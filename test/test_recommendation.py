# test/test_recommendation.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend_weights, RecommendedWeights
from quant_reporter.recommendation import rebalance_trades, Trade, RebalancePlan
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


def test_rebalance_trades_basic_deltas():
    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.3, "BBB": 0.7})
    assert isinstance(plan, RebalancePlan)
    byt = {o.ticker: o for o in plan.orders}
    assert byt["AAA"].side == "sell" and byt["AAA"].delta == pytest.approx(-0.2)
    assert byt["BBB"].side == "buy" and byt["BBB"].delta == pytest.approx(0.2)
    assert plan.turnover == pytest.approx(0.2)  # one-way: 0.5*(0.2+0.2)


def test_rebalance_trades_union_missing_as_zero():
    plan = rebalance_trades({"AAA": 1.0}, {"AAA": 0.5, "BBB": 0.5})
    byt = {o.ticker: o for o in plan.orders}
    assert byt["BBB"].current_weight == 0.0 and byt["BBB"].side == "buy"
    assert byt["BBB"].delta == pytest.approx(0.5)


def test_rebalance_threshold_band_holds_small_trades():
    plan = rebalance_trades({"AAA": 0.50, "BBB": 0.50},
                            {"AAA": 0.49, "BBB": 0.51}, threshold=0.05)
    assert plan.orders == []
    assert set(plan.held) == {"AAA", "BBB"}


def test_rebalance_cost_uses_default_model():
    # executed deltas AAA -0.5, BBB +0.5 -> two-way notional 1.0;
    # default model: 1bps commission + 2.5bps half-spread -> cost_frac 3.5bps
    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.0, "BBB": 1.0})
    assert plan.est_cost == pytest.approx((1.0 / 1e4) + (2.5 / 1e4), rel=1e-9)


def test_rebalance_no_trades_when_identical():
    w = {"AAA": 0.5, "BBB": 0.5}
    plan = rebalance_trades(w, w)
    assert plan.orders == [] and plan.turnover == pytest.approx(0.0)
    assert plan.est_cost == pytest.approx(0.0)
