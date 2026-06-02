# test/test_recommendation.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend_weights, RecommendedWeights
from quant_reporter.recommendation import rebalance_trades, Trade, RebalancePlan
from quant_reporter.recommendation import risk_alerts, RiskAlert
from quant_reporter.recommendation import compare_verdict, StrategyVerdict
from quant_reporter.recommendation import recommend, Recommendation
from quant_reporter.objectives import neg_sharpe
from quant_reporter.opt_core import get_optimization_inputs, get_portfolio_stats
from quant_reporter.strategy import backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from quant_reporter.performance_stats import compare_strategies_oos
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


_EQ = {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
_OFF = dict(vol_target=99, max_drawdown_limit=99, max_weight=0.99, max_risk_contribution=0.99)


def test_no_alerts_when_within_all_limits():
    assert risk_alerts(_EQ, _prices(), **_OFF) == []


def test_concentration_breach_fires():
    alerts = risk_alerts({"AAA": 0.8, "BBB": 0.1, "CCC": 0.1}, _prices(),
                         vol_target=99, max_drawdown_limit=99,
                         max_weight=0.40, max_risk_contribution=0.99)
    conc = [a for a in alerts if a.evidence.get("metric") == "max_weight"]
    assert conc and conc[0].kind == "concentration" and conc[0].severity == "breach"
    assert conc[0].evidence["asset"] == "AAA"
    assert conc[0].evidence["value"] == pytest.approx(0.8)


def test_concentration_warning_in_band():
    # max weight 0.38, cap 0.40, warn band 0.36 -> warning
    alerts = risk_alerts({"AAA": 0.38, "BBB": 0.32, "CCC": 0.30}, _prices(),
                         vol_target=99, max_drawdown_limit=99,
                         max_weight=0.40, max_risk_contribution=0.99)
    conc = [a for a in alerts if a.evidence.get("metric") == "max_weight"]
    assert conc and conc[0].severity == "warning"


def test_vol_breach_fires_with_tiny_target():
    alerts = risk_alerts(_EQ, _prices(), vol_target=1e-6, max_drawdown_limit=99,
                         max_weight=0.99, max_risk_contribution=0.99)
    vb = [a for a in alerts if a.kind == "vol_breach"]
    assert vb and vb[0].severity == "breach"
    assert vb[0].evidence["value"] > vb[0].evidence["threshold"]


def test_drawdown_breach_fires_with_tiny_limit():
    alerts = risk_alerts(_EQ, _prices(), max_drawdown_limit=1e-6, vol_target=99,
                         max_weight=0.99, max_risk_contribution=0.99)
    assert any(a.kind == "drawdown_breach" for a in alerts)


def test_sector_cap_breach():
    alerts = risk_alerts({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, _prices(),
                         sector_map={"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"},
                         sector_caps={"Tech": 0.5}, **_OFF)
    sect = [a for a in alerts if a.kind == "sector_cap"]
    assert sect and sect[0].evidence["sector"] == "Tech"
    assert sect[0].evidence["value"] == pytest.approx(0.8)


def test_factor_drift_silent_without_inputs():
    assert not any(a.kind == "factor_drift" for a in risk_alerts(_EQ, _prices(), **_OFF))


def test_factor_drift_fires_with_tiny_limit():
    rng = np.random.default_rng(0)
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(_prices()) - 1, 2)),
                       index=_prices().index[1:], columns=["MKT", "SMB"])
    alerts = risk_alerts(_EQ, _prices(), factor_returns=fac, factor_loading_limit=1e-9, **_OFF)
    assert any(a.kind == "factor_drift" for a in alerts)


def _results(n=700, seed=3):
    prices = make_synthetic_prices(seed=seed, n_days=n)   # AAA/BBB/CCC + BMK
    return backtest_many({"EW": equal_weight, "RP": risk_parity}, prices, benchmark="BMK")


def test_compare_verdict_picks_best_by_dsr():
    results = _results()
    v = compare_verdict(results)
    assert isinstance(v, StrategyVerdict)
    cmp = compare_strategies_oos({k: r.returns for k, r in results.items()}, n_trials=2)
    assert v.winner == cmp["best_by_dsr"]
    assert v.ranking[0]["name"] == v.winner


def test_compare_verdict_evidence_has_summary():
    v = compare_verdict(_results())
    assert set(v.evidence["summary"]) == {"EW", "RP"}
    assert v.evidence["select_by"] == "dsr"
    assert v.evidence["n_trials"] == 2


def test_compare_verdict_single_result_no_comparison():
    prices = make_synthetic_prices(n_days=700)
    results = backtest_many({"EW": equal_weight}, prices)
    v = compare_verdict(results)
    assert v.winner == "EW"
    assert "no comparison" in v.rationale.lower()


def test_compare_verdict_empty():
    v = compare_verdict({})
    assert v.winner is None and v.ranking == []


def test_recommend_bundle_full():
    prices = _prices(n=700)
    results = backtest_many({"EW": equal_weight, "RP": risk_parity},
                            make_synthetic_prices(n_days=700), benchmark="BMK")
    rec = recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                    results=results, **_OFF)
    assert isinstance(rec, Recommendation)
    assert isinstance(rec.target_weights, RecommendedWeights)
    assert isinstance(rec.trades, RebalancePlan)
    assert isinstance(rec.alerts, list)
    assert isinstance(rec.verdict, StrategyVerdict)


def test_recommend_no_current_weights_no_trades_no_verdict():
    rec = recommend(_prices(n=700), **_OFF)
    assert rec.trades is None
    assert rec.verdict is None


def test_recommend_alerts_target_when_no_current():
    # no current_weights -> alerts run on the recommended target; tiny name cap
    # almost surely trips a concentration alert
    rec = recommend(_prices(n=700), vol_target=99, max_drawdown_limit=99,
                    max_weight=1e-6, max_risk_contribution=0.99)
    assert any(a.kind == "concentration" for a in rec.alerts)


def test_recommendation_to_dict_keys():
    rec = recommend(_prices(n=700), current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, **_OFF)
    d = rec.to_dict()
    assert set(d) == {"target_weights", "trades", "alerts", "verdict", "suitability"}
    assert d["trades"]["orders"] and isinstance(d["alerts"], list)
    assert d["verdict"] is None


def test_recommendation_to_text_is_string():
    rec = recommend(_prices(n=700), current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, **_OFF)
    txt = rec.to_text()
    assert isinstance(txt, str) and "RECOMMENDATION" in txt


# --- Coverage hardening (deferred per-task review items) ---

def test_rebalance_custom_cost_model_is_used():
    seen = {}

    def fake_cost(trades):
        seen["n"] = len(trades)
        return {"cost_frac": 0.0123}

    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.0, "BBB": 1.0},
                            cost_model=fake_cost)
    assert plan.est_cost == pytest.approx(0.0123)
    assert seen["n"] == 2  # both executed deltas were passed to the injected model


def test_rebalance_threshold_boundary_executes():
    # |delta| exactly == threshold is NOT inside the band (strict <) -> executes.
    # 0.25/0.5/0.75 are float-exact, so the boundary is unambiguous.
    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5},
                            {"AAA": 0.25, "BBB": 0.75}, threshold=0.25)
    assert {o.ticker for o in plan.orders} == {"AAA", "BBB"}
    assert plan.held == []


def test_drawdown_breach_severity_and_evidence():
    alerts = risk_alerts(_EQ, _prices(), max_drawdown_limit=1e-6, vol_target=99,
                         max_weight=0.99, max_risk_contribution=0.99)
    dd = [a for a in alerts if a.kind == "drawdown_breach"]
    assert dd and dd[0].severity == "breach"
    assert dd[0].evidence["metric"] == "max_drawdown"
    assert dd[0].evidence["value"] < 0  # max_dd is a negative decimal


def test_risk_contribution_concentration_fires():
    # tiny risk-contribution cap, generous name cap -> only the rc sub-check fires
    alerts = risk_alerts({"AAA": 0.8, "BBB": 0.1, "CCC": 0.1}, _prices(),
                         vol_target=99, max_drawdown_limit=99,
                         max_weight=0.99, max_risk_contribution=1e-6)
    rc = [a for a in alerts if a.evidence.get("metric") == "risk_contribution"]
    assert rc and rc[0].kind == "concentration" and rc[0].severity == "breach"
    assert "asset" in rc[0].evidence


def test_recommendation_to_text_includes_sections():
    rec = recommend(_prices(n=700), current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, **_OFF)
    txt = rec.to_text()
    assert "Rebalance:" in txt and "Risk alerts:" in txt


def test_risk_alerts_unknown_ticker_raises_valueerror():
    with pytest.raises(ValueError, match="not in prices"):
        risk_alerts({"AAA": 0.5, "ZZZ": 0.5}, _prices(), **_OFF)
