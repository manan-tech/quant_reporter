# test/test_opt_core_sp2.py
"""Tests for SP2 Phase 4 additions: risk_contributions, optimize_risk_budget, portfolio_cvar."""
import numpy as np
import pandas as pd
import pytest

from quant_reporter.opt_core import risk_contributions, optimize_risk_budget, portfolio_cvar
from conftest import make_synthetic_prices


def _cov2(va=0.04, vb=0.09, rho=0.25):
    """2-asset annualized covariance matrix."""
    cov = rho * np.sqrt(va * vb)
    return pd.DataFrame([[va, cov], [cov, vb]], index=["A", "B"], columns=["A", "B"])


def _returns(n=500, seed=11):
    prices = make_synthetic_prices(seed=seed, n_days=n)
    return prices[["AAA", "BBB", "CCC"]].pct_change().dropna()


# ---------------------------------------------------------------------------
# risk_contributions
# ---------------------------------------------------------------------------

def test_rc_sums_to_one():
    cov = _cov2()
    rc = risk_contributions({"A": 0.5, "B": 0.5}, cov)
    assert rc.sum() == pytest.approx(1.0, abs=1e-10)


def test_rc_indexed_by_ticker():
    cov = _cov2()
    rc = risk_contributions({"A": 0.6, "B": 0.4}, cov)
    assert list(rc.index) == ["A", "B"]


def test_rc_concentrated_portfolio():
    cov = _cov2()
    # 100% in A → A contribution should be 1.0
    rc = risk_contributions({"A": 1.0, "B": 0.0}, cov)
    assert rc["A"] == pytest.approx(1.0, abs=1e-9)
    assert rc["B"] == pytest.approx(0.0, abs=1e-9)


def test_rc_zero_portfolio_vol():
    cov = pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=["A", "B"], columns=["A", "B"])
    rc = risk_contributions({"A": 0.5, "B": 0.5}, cov)
    assert (rc == 0.0).all()


def test_rc_array_weights():
    cov = _cov2()
    rc = risk_contributions([0.5, 0.5], cov)
    assert rc.sum() == pytest.approx(1.0, abs=1e-10)


def test_rc_higher_vol_asset_higher_contribution():
    cov = _cov2()  # B has higher vol (0.3 vs 0.2)
    rc = risk_contributions({"A": 0.5, "B": 0.5}, cov)
    assert rc["B"] > rc["A"]


def test_rc_requires_dataframe_for_dict():
    with pytest.raises(ValueError):
        risk_contributions({"A": 1.0}, np.eye(1))


# ---------------------------------------------------------------------------
# optimize_risk_budget (equal risk parity)
# ---------------------------------------------------------------------------

def test_rb_returns_expected_keys():
    cov = _cov2()
    result = optimize_risk_budget(cov)
    assert {"weights", "risk_contributions", "success", "message"} == set(result)


def test_rb_weights_sum_to_one():
    cov = _cov2()
    result = optimize_risk_budget(cov)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_rb_equal_risk_contributions():
    cov = _cov2()
    result = optimize_risk_budget(cov)
    rc = result["risk_contributions"]
    # Equal risk parity → contributions should be equal (0.5 each for 2 assets)
    assert rc.iloc[0] == pytest.approx(rc.iloc[1], abs=0.01)


def test_rb_inverse_vol_weights_for_diagonal_cov():
    # For a diagonal cov matrix, RP weights = 1/vol, normalized
    va, vb = 0.04, 0.09
    cov = pd.DataFrame([[va, 0.0], [0.0, vb]], index=["A", "B"], columns=["A", "B"])
    result = optimize_risk_budget(cov)
    expected_a = (1 / np.sqrt(va)) / (1 / np.sqrt(va) + 1 / np.sqrt(vb))
    assert result["weights"]["A"] == pytest.approx(expected_a, abs=0.005)


def test_rb_custom_budget():
    cov = _cov2()
    # 70/30 budget
    result = optimize_risk_budget(cov, budget={"A": 0.7, "B": 0.3})
    rc = result["risk_contributions"]
    assert rc["A"] == pytest.approx(0.7, abs=0.02)
    assert rc["B"] == pytest.approx(0.3, abs=0.02)


def test_rb_rejects_zero_budget():
    cov = _cov2()
    with pytest.raises(ValueError):
        optimize_risk_budget(cov, budget={"A": 0.0, "B": 0.0})


def test_rb_all_weights_nonnegative():
    cov = _cov2()
    result = optimize_risk_budget(cov)
    assert all(v >= 0 for v in result["weights"].values())


def test_rb_three_assets_equal_rc():
    prices = make_synthetic_prices(n_days=500)
    r = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    cov = pd.DataFrame(r.values.T @ r.values / len(r) * 252,
                       index=["AAA", "BBB", "CCC"], columns=["AAA", "BBB", "CCC"])
    result = optimize_risk_budget(cov)
    rc = result["risk_contributions"]
    assert rc.max() - rc.min() < 0.05  # contributions should be roughly equal


# ---------------------------------------------------------------------------
# portfolio_cvar
# ---------------------------------------------------------------------------

def test_cvar_positive():
    r = make_synthetic_prices(n_days=500)[["AAA"]].pct_change().dropna().iloc[:, 0]
    cvar = portfolio_cvar(r, confidence=0.95)
    assert cvar > 0


def test_cvar_nan_on_too_few_obs():
    assert np.isnan(portfolio_cvar(pd.Series([0.01])))


def test_cvar_higher_confidence_gives_larger_cvar():
    r = make_synthetic_prices(n_days=1000)[["AAA"]].pct_change().dropna().iloc[:, 0]
    cvar_95 = portfolio_cvar(r, confidence=0.95)
    cvar_99 = portfolio_cvar(r, confidence=0.99)
    assert cvar_99 >= cvar_95


def test_cvar_annualization():
    r = make_synthetic_prices(n_days=500)[["AAA"]].pct_change().dropna().iloc[:, 0]
    cvar_daily = portfolio_cvar(r, confidence=0.95, periods_per_year=1)
    cvar_ann = portfolio_cvar(r, confidence=0.95, periods_per_year=252)
    assert cvar_ann == pytest.approx(cvar_daily * np.sqrt(252), rel=1e-9)


def test_cvar_normal_tail_approx():
    # For N(0, σ) returns, daily CVaR at 95% ≈ σ * φ(1.645)/0.05 ≈ 2.063 σ
    rng = np.random.default_rng(42)
    sigma = 0.01
    r = pd.Series(rng.normal(0, sigma, 50_000))
    cvar_daily = portfolio_cvar(r, confidence=0.95, periods_per_year=1)
    expected = sigma * 2.063
    assert cvar_daily == pytest.approx(expected, rel=0.05)
