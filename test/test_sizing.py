# test/test_sizing.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.sizing import (
    forecast_portfolio_vol,
    target_volatility_scalar,
    inverse_volatility_weights,
    realized_tracking_error,
    kelly_fraction,
    cppi_weights,
)
from conftest import make_synthetic_prices


def _returns(n=252, seed=7):
    prices = make_synthetic_prices(seed=seed, n_days=n)
    return prices[["AAA", "BBB", "CCC"]].pct_change().dropna()


# ---------------------------------------------------------------------------
# forecast_portfolio_vol
# ---------------------------------------------------------------------------

def test_fvol_dict_weights_matches_formula():
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    weights = {"A": 0.5, "B": 0.5}
    expected = np.sqrt(0.5 ** 2 * 0.04 + 2 * 0.5 * 0.5 * 0.01 + 0.5 ** 2 * 0.09)
    assert forecast_portfolio_vol(weights, cov) == pytest.approx(expected, rel=1e-9)


def test_fvol_array_weights():
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    assert forecast_portfolio_vol([1.0, 0.0], cov) == pytest.approx(0.2, rel=1e-9)


def test_fvol_zero_weights_gives_zero():
    cov = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=["A", "B"], columns=["A", "B"])
    assert forecast_portfolio_vol({"A": 0.0, "B": 0.0}, cov) == pytest.approx(0.0)


def test_fvol_requires_dataframe_for_dict():
    with pytest.raises(ValueError):
        forecast_portfolio_vol({"A": 1.0}, np.eye(1))


# ---------------------------------------------------------------------------
# target_volatility_scalar
# ---------------------------------------------------------------------------

def test_tvs_positive():
    r = _returns().iloc[:, 0]
    assert target_volatility_scalar(r, target_vol=0.10) > 0.0


def test_tvs_clipped_to_5():
    r = pd.Series([1e-10] * 100)
    assert target_volatility_scalar(r, target_vol=0.10, lookback=20) <= 5.0


def test_tvs_fallback_on_empty():
    assert target_volatility_scalar(pd.Series([], dtype=float)) == 1.0


def test_tvs_fallback_on_single_obs():
    assert target_volatility_scalar(pd.Series([0.01])) == 1.0


def test_tvs_higher_target_gives_larger_scalar():
    r = _returns().iloc[:, 0]
    s1 = target_volatility_scalar(r, target_vol=0.10)
    s2 = target_volatility_scalar(r, target_vol=0.20)
    assert s2 > s1


def test_tvs_monotone_in_target():
    r = _returns().iloc[:, 0]
    targets = [0.05, 0.10, 0.15, 0.20]
    scalars = [target_volatility_scalar(r, t) for t in targets]
    assert scalars == sorted(scalars)


# ---------------------------------------------------------------------------
# inverse_volatility_weights
# ---------------------------------------------------------------------------

def test_ivw_sums_to_one():
    w = inverse_volatility_weights(_returns())
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)


def test_ivw_low_vol_gets_higher_weight():
    from quant_reporter.signals import compute_trailing_volatility
    r = _returns()
    vols = compute_trailing_volatility(r).iloc[-1]
    w = inverse_volatility_weights(r)
    assert w[vols.idxmin()] > w[vols.idxmax()]


def test_ivw_all_keys_present():
    r = _returns()
    w = inverse_volatility_weights(r)
    assert set(w.keys()) == set(r.columns)


def test_ivw_all_weights_positive():
    r = _returns()
    w = inverse_volatility_weights(r)
    assert all(v > 0 for v in w.values())


# ---------------------------------------------------------------------------
# realized_tracking_error
# ---------------------------------------------------------------------------

def test_rte_zero_for_identical():
    r = pd.Series([0.01, -0.02, 0.005] * 30)
    assert realized_tracking_error(r, r) == pytest.approx(0.0, abs=1e-12)


def test_rte_positive():
    r = _returns()
    assert realized_tracking_error(r["AAA"], r["BBB"]) > 0


def test_rte_annualization():
    r = _returns()
    active = (r["AAA"] - r["BBB"]).dropna()
    expected = float(active.std(ddof=1) * np.sqrt(252))
    assert realized_tracking_error(r["AAA"], r["BBB"], 252) == pytest.approx(expected, rel=1e-9)


def test_rte_nan_on_too_few():
    assert np.isnan(realized_tracking_error(pd.Series([0.01]), pd.Series([0.02])))


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

def test_kelly_positive_edge():
    assert kelly_fraction(0.001, 0.0001) == pytest.approx(10.0)


def test_kelly_zero_variance_returns_zero():
    assert kelly_fraction(0.001, 0.0) == 0.0


def test_kelly_negative_edge_is_negative():
    assert kelly_fraction(-0.001, 0.0001) < 0.0


# ---------------------------------------------------------------------------
# cppi_weights
# ---------------------------------------------------------------------------

def test_cppi_full_cushion_all_risky():
    r = cppi_weights(floor_value=0.0, multiplier=4.0, portfolio_value=1.0)
    assert r["risky"] == pytest.approx(1.0)
    assert r["safe"] == pytest.approx(0.0)


def test_cppi_sums_to_one():
    for pv in [0.8, 1.0, 1.2]:
        r = cppi_weights(floor_value=0.7, multiplier=3.0, portfolio_value=pv)
        assert r["risky"] + r["safe"] == pytest.approx(1.0, abs=1e-12)


def test_cppi_at_floor_all_safe():
    r = cppi_weights(floor_value=1.0, multiplier=4.0, portfolio_value=1.0)
    assert r["risky"] == pytest.approx(0.0)
    assert r["safe"] == pytest.approx(1.0)


def test_cppi_below_floor_all_safe():
    r = cppi_weights(floor_value=1.0, multiplier=4.0, portfolio_value=0.9)
    assert r["risky"] == pytest.approx(0.0)


def test_cppi_risky_never_exceeds_one():
    r = cppi_weights(floor_value=0.0, multiplier=100.0, portfolio_value=1.0)
    assert r["risky"] <= 1.0


def test_cppi_zero_portfolio_value_all_safe():
    r = cppi_weights(floor_value=0.5, multiplier=4.0, portfolio_value=0.0)
    assert r["risky"] == pytest.approx(0.0)
    assert r["safe"] == pytest.approx(1.0)
