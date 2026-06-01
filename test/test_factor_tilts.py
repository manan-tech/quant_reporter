# test/test_factor_tilts.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.factor_tilts import (
    characteristic_tilt_weights,
    factor_neutralize_returns,
    resample_portfolio,
)
from conftest import make_synthetic_prices


def _pw():
    prices = make_synthetic_prices()
    weights = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    return prices, weights


# ---------------------------------------------------------------------------
# characteristic_tilt_weights
# ---------------------------------------------------------------------------

def test_ctw_sums_to_one():
    base = {"A": 0.5, "B": 0.3, "C": 0.2}
    scores = {"A": 1.0, "B": 0.5, "C": 2.0}
    result = characteristic_tilt_weights(base, scores, tilt_strength=0.3)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-10)


def test_ctw_zero_tilt_is_unchanged():
    base = {"A": 0.5, "B": 0.3, "C": 0.2}
    scores = {"A": 1.0, "B": 0.5, "C": 2.0}
    result = characteristic_tilt_weights(base, scores, tilt_strength=0.0)
    for k in base:
        assert result[k] == pytest.approx(base[k], abs=1e-10)


def test_ctw_high_score_asset_gets_more_weight():
    base = {"A": 0.5, "B": 0.5}
    scores = {"A": 3.0, "B": 1.0}
    result = characteristic_tilt_weights(base, scores, tilt_strength=0.5)
    assert result["A"] > result["B"]


def test_ctw_all_keys_present():
    base = {"A": 0.5, "B": 0.3, "C": 0.2}
    scores = {"A": 1.0}  # B and C get default 0
    result = characteristic_tilt_weights(base, scores, tilt_strength=0.2)
    assert set(result.keys()) == {"A", "B", "C"}


def test_ctw_all_weights_nonnegative():
    base = {"A": 0.5, "B": 0.3, "C": 0.2}
    scores = {"A": 10.0, "B": 0.0, "C": 0.0}
    result = characteristic_tilt_weights(base, scores, tilt_strength=1.0)
    assert all(v >= 0 for v in result.values())


def test_ctw_no_positive_scores_returns_base():
    base = {"A": 0.6, "B": 0.4}
    scores = {"A": -1.0, "B": 0.0}
    result = characteristic_tilt_weights(base, scores, tilt_strength=0.5)
    assert result["A"] == pytest.approx(0.6, abs=1e-9)
    assert result["B"] == pytest.approx(0.4, abs=1e-9)


def test_ctw_rejects_invalid_tilt():
    with pytest.raises(ValueError):
        characteristic_tilt_weights({"A": 1.0}, {"A": 1.0}, tilt_strength=1.5)


def test_ctw_rejects_zero_base():
    with pytest.raises(ValueError):
        characteristic_tilt_weights({"A": 0.0, "B": 0.0}, {"A": 1.0})


# ---------------------------------------------------------------------------
# factor_neutralize_returns
# ---------------------------------------------------------------------------

def test_fnr_shape_matches_returns():
    prices = make_synthetic_prices()
    returns = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    factors = pd.DataFrame({"F1": returns["AAA"] + 0.001}, index=returns.index)
    resid = factor_neutralize_returns(returns, factors)
    assert resid.shape == returns.shape


def test_fnr_residuals_have_lower_correlation_with_factor():
    prices = make_synthetic_prices()
    returns = prices[["AAA", "BBB"]].pct_change().dropna()
    # Use AAA as a factor — should reduce AAA's self-correlation after neutralization
    factors = pd.DataFrame({"F1": returns["AAA"]}, index=returns.index)
    resid = factor_neutralize_returns(returns, factors)
    original_corr = abs(returns["BBB"].corr(returns["AAA"]))
    resid_corr = abs(resid["BBB"].dropna().corr(returns["AAA"]))
    assert resid_corr < original_corr


def test_fnr_single_factor_self_residual_near_zero_corr():
    """An asset regressed on itself → residuals uncorrelated with the factor."""
    prices = make_synthetic_prices()
    returns = prices[["AAA"]].pct_change().dropna()
    factors = pd.DataFrame({"F1": returns["AAA"]}, index=returns.index)
    resid = factor_neutralize_returns(returns, factors)
    corr = abs(resid["AAA"].dropna().corr(returns["AAA"]))
    assert corr < 0.01


def test_fnr_insufficient_data_gives_nan():
    prices = make_synthetic_prices(n_days=5)
    returns = prices[["AAA", "BBB"]].pct_change().dropna()
    # 4 observations, 5 factors → cannot regress
    factors = pd.DataFrame({f"F{i}": returns["AAA"] * i for i in range(1, 6)},
                           index=returns.index)
    resid = factor_neutralize_returns(returns, factors)
    assert resid.isna().all().all()


# ---------------------------------------------------------------------------
# resample_portfolio
# ---------------------------------------------------------------------------

def test_rp_sums_to_one():
    prices, weights = _pw()
    result = resample_portfolio(prices[["AAA", "BBB", "CCC"]], weights, n_samples=10, seed=1)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


def test_rp_all_tickers_present():
    prices, weights = _pw()
    result = resample_portfolio(prices[["AAA", "BBB", "CCC"]], weights, n_samples=10, seed=1)
    assert set(result.keys()) == {"AAA", "BBB", "CCC"}


def test_rp_all_weights_nonnegative():
    prices, weights = _pw()
    result = resample_portfolio(prices[["AAA", "BBB", "CCC"]], weights, n_samples=10, seed=1)
    assert all(v >= 0 for v in result.values())


def test_rp_reproducible_with_same_seed():
    prices, weights = _pw()
    r1 = resample_portfolio(prices[["AAA", "BBB", "CCC"]], weights, n_samples=10, seed=7)
    r2 = resample_portfolio(prices[["AAA", "BBB", "CCC"]], weights, n_samples=10, seed=7)
    for k in r1:
        assert r1[k] == pytest.approx(r2[k])
