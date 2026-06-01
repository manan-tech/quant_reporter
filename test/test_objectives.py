# test/test_objectives.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.objectives import (
    neg_sharpe, variance, cvar_objective, tracking_error_objective,
    mean_squared_error, mean_absolute_error,
)


def test_neg_sharpe_matches_manual():
    w = np.array([0.5, 0.5])
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    port_ret = 0.5 * 0.10 + 0.5 * 0.20
    port_vol = np.sqrt(0.5 ** 2 * 0.04 + 0.5 ** 2 * 0.09)
    assert neg_sharpe(w, mu, cov, 0.02) == pytest.approx(-(port_ret - 0.02) / port_vol, rel=1e-9)


def test_variance_matches_quadratic_form():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert variance(w, None, cov) == pytest.approx(float(w @ cov @ w), rel=1e-12)


def test_cvar_objective_positive_for_losses():
    rng = np.random.default_rng(0)
    R = rng.normal(-0.001, 0.01, (500, 2))
    assert cvar_objective(np.array([0.5, 0.5]), R, 0.95) > 0


def test_tracking_error_objective_zero_when_matched():
    R = np.array([[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01]])
    bench = np.array([0.01, 0.02, -0.01])
    assert tracking_error_objective(np.array([1.0, 0.0]), R, bench) == pytest.approx(0.0, abs=1e-12)


def test_mse_and_mae_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0
    assert mean_absolute_error(y, y) == 0.0


def test_mse_known_value():
    assert mean_squared_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx((1 + 9) / 2)


def test_neg_sharpe_drop_in_for_find_optimal_portfolio():
    import scipy.optimize as sco
    mu = np.array([0.10, 0.20]); cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    res = sco.minimize(neg_sharpe, np.array([0.5, 0.5]), args=(mu, cov, 0.02),
                       method="SLSQP", bounds=[(0, 1)] * 2,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    assert res.success and 0 <= res.x[0] <= 1
