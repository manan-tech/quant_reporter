# test/test_metrics_lib.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.metrics import (
    cagr, annual_volatility, sharpe, sortino, calmar, omega,
    max_drawdown, avg_drawdown, ulcer_index, value_at_risk, conditional_var,
    downside_deviation, tracking_error, information_ratio, hit_rate,
    win_loss_ratio, tail_ratio, skewness, kurtosis, summary_metrics,
)
from conftest import make_synthetic_prices


def _rets(seed=1, n=504):
    return make_synthetic_prices(seed=seed, n_days=n)["AAA"].pct_change().dropna()


def test_annual_vol_matches_formula():
    r = _rets()
    assert annual_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_sharpe_zero_rfr_matches_formula():
    r = _rets()
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, risk_free_rate=0.0) == pytest.approx(expected, rel=1e-9)


def test_cagr_constant_growth():
    r = pd.Series([0.01] * 252)
    assert cagr(r) == pytest.approx((1.01 ** 252) - 1, rel=1e-9)


def test_max_drawdown_non_positive():
    assert max_drawdown(_rets()) <= 0.0


def test_max_drawdown_known_path():
    # up then down: peak 2.0, trough 1.0 -> dd = -0.5
    assert max_drawdown(pd.Series([1.0, -0.5])) == pytest.approx(-0.5, rel=1e-9)
    # FIRST-period loss must be counted (regression test for the initial-peak bug)
    assert max_drawdown(pd.Series([-0.05, 0.021])) == pytest.approx(-0.05, rel=1e-9)


def test_calmar_sign():
    assert calmar(_rets()) == pytest.approx(cagr(_rets()) / abs(max_drawdown(_rets())), rel=1e-9)


def test_value_at_risk_positive_loss():
    assert value_at_risk(_rets(), 0.95) > 0


def test_cvar_ge_var():
    r = _rets()
    assert conditional_var(r, 0.95) >= value_at_risk(r, 0.95)


def test_hit_rate_bounds():
    assert 0.0 <= hit_rate(_rets()) <= 1.0


def test_omega_above_one_for_positive_drift():
    r = pd.Series([0.02, -0.01, 0.02, -0.01, 0.03])
    assert omega(r, 0.0) > 1.0


def test_tracking_error_zero_for_identical():
    r = _rets()
    assert tracking_error(r, r) == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_finite():
    r = _rets(seed=1)
    b = _rets(seed=2)
    assert np.isfinite(information_ratio(r, b))


def test_downside_deviation_le_total_vol():
    r = _rets()
    assert downside_deviation(r) <= annual_volatility(r) + 1e-9


def test_empty_returns_nan_not_raise():
    empty = pd.Series([], dtype=float)
    for fn in (cagr, annual_volatility, sharpe, sortino, max_drawdown,
               value_at_risk, hit_rate, ulcer_index):
        assert np.isnan(fn(empty)) or fn(empty) == 0.0


def test_summary_metrics_keys_no_benchmark():
    m = summary_metrics(_rets())
    for k in ("CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
              "Avg Drawdown", "Ulcer Index", "VaR (95%)", "CVaR (95%)", "Downside Dev",
              "Omega", "Hit Rate", "Win/Loss", "Tail Ratio", "Skew", "Kurtosis"):
        assert k in m
    assert "Tracking Error" not in m


def test_summary_metrics_adds_benchmark_keys():
    m = summary_metrics(_rets(seed=1), benchmark=_rets(seed=2))
    assert "Tracking Error" in m and "Information Ratio" in m


def test_ulcer_index_positive_and_zero_for_monotonic_gains():
    from quant_reporter.metrics import ulcer_index
    assert ulcer_index(pd.Series([0.01, 0.02, 0.01])) == pytest.approx(0.0, abs=1e-12)
    assert ulcer_index(_rets()) > 0


def test_avg_drawdown_non_positive():
    from quant_reporter.metrics import avg_drawdown
    assert avg_drawdown(_rets()) <= 0.0


def test_win_loss_ratio_known():
    from quant_reporter.metrics import win_loss_ratio
    # wins mean = 0.02, losses mean magnitude = 0.01 -> ratio 2.0
    assert win_loss_ratio(pd.Series([0.02, 0.02, -0.01, -0.01])) == pytest.approx(2.0, rel=1e-9)


def test_win_loss_ratio_nan_when_no_losses():
    from quant_reporter.metrics import win_loss_ratio
    assert np.isnan(win_loss_ratio(pd.Series([0.01, 0.02, 0.03])))


def test_tail_ratio_positive():
    from quant_reporter.metrics import tail_ratio
    assert tail_ratio(_rets()) > 0


def test_skew_kurtosis_finite():
    from quant_reporter.metrics import skewness, kurtosis
    r = _rets()
    assert np.isfinite(skewness(r)) and np.isfinite(kurtosis(r))
