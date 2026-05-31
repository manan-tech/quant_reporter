import numpy as np
import pandas as pd
import pytest
from quant_reporter.metrics import compute_drawdown, calculate_max_drawdown, DrawdownResult


def test_compute_drawdown_scalar_equals_curve_min():
    cum = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6])
    dd = compute_drawdown(cum)
    assert isinstance(dd, DrawdownResult)
    assert dd.max_dd == dd.curve.min()
    assert dd.max_dd == pytest.approx((0.6 - 1.2) / 1.2)  # -0.5 from the 1.2 peak


def test_calculate_max_drawdown_backcompat_scalar():
    cum = pd.Series([1.0, 1.2, 0.9])
    assert calculate_max_drawdown(cum) == pytest.approx((0.9 - 1.2) / 1.2)


from quant_reporter.analytics import portfolio_returns, ReturnsBundle
from quant_reporter.opt_core import get_portfolio_price


def test_portfolio_returns_buy_and_hold_matches_closed_form(synthetic_prices):
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    rb = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq=None)
    closed = get_portfolio_price(synthetic_prices[["AAA", "BBB", "CCC"]], w)
    assert isinstance(rb, ReturnsBundle)
    assert rb.growth["Portfolio"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    # iterative buy&hold == closed-form buy&hold
    assert rb.growth["Portfolio"].iloc[-1] == pytest.approx(closed.iloc[-1], rel=1e-6)
    assert rb.terminal == pytest.approx(closed.iloc[-1] - 1.0, rel=1e-6)


def test_portfolio_returns_rebalance_differs_from_buy_and_hold(synthetic_prices):
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    bh = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq=None)
    mo = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq="M")
    assert mo.weights_history is not None
    assert not np.isclose(mo.terminal, bh.terminal)
    assert list(mo.daily.columns) == ["Portfolio", "Benchmark"]
