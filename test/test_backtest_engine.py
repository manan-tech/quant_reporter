import numpy as np
import pandas as pd
import pytest

from quant_reporter.backtest import transaction_cost_model, generate_rebalance_dates


def test_cost_model_golden():
    trades = pd.Series({"A": 0.2, "B": -0.2})  # notional traded (two-way) = 0.4
    out = transaction_cost_model(trades, commission_bps=10.0, spread_bps=20.0)
    # commission: 10/1e4 * 0.4 = 0.0004 ; half-spread: (20/2)/1e4 * 0.4 = 0.0004
    assert out["cost_breakdown"]["commission"] == pytest.approx(0.0004)
    assert out["cost_breakdown"]["spread"] == pytest.approx(0.0004)
    assert out["cost_frac"] == pytest.approx(0.0008)


def test_cost_model_zero_bps_zero_cost():
    out = transaction_cost_model({"A": 0.5, "B": -0.5}, commission_bps=0.0, spread_bps=0.0)
    assert out["cost_frac"] == 0.0
    assert out["cost_cash"] == 0.0


def test_cost_model_monotone_in_bps():
    t = {"A": 0.3, "B": -0.3}
    lo = transaction_cost_model(t, commission_bps=1.0, spread_bps=5.0)["cost_frac"]
    hi = transaction_cost_model(t, commission_bps=2.0, spread_bps=10.0)["cost_frac"]
    assert hi > lo


def test_cost_model_cash_and_portfolio_value():
    out = transaction_cost_model({"A": 0.2, "B": -0.2}, commission_bps=10.0, spread_bps=0.0,
                                 portfolio_value=1_000_000.0)
    assert out["cost_cash"] == pytest.approx(out["cost_frac"] * 1_000_000.0)


def test_cost_model_impact_hook_adds_cost():
    base = transaction_cost_model({"A": 0.2, "B": -0.2}, commission_bps=1.0, spread_bps=1.0)["cost_frac"]
    with_impact = transaction_cost_model({"A": 0.2, "B": -0.2}, commission_bps=1.0, spread_bps=1.0,
                                         impact_model=lambda trades: 0.0005)
    assert with_impact["cost_breakdown"]["impact"] == pytest.approx(0.0005)
    assert with_impact["cost_frac"] == pytest.approx(base + 0.0005)


def test_generate_rebalance_dates_monthly_includes_first():
    idx = pd.bdate_range("2021-01-01", periods=260)
    dates = generate_rebalance_dates(idx, mode="calendar", freq="M")
    assert idx[0] in dates
    # one rebalance per distinct month
    assert len(dates) == idx.to_series().dt.to_period("M").nunique()
    assert dates.isin(idx).all()


def test_generate_rebalance_dates_int_every_n():
    idx = pd.bdate_range("2021-01-01", periods=100)
    dates = generate_rebalance_dates(idx, mode="calendar", freq=20)
    assert idx[0] in dates
    assert dates.isin(idx).all()


def test_generate_rebalance_dates_non_calendar_raises():
    idx = pd.bdate_range("2021-01-01", periods=50)
    with pytest.raises(NotImplementedError):
        generate_rebalance_dates(idx, mode="threshold")


from quant_reporter.backtest import simulate_strategy, transaction_cost_model
from quant_reporter.rebalancing import simulate_rebalanced_portfolio
from conftest import make_synthetic_prices
import functools


def _prices():
    return make_synthetic_prices(n_days=400)[["AAA", "BBB", "CCC"]]


def test_simulate_backcompat_frictionless_buy_and_hold():
    prices = _prices()
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    res = simulate_strategy(prices, w, cost_model=None, rebalance=None, initial_value=1.0)
    ref_wealth, _ = simulate_rebalanced_portfolio(prices, w, None)
    pd.testing.assert_series_equal(res["wealth"], ref_wealth.rename("Portfolio"),
                                   check_names=False, rtol=1e-9, atol=1e-12)
    assert res["cost_drag"] == pytest.approx(0.0)


def test_simulate_returns_expected_keys():
    res = simulate_strategy(_prices(), {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}, rebalance="M")
    assert set(res) >= {"wealth", "weights", "blotter", "turnover", "cost_drag", "summary"}
    assert set(res["summary"]) >= {"terminal_wealth", "total_return", "n_rebalances",
                                   "avg_turnover", "cost_drag", "max_drawdown"}


def test_simulate_costs_reduce_terminal_monotonically():
    prices = _prices()
    w = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    free = simulate_strategy(prices, w, cost_model=None, rebalance="M")
    cheap = simulate_strategy(prices, w, rebalance="M",
                              cost_model=functools.partial(transaction_cost_model,
                                                           commission_bps=1.0, spread_bps=5.0))
    pricey = simulate_strategy(prices, w, rebalance="M",
                               cost_model=functools.partial(transaction_cost_model,
                                                            commission_bps=10.0, spread_bps=50.0))
    assert free["summary"]["terminal_wealth"] > cheap["summary"]["terminal_wealth"]
    assert cheap["summary"]["terminal_wealth"] > pricey["summary"]["terminal_wealth"]
    assert pricey["cost_drag"] > cheap["cost_drag"] > 0.0


def test_simulate_accepts_dataframe_schedule():
    prices = _prices()
    dates = prices.index[[0, 120, 240]]
    sched = pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.34, 0.33, 0.33]],
        index=dates, columns=["AAA", "BBB", "CCC"],
    )
    res = simulate_strategy(prices, sched, cost_model=None)
    assert res["wealth"].iloc[0] == pytest.approx(1.0)
    # at least the 3 schedule changes are rebalance points
    assert res["summary"]["n_rebalances"] >= 3


def test_simulate_initial_value_scales_wealth():
    prices = _prices()
    w = {"AAA": 1.0}
    a = simulate_strategy(prices, w, rebalance=None, initial_value=1.0)["wealth"]
    b = simulate_strategy(prices, w, rebalance=None, initial_value=1000.0)["wealth"]
    assert np.allclose(b.values, a.values * 1000.0, rtol=1e-9, atol=0.0)
    assert a.iloc[0] == pytest.approx(1.0)
    assert b.iloc[0] == pytest.approx(1000.0)


def test_simulate_cash_drag_compounds_daily():
    prices = _prices()
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    base = simulate_strategy(prices, w, cost_model=None, rebalance=None, cash_drag=0.0)
    drag = simulate_strategy(prices, w, cost_model=None, rebalance=None, cash_drag=0.02)
    # drag applied on days 1..n-1 (not day 0), so exponent = len(prices)-1
    expected_ratio = (1 - 0.02 / 252) ** (len(prices) - 1)
    assert drag["wealth"].iloc[-1] / base["wealth"].iloc[-1] == pytest.approx(expected_ratio, rel=1e-12)


def test_simulate_blotter_contents():
    prices = _prices()
    w = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    cm = functools.partial(transaction_cost_model, commission_bps=2, spread_bps=8)
    res = simulate_strategy(prices, w, rebalance="M", cost_model=cm)
    blotter = res["blotter"]
    assert len(blotter) >= 1
    for row in blotter:
        assert set(row) == {"date", "turnover", "cost_frac"}
    # day-0 entry: cash -> target, one-way turnover = 0.5*sum|deltas| = 0.5
    assert blotter[0]["turnover"] == pytest.approx(0.5)
    assert blotter[0]["cost_frac"] > 0
    # blotter length matches n_rebalances (day-0 entry is included in both)
    assert len(blotter) == res["summary"]["n_rebalances"]


@pytest.mark.parametrize("freq", ["M", "Q", "Y"])
def test_generate_rebalance_dates_parametrized(freq):
    idx = pd.bdate_range("2020-01-01", periods=520)
    dates = generate_rebalance_dates(idx, mode="calendar", freq=freq)
    assert idx[0] in dates
    assert len(dates) == idx.to_series().dt.to_period(freq).nunique()
    assert dates.isin(idx).all()


def test_simulate_missing_ticker_raises():
    prices = _prices()
    with pytest.raises(ValueError, match="absent from price_data"):
        simulate_strategy(prices, {"AAA": 0.5, "ZZZ": 0.5})


def test_simulate_empty_prices_raises():
    prices = _prices().iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        simulate_strategy(prices, {"AAA": 0.5, "BBB": 0.5})


def test_simulate_net_zero_weights_raises():
    prices = _prices()
    with pytest.raises(ValueError, match="net-positive"):
        simulate_strategy(prices, {"AAA": 0.5, "BBB": -0.5})


from hypothesis import given, settings, strategies as st


@settings(max_examples=20, deadline=None)
@given(cut=st.integers(min_value=120, max_value=300))
def test_simulate_is_causal_under_future_shuffle(cut):
    """Shuffling price rows strictly after `cut` must not change wealth at/<=cut."""
    prices = make_synthetic_prices(n_days=400, seed=5)[["AAA", "BBB", "CCC"]]
    w = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}
    base = simulate_strategy(prices, w, rebalance="M",
                             cost_model=functools.partial(transaction_cost_model,
                                                          commission_bps=2.0, spread_bps=8.0))["wealth"]
    shuffled = prices.copy()
    rng = np.random.default_rng(7)
    tail = np.arange(cut + 1, len(prices))
    shuffled.iloc[cut + 1:] = prices.iloc[rng.permutation(tail)].values
    shuf = simulate_strategy(shuffled, w, rebalance="M",
                             cost_model=functools.partial(transaction_cost_model,
                                                          commission_bps=2.0, spread_bps=8.0))["wealth"]
    pd.testing.assert_series_equal(base.iloc[: cut + 1], shuf.iloc[: cut + 1],
                                   check_names=False, rtol=1e-9, atol=1e-12)
