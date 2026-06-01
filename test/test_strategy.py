# test/test_strategy.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.strategy import Strategy, backtest, backtest_many, BacktestResult
from quant_reporter.backtest import simulate_strategy
from quant_reporter.strategies import equal_weight, risk_parity
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)  # includes BMK benchmark column


def test_backtest_accepts_callable():
    res = backtest(equal_weight, _prices()[["AAA", "BBB", "CCC"]], rebalance="M")
    assert isinstance(res, BacktestResult)
    assert res.wealth.iloc[-1] > 0


def test_backtest_accepts_strategy_wrapper():
    strat = Strategy("EW", equal_weight)
    res = backtest(strat, _prices()[["AAA", "BBB", "CCC"]])
    assert res.name == "EW"


def test_backtest_accepts_static_dict():
    res = backtest({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, _prices()[["AAA", "BBB", "CCC"]])
    assert isinstance(res, BacktestResult)


def test_frictionless_backtest_matches_simulate_strategy():
    prices = _prices()[["AAA", "BBB", "CCC"]]
    w = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    res = backtest(w, prices, rebalance="M", cost_model=None)
    sim = simulate_strategy(prices, w, cost_model=None, rebalance="M")
    pd.testing.assert_series_equal(res.wealth, sim["wealth"])


def test_benchmark_column_is_split_off():
    prices = _prices()  # AAA/BBB/CCC + BMK
    res = backtest(equal_weight, prices, benchmark="BMK")
    assert res.benchmark is not None
    assert "BMK" not in res.weights.columns


def test_metrics_cached_and_have_keys():
    res = backtest(risk_parity, _prices()[["AAA", "BBB", "CCC"]])
    m = res.metrics
    assert res.metrics is m  # cached identity
    assert "Sharpe" in m and "Max Drawdown" in m


def test_oos_stats_present():
    res = backtest(equal_weight, _prices()[["AAA", "BBB", "CCC"]])
    assert set(res.oos_stats) == {"psr", "dsr"}


def test_backtest_many_returns_dict_of_results():
    prices = _prices()[["AAA", "BBB", "CCC"]]
    out = backtest_many({"EW": equal_weight, "RP": risk_parity}, prices)
    assert set(out) == {"EW", "RP"}
    assert all(isinstance(r, BacktestResult) for r in out.values())
    assert out["EW"].n_trials == 2  # used for DSR deflation


def test_series_benchmark_no_overlap_warns():
    import warnings
    prices = _prices()[["AAA", "BBB", "CCC"]]
    # Integer RangeIndex benchmark cannot align to the DatetimeIndex -> all NaN
    bad_bench = pd.Series(np.linspace(100, 120, len(prices)))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = backtest(equal_weight, prices, benchmark=bad_bench)
    assert any("no overlap" in str(x.message) for x in w)
    assert np.isnan(res.metrics["Tracking Error"])


def test_missing_benchmark_column_raises_clear_keyerror():
    prices = _prices()[["AAA", "BBB", "CCC"]]
    with pytest.raises(KeyError, match="not found in prices"):
        backtest(equal_weight, prices, benchmark="SPY")


def test_partial_overlap_benchmark_no_futurewarning():
    import warnings
    prices = _prices(n=600)[["AAA", "BBB", "CCC"]]
    half = len(prices) // 2
    # Benchmark covering only the last half -> leading NaN after reindex+ffill
    bench = pd.Series(np.linspace(100, 130, len(prices) - half), index=prices.index[half:])
    res = backtest(equal_weight, prices, benchmark=bench)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        m = res.metrics  # pct_change on a NaN-containing benchmark must NOT FutureWarn
    assert "Tracking Error" in m
