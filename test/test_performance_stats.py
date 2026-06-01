# test/test_performance_stats.py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quant_reporter.performance_stats import (
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, compare_strategies_oos,
    _sharpe_diff_pvalue,
)


def _good(n=750, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0008, 0.01, n))  # high positive Sharpe


def _meh(n=750, seed=2):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.00005, 0.012, n))  # near-zero Sharpe


def test_psr_in_unit_interval():
    psr = probabilistic_sharpe_ratio(_good())
    assert 0.0 <= psr <= 1.0


def test_psr_high_for_strong_series_low_for_high_threshold():
    r = _good()
    # With the corrected per-period basis, PSR at threshold=0 is a realistic probability.
    assert probabilistic_sharpe_ratio(r, sr_threshold=0.0) > 0.5
    # Absurd annualized threshold (SR=2.0) should push PSR very low.
    assert probabilistic_sharpe_ratio(r, sr_threshold=2.0) < 0.1


def test_psr_scale_invariant_at_zero_threshold():
    # PSR(r, 0, ppy) must be identical for all ppy because at threshold=0,
    # the per-period threshold is 0 regardless of the annualization factor.
    r = _good()
    psr_1 = probabilistic_sharpe_ratio(r, sr_threshold=0.0, periods_per_year=1)
    psr_252 = probabilistic_sharpe_ratio(r, sr_threshold=0.0, periods_per_year=252)
    assert psr_1 == pytest.approx(psr_252, rel=1e-9)
    # And it must not be saturated at 1.0 for a realistic daily Sharpe.
    assert psr_252 < 0.9999


def test_psr_undefined_returns_nan():
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([1.0, 1.0, 1.0])))  # zero variance
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.01, 0.02])))     # n < 3


def test_dsr_in_unit_interval_and_decreasing_in_trials():
    r = _good()
    d_few = deflated_sharpe_ratio(r, n_trials=2)
    d_many = deflated_sharpe_ratio(r, n_trials=200)
    assert 0.0 <= d_many <= d_few <= 1.0
    assert d_many < d_few  # more trials => harder to beat => lower DSR


def test_dsr_not_saturated_at_one():
    # With enough trials, DSR must drop meaningfully below 1.
    r = _good()
    assert deflated_sharpe_ratio(r, n_trials=1000) < 0.9


def test_dsr_undefined_returns_nan():
    assert np.isnan(deflated_sharpe_ratio(pd.Series([1.0, 1.0, 1.0]), n_trials=5))


def test_compare_ranks_better_strategy_first():
    out = compare_strategies_oos({"good": _good(), "meh": _meh()})
    assert out["best_by_dsr"] == "good"
    assert set(out["summary"]["good"]) >= {"sharpe", "psr", "dsr"}
    assert 0.0 <= out["summary"]["good"]["psr"] <= 1.0
    assert isinstance(out["sharpe_diff_pvalues"], dict)


def test_compare_empty_returns_none_best():
    out = compare_strategies_oos({})
    assert out["best_by_dsr"] is None
    assert out["summary"] == {}


def test_compare_degenerate_strategy_cannot_win():
    # A zero-vol (undefined) series must never be selected as best.
    out = compare_strategies_oos({"good": _good(), "flat": pd.Series([0.001] * 750)})
    assert out["best_by_dsr"] == "good"


def test_compare_pvalues_value_pinned():
    # Pin the actual Jobson-Korkie/Memmel p-value formula, not just range.
    rng = np.random.default_rng(42)
    a = pd.Series(rng.normal(0.0008, 0.01, 500))
    b = pd.Series(rng.normal(0.0003, 0.01, 500))
    p = _sharpe_diff_pvalue(a, b)
    # Recompute independently with the same formula.
    x, y = a.values[:len(b)], b.values[:len(a)]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    srx, sry = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = np.corrcoef(x, y)[0, 1]
    var = (1.0 / n) * (2 - 2 * rho + 0.5 * (srx**2 + sry**2 - 2 * srx * sry * rho**2))
    z_ref = (srx - sry) / np.sqrt(var)
    p_ref = float(2 * (1 - stats.norm.cdf(abs(z_ref))))
    assert p == pytest.approx(p_ref, rel=1e-9)


def test_compare_with_benchmark():
    bm = pd.Series(np.random.default_rng(99).normal(0.0004, 0.01, 750))
    out = compare_strategies_oos({"a": _good(seed=5), "b": _meh(seed=6)},
                                 benchmark_returns=bm)
    assert "a vs Benchmark" in out["sharpe_diff_pvalues"]
    assert "b vs Benchmark" in out["sharpe_diff_pvalues"]
    # Values must match direct _sharpe_diff_pvalue calls.
    assert out["sharpe_diff_pvalues"]["a vs Benchmark"] == pytest.approx(
        _sharpe_diff_pvalue(_good(seed=5), bm), rel=1e-9)


def test_compare_pvalues_in_unit_interval():
    out = compare_strategies_oos({"a": _good(seed=3), "b": _good(seed=4)})
    for p in out["sharpe_diff_pvalues"].values():
        assert np.isnan(p) or 0.0 <= p <= 1.0
