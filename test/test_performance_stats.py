# test/test_performance_stats.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.performance_stats import (
    probabilistic_sharpe_ratio, deflated_sharpe_ratio, compare_strategies_oos,
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
    assert probabilistic_sharpe_ratio(r, sr_threshold=0.0) > 0.8
    assert probabilistic_sharpe_ratio(r, sr_threshold=5.0) < 0.2  # absurd per-period threshold


def test_dsr_in_unit_interval_and_decreasing_in_trials():
    r = _good()
    d_few = deflated_sharpe_ratio(r, n_trials=2)
    d_many = deflated_sharpe_ratio(r, n_trials=200)
    assert 0.0 <= d_many <= d_few <= 1.0
    assert d_many < d_few  # more trials => harder to beat => lower DSR


def test_compare_ranks_better_strategy_first():
    out = compare_strategies_oos({"good": _good(), "meh": _meh()})
    assert out["best_by_dsr"] == "good"
    assert set(out["summary"]["good"]) >= {"sharpe", "psr", "dsr"}
    assert 0.0 <= out["summary"]["good"]["psr"] <= 1.0
    assert isinstance(out["sharpe_diff_pvalues"], dict)


def test_compare_pvalues_in_unit_interval():
    out = compare_strategies_oos({"a": _good(seed=3), "b": _good(seed=4)})
    for p in out["sharpe_diff_pvalues"].values():
        assert 0.0 <= p <= 1.0
