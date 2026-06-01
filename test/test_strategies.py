# test/test_strategies.py
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from quant_reporter.strategies import (
    equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe,
    trend_following, cross_sectional_momentum, vol_target_overlay, REGISTRY,
)
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


@pytest.mark.parametrize("fn", [equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe])
def test_static_strategy_weights_sum_to_one(fn):
    w = fn(_prices())
    assert isinstance(w, dict)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(w) == {"AAA", "BBB", "CCC"}
    assert all(v >= -1e-9 for v in w.values())


def test_registry_contains_expected():
    for name in ("equal_weight", "inverse_vol", "min_variance", "risk_parity",
                 "max_sharpe", "trend_following", "cross_sectional_momentum"):
        assert name in REGISTRY and callable(REGISTRY[name])


def test_trend_following_returns_valid_schedule():
    sched = trend_following(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()
    assert list(sched.columns) == ["AAA", "BBB", "CCC"]


def test_cross_sectional_momentum_returns_valid_schedule():
    sched = cross_sectional_momentum(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()


def test_vol_target_overlay_wraps_base():
    overlay = vol_target_overlay(equal_weight, target_vol=0.10)
    sched = overlay(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()


@settings(max_examples=15, deadline=None)
@given(cut=st.integers(min_value=300, max_value=500))
def test_trend_following_is_causal(cut):
    prices = _prices(n=600, seed=5)
    base = trend_following(prices)
    shuffled = prices.copy()
    rng = np.random.default_rng(7)
    tail = np.arange(cut + 1, len(prices))
    shuffled.iloc[cut + 1:] = prices.iloc[rng.permutation(tail)].values
    shuf = trend_following(shuffled)
    common = base.index.intersection(shuf.index)
    common = [d for d in common if prices.index.get_loc(d) <= cut]
    pd.testing.assert_frame_equal(base.loc[common], shuf.loc[common])
