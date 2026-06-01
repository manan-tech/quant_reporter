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


def test_trend_following_drops_warmup_rows():
    """C1 regression: warmup rows (no valid signal) must be dropped, NOT equal-weighted."""
    prices = _prices(n=800)
    sched = trend_following(prices)
    # Schedule must start well after the lookback warmup (no leading 1/3,1/3,1/3 fill).
    first_loc = prices.index.get_loc(sched.index[0])
    assert first_loc >= 200, f"schedule starts at row {first_loc}; warmup not dropped"


def test_cross_sectional_momentum_drops_warmup_rows():
    prices = _prices(n=800)
    sched = cross_sectional_momentum(prices)
    first_loc = prices.index.get_loc(sched.index[0])
    # score lookback=126 drives NaN warmup; schedule must start well after that boundary.
    assert first_loc >= 100, f"schedule starts at row {first_loc}; warmup not dropped"


def test_vol_target_overlay_tilts_toward_low_vol():
    """I2 regression: overlay must actually change weights (inverse-vol tilt), not be a no-op.
    In the synthetic fixture AAA has the lowest vol and CCC the highest, so AAA should
    receive more weight than CCC under the inverse-vol tilt."""
    overlay = vol_target_overlay(equal_weight, target_vol=0.10)
    sched = overlay(_prices(n=800))
    last = sched.iloc[-1]
    assert last["AAA"] > last["CCC"]
    # And it must NOT be uniform equal weight (proves it isn't a no-op).
    assert abs(last["AAA"] - last["CCC"]) > 1e-3
