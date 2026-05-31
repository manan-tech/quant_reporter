import numpy as np
import pandas as pd
import pytest

from quant_reporter.signals import compute_trailing_volatility


def _returns(n=260, seed=1, cols=("AAA", "BBB")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(
        {c: rng.normal(0.0004, 0.01 + 0.003 * i, n) for i, c in enumerate(cols)},
        index=idx,
    )


def test_trailing_vol_simple_golden():
    s = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01], name="AAA")
    df = compute_trailing_volatility(s.to_frame(), lookback=3, method="simple", annualize=False)
    # rolling std (ddof=1) of [0.01,-0.01,0.02] at row 2
    assert df["AAA"].iloc[2] == pytest.approx(0.0152752523, rel=1e-6)
    assert np.isnan(df["AAA"].iloc[0]) and np.isnan(df["AAA"].iloc[1])


def test_trailing_vol_annualize_scales_by_sqrt_252():
    r = _returns()
    raw = compute_trailing_volatility(r, lookback=63, method="simple", annualize=False)
    ann = compute_trailing_volatility(r, lookback=63, method="simple", annualize=True)
    ratio = (ann / raw).dropna()
    assert np.allclose(ratio.values, np.sqrt(252))


def test_trailing_vol_ewma_runs_and_is_positive():
    r = _returns()
    vol = compute_trailing_volatility(r, lookback=63, method="ewma", annualize=True)
    tail = vol.dropna()
    assert (tail.values > 0).all()
    assert list(vol.columns) == list(r.columns)


def test_trailing_vol_accepts_series_returns_dataframe():
    s = _returns(cols=("AAA",))["AAA"]
    vol = compute_trailing_volatility(s, lookback=20)
    assert isinstance(vol, pd.DataFrame)
    assert list(vol.columns) == ["AAA"]


def test_trailing_vol_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_trailing_volatility(_returns(), method="garch")


from hypothesis import given, settings, strategies as st


@settings(max_examples=30, deadline=None)
@given(cut=st.integers(min_value=80, max_value=180))
def test_trailing_vol_is_causal_under_future_shuffle(cut):
    """Shuffling rows strictly after `cut` must not change vol values at/<=cut."""
    r = _returns(n=200, seed=7)
    vol_full = compute_trailing_volatility(r, lookback=40, method="ewma")
    shuffled = r.copy()
    rng = np.random.default_rng(123)
    tail_idx = np.arange(cut + 1, len(r))
    perm = rng.permutation(tail_idx)
    shuffled.iloc[cut + 1:] = r.iloc[perm].values
    vol_shuf = compute_trailing_volatility(shuffled, lookback=40, method="ewma")
    pd.testing.assert_frame_equal(vol_full.iloc[: cut + 1], vol_shuf.iloc[: cut + 1])


from quant_reporter.signals import volatility_target_positions


def test_vol_target_per_asset_scales_toward_target():
    r = _returns(n=400, seed=3)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)  # always long 1.0
    pos = volatility_target_positions(signal, r, target_vol=0.10, vol_lookback=63,
                                      method="simple", max_leverage=5.0, scaling="per_asset")
    # Lower-vol asset (AAA) should get a larger position than higher-vol asset (BBB)
    tail = pos.dropna()
    assert tail["AAA"].mean() > tail["BBB"].mean()
    assert (tail.values >= 0).all()


def test_vol_target_respects_max_leverage():
    r = _returns(n=400, seed=4)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    pos = volatility_target_positions(signal, r, target_vol=10.0,  # absurd target forces clipping
                                      vol_lookback=63, method="simple", max_leverage=2.0,
                                      scaling="per_asset")
    gross = pos.abs().sum(axis=1).dropna()
    assert (gross <= 2.0 + 1e-9).all()


def test_vol_target_zero_signal_zero_position():
    r = _returns(n=200, seed=5)
    signal = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    pos = volatility_target_positions(signal, r, target_vol=0.10, vol_lookback=40,
                                      method="simple", scaling="per_asset")
    assert np.allclose(pos.fillna(0.0).values, 0.0)


@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=90, max_value=160))
def test_vol_target_positions_causal_under_future_shuffle(cut):
    r = _returns(n=200, seed=11)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    base = volatility_target_positions(signal, r, target_vol=0.1, vol_lookback=40,
                                       method="simple", scaling="per_asset")
    shuffled = r.copy()
    rng = np.random.default_rng(99)
    tail_idx = np.arange(cut + 1, len(r))
    shuffled.iloc[cut + 1:] = r.iloc[rng.permutation(tail_idx)].values
    shuf = volatility_target_positions(signal, shuffled, target_vol=0.1, vol_lookback=40,
                                       method="simple", scaling="per_asset")
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], shuf.iloc[: cut + 1])
