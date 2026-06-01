"""Volatility signals & vol-targeting position sizing.

Pure, causal (look-ahead-safe) functions. The trailing-vol estimator here is the
sizing engine the whole tactical/overlay family (SP2) composes through.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_frame(returns):
    """Coerce a Series to a one-column DataFrame; pass DataFrames through."""
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    return returns


def compute_trailing_volatility(returns, lookback=63, method="ewma", annualize=True):
    """Trailing (causal) volatility per asset.

    Args:
        returns: DataFrame (or Series) of periodic simple returns.
        lookback: window length (rolling window for 'simple'; EWMA span for 'ewma').
        method: 'ewma' (exponentially weighted) or 'simple' (equal-weighted rolling).
        annualize: if True, multiply by sqrt(252).

    Returns:
        DataFrame of trailing volatility, same columns as `returns`. Value at row d
        uses only rows <= d (no center=True), so it is safe to .shift(1) for sizing.
    """
    df = _as_frame(returns)
    if method == "simple":
        vol = df.rolling(window=lookback).std(ddof=1)
    elif method == "ewma":
        vol = df.ewm(span=lookback, adjust=False, min_periods=lookback).std(bias=False)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'ewma' or 'simple'.")
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol


def volatility_target_positions(signal, returns, target_vol=0.10, vol_lookback=63,
                                method="ewma", max_leverage=2.0, scaling="per_asset", cov=None):
    """Scale a position `signal` so realized volatility targets `target_vol`.

    Look-ahead-safe: the volatility estimate is lagged one period (.shift(1)) before
    scaling, so the position at row d uses vol known as of d-1 and the signal at d.

    Args:
        signal: DataFrame of desired positions/weights over time (same cols as returns).
        returns: DataFrame of asset periodic returns.
        target_vol: annualized volatility target.
        vol_lookback: lookback for the vol estimate.
        method: passed to compute_trailing_volatility ('ewma'/'simple').
        max_leverage: cap on gross exposure (sum of |positions|) per row.
        scaling: 'per_asset' (scale each asset by target/own-vol) or
                 'portfolio' (single scalar/row from portfolio vol).
        cov: optional annualized covariance DataFrame for scaling='portfolio'
             (if None, portfolio vol is estimated from `returns`).

    Returns:
        DataFrame of scaled positions (NaN where the lagged vol is undefined).
    """
    sig = _as_frame(signal)
    rets = _as_frame(returns)
    sig, rets = sig.align(rets, join="inner", axis=1)

    if scaling == "per_asset":
        vol = compute_trailing_volatility(rets, lookback=vol_lookback, method=method,
                                          annualize=True).shift(1)
        scale = target_vol / vol.replace(0.0, np.nan)
        pos = sig * scale
    elif scaling == "portfolio":
        if cov is None:
            vol = compute_trailing_volatility(rets, lookback=vol_lookback, method=method,
                                              annualize=True).shift(1)
            port_vol = vol.mul(sig.abs()).sum(axis=1)  # crude proxy when no cov given
        else:
            w = sig.abs()
            cov_arr = cov.reindex(index=rets.columns, columns=rets.columns).values
            port_var = (w.values * (w.values @ cov_arr)).sum(axis=1)
            port_vol = pd.Series(np.sqrt(np.clip(port_var, 0, None)), index=sig.index).shift(1)
        scale = target_vol / port_vol.replace(0.0, np.nan)
        pos = sig.mul(scale, axis=0)
    else:
        raise ValueError(f"Unknown scaling {scaling!r}; expected 'per_asset' or 'portfolio'.")

    # Cap gross leverage per row, preserving relative sizing.
    gross = pos.abs().sum(axis=1)
    over = gross > max_leverage
    if over.any():
        factor = pd.Series(1.0, index=pos.index)
        factor[over] = max_leverage / gross[over]
        pos = pos.mul(factor, axis=0)
    return pos


# ---------------------------------------------------------------------------
# Tactical signals (SP2 Phase 5)
# All functions are causal: signal at row d uses only price data <= d.
# ---------------------------------------------------------------------------

def time_series_momentum_signal(prices, lookback=252, skip_recent=21):
    """Binary ±1 time-series momentum signal per asset.

    Signal at row d = sign(prices[d - skip_recent] / prices[d - lookback] - 1).
    NaN where either shifted price is unavailable.

    Args:
        prices: DataFrame of asset prices (not returns).
        lookback: total lookback window in rows.
        skip_recent: skip the most recent `skip_recent` rows to avoid microstructure noise.

    Returns:
        DataFrame of {-1, 0, 1} signals, same columns as prices.
    """
    df = _as_frame(prices) if isinstance(prices, pd.Series) else prices
    ret = df.shift(skip_recent) / df.shift(lookback) - 1
    return np.sign(ret)


def moving_average_crossover_signal(prices, fast=50, slow=200):
    """±1 signal based on fast vs. slow moving average crossover.

    Signal at row d = sign(fast_MA[d] - slow_MA[d]).
    Uses equal-weighted rolling MAs (causal, min_periods=window).
    NaN during the warm-up period (first `slow` rows).

    Args:
        prices: DataFrame of asset prices.
        fast: fast MA window in rows.
        slow: slow MA window in rows.

    Returns:
        DataFrame of {-1, 0, 1} signals.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be less than slow ({slow}).")
    df = _as_frame(prices) if isinstance(prices, pd.Series) else prices
    fast_ma = df.rolling(fast, min_periods=fast).mean()
    slow_ma = df.rolling(slow, min_periods=slow).mean()
    return np.sign(fast_ma - slow_ma)


def cross_sectional_momentum_score(prices, lookback=252, skip_recent=21):
    """Cross-sectional z-score of lookback returns per date.

    At each row d, compute the lookback return for each asset, then z-score
    across assets (mean 0, std 1). NaN rows are excluded from the z-score.

    Args:
        prices: DataFrame of asset prices.
        lookback: lookback window in rows.
        skip_recent: skip the most recent rows to avoid microstructure noise.

    Returns:
        DataFrame of z-scores (same shape as prices).
    """
    df = _as_frame(prices) if isinstance(prices, pd.Series) else prices
    raw = df.shift(skip_recent) / df.shift(lookback) - 1
    mean = raw.mean(axis=1)
    std = raw.std(axis=1, ddof=1).replace(0, np.nan)
    z = raw.sub(mean, axis=0).div(std, axis=0)
    return z


def zscore_reversion_signal(prices, lookback=20):
    """Z-score of price relative to its rolling mean/std.

    A positive z-score means the price is above its recent mean (overbought
    from a mean-reversion perspective); negate to use as a reversion signal.
    Causal: only uses data up to and including row d.

    Args:
        prices: DataFrame of asset prices.
        lookback: rolling window for mean and std.

    Returns:
        DataFrame of z-scores.
    """
    df = _as_frame(prices) if isinstance(prices, pd.Series) else prices
    roll = df.rolling(lookback, min_periods=lookback)
    mean = roll.mean()
    std = roll.std(ddof=1).replace(0, np.nan)
    return (df - mean) / std
