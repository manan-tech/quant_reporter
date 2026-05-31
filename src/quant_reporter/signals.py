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
