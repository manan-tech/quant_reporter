# src/quant_reporter/strategies.py
"""Prebuilt strategies (SP-Strategy).

Each is a function (prices, **params) -> weights, where weights is a dict
(static) or a DataFrame (dated, look-ahead-safe schedule). REGISTRY lists the
no-argument-factory strategies for discoverability.
"""
import numpy as np
import pandas as pd


def _long_only_schedule(sized, prices):
    """Clip to long-only and renormalize each row to sum to 1.

    Rows with any NaN (signal/vol warmup, i.e. undefined) are DROPPED. Rows that
    are fully defined but all-zero/negative after clipping fall back to equal weight.
    """
    cols = list(prices.columns)
    n = len(cols)
    long_only = sized.reindex(columns=cols).clip(lower=0)
    valid = ~long_only.isna().any(axis=1)            # fully-defined rows only
    row_sum = long_only.sum(axis=1)                  # skipna=True (NaN rows -> 0)
    normalized = long_only.div(row_sum.where(row_sum > 0, other=1.0), axis=0)
    eq = pd.DataFrame(1.0 / n, index=normalized.index, columns=cols)
    # Equal-weight fallback ONLY for fully-defined rows that clipped to all-zero.
    flat_valid = (valid & (row_sum == 0)).values
    use_eq = pd.DataFrame(np.repeat(flat_valid[:, None], n, axis=1),
                          index=normalized.index, columns=cols)
    result = normalized.mask(use_eq, eq)
    return result.dropna(how="any")                  # drops the NaN warmup rows


def equal_weight(prices, **kwargs):
    cols = list(prices.columns)
    return {c: 1.0 / len(cols) for c in cols}


def inverse_vol(prices, lookback=63, method="ewma", **kwargs):
    from .sizing import inverse_volatility_weights
    returns = prices.pct_change().dropna()
    return inverse_volatility_weights(returns, lookback=lookback, method=method)


def min_variance(prices, **kwargs):
    from .robust_estimators import ledoit_wolf_covariance
    from .opt_core import find_optimal_portfolio, objective_min_variance, build_constraints
    returns = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(returns)["cov_matrix"]
    mean = returns.mean() * 252
    n = len(cov.columns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = build_constraints(n, list(cov.columns))
    w = find_optimal_portfolio(objective_min_variance, mean, cov, bounds, constraints)
    return dict(zip(cov.columns, np.asarray(w, dtype=float)))


def risk_parity(prices, **kwargs):
    from .robust_estimators import ledoit_wolf_covariance
    from .opt_core import optimize_risk_budget
    returns = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(returns)["cov_matrix"]
    return optimize_risk_budget(cov)["weights"]


def max_sharpe(prices, risk_free_rate=0.02, **kwargs):
    from .opt_core import (get_optimization_inputs, find_optimal_portfolio,
                           objective_neg_sharpe, build_constraints)
    mean, cov, _ = get_optimization_inputs(prices)
    n = len(cov.columns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = build_constraints(n, list(cov.columns))
    w = find_optimal_portfolio(objective_neg_sharpe, mean, cov, bounds, constraints, risk_free_rate)
    return dict(zip(cov.columns, np.asarray(w, dtype=float)))


def trend_following(prices, lookback=252, skip_recent=21, fast=50, slow=200,
                    target_vol=0.10, vol_lookback=63, max_leverage=2.0, **kwargs):
    from .signals import (time_series_momentum_signal, moving_average_crossover_signal,
                          volatility_target_positions)
    tsm = time_series_momentum_signal(prices, lookback=lookback, skip_recent=skip_recent)
    mac = moving_average_crossover_signal(prices, fast=fast, slow=slow)
    ensemble = ((tsm + mac) / 2.0).dropna(how="any")
    sized = volatility_target_positions(ensemble, prices.pct_change(), target_vol=target_vol,
                                        vol_lookback=vol_lookback, method="ewma",
                                        max_leverage=max_leverage, scaling="per_asset")
    return _long_only_schedule(sized, prices)


def cross_sectional_momentum(prices, lookback=126, skip_recent=5, target_vol=0.10,
                             vol_lookback=63, max_leverage=2.0, **kwargs):
    from .signals import cross_sectional_momentum_score, volatility_target_positions
    score = cross_sectional_momentum_score(prices, lookback=lookback, skip_recent=skip_recent)
    # Preserve NaN during the score warmup so undefined rows are dropped (not
    # silently treated as a flat/equal-weight position) by _long_only_schedule.
    longs = (score > 0).astype(float).where(score.notna())
    sized = volatility_target_positions(longs, prices.pct_change(), target_vol=target_vol,
                                        vol_lookback=vol_lookback, method="ewma",
                                        max_leverage=max_leverage, scaling="per_asset")
    return _long_only_schedule(sized, prices)


def vol_target_overlay(base_fn, target_vol=0.10, vol_lookback=63, max_leverage=2.0):
    """Higher-order: wrap base_fn with an inverse-volatility tilt.

    NOTE: the backtest engine is fully invested (weights renormalize to sum 1),
    so absolute portfolio-vol targeting (gross-exposure scaling) is not expressible
    here — a uniform scalar cancels under renormalization. This overlay therefore
    applies a per-asset inverse-vol tilt: lower-realized-vol assets receive more
    weight. `target_vol` sets the pre-normalization scale (affects only the
    max_leverage cap interaction).
    """
    def _strategy(prices, **kwargs):
        from .signals import volatility_target_positions
        base = base_fn(prices, **kwargs)
        if isinstance(base, dict):
            sig = pd.DataFrame([base] * len(prices), index=prices.index)
        else:
            sig = base
        sig = sig.reindex(columns=prices.columns).fillna(0.0)
        sized = volatility_target_positions(sig, prices.pct_change(), target_vol=target_vol,
                                            vol_lookback=vol_lookback, method="ewma",
                                            max_leverage=max_leverage, scaling="per_asset")
        return _long_only_schedule(sized, prices)
    return _strategy


# vol_target_overlay is intentionally excluded: it is a factory that requires a
# base_fn argument, not a zero-arg strategy.
REGISTRY = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_sharpe": max_sharpe,
    "trend_following": trend_following,
    "cross_sectional_momentum": cross_sectional_momentum,
}
