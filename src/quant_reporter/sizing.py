# src/quant_reporter/sizing.py
"""Risk overlay & position-sizing primitives (SP2 Phase 3).

Pure, look-ahead-safe functions that scale and overlay positions.
All vol conventions are annualized unless documented otherwise.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def forecast_portfolio_vol(weights, cov_matrix):
    """Annualized portfolio volatility from weight vector and covariance.

    Args:
        weights: dict {ticker: weight} or array-like aligned with cov_matrix.
        cov_matrix: annualized covariance DataFrame or ndarray.

    Returns:
        float: annualized portfolio volatility.
    """
    if isinstance(weights, dict):
        if not hasattr(cov_matrix, "columns"):
            raise ValueError("cov_matrix must be a DataFrame when weights is a dict.")
        w = np.array([weights.get(c, 0.0) for c in cov_matrix.columns], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    cov = cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix)
    variance = float(w @ cov @ w)
    return float(np.sqrt(max(0.0, variance)))


def target_volatility_scalar(portfolio_returns, target_vol=0.10, lookback=63,
                              periods_per_year=252):
    """Scalar to multiply current positions to hit `target_vol`.

    Uses trailing EWMA vol (span=`lookback`), annualized.
    Clipped to [0, 5] to prevent blow-up on near-zero vol.

    Args:
        portfolio_returns: Series (or array) of periodic portfolio returns.
        target_vol: annualized target volatility.
        lookback: EWMA span for vol estimation.
        periods_per_year: annualization factor.

    Returns:
        float: scaling factor. Returns 1.0 when vol cannot be estimated.
    """
    r = pd.Series(portfolio_returns).dropna()
    if len(r) < 2:
        return 1.0
    realized_vol = float(
        r.ewm(span=lookback, adjust=False, min_periods=max(2, lookback // 4))
        .std(bias=False)
        .iloc[-1]
        * np.sqrt(periods_per_year)
    )
    if realized_vol <= 0 or not np.isfinite(realized_vol):
        return 1.0
    return float(np.clip(target_vol / realized_vol, 0.0, 5.0))


def inverse_volatility_weights(returns, lookback=63, method="ewma", periods_per_year=252):
    """Weights proportional to 1/vol, normalized to sum to 1.

    Look-ahead-safe: vol estimated on the last `lookback` rows only.

    Args:
        returns: DataFrame of periodic asset returns.
        lookback: span ('ewma') or window ('simple') for vol estimation.
        method: 'ewma' or 'simple'.
        periods_per_year: annualization factor (passed through to vol estimator).

    Returns:
        dict {ticker: weight} — weights sum to 1.0.
    """
    from .signals import compute_trailing_volatility
    df = returns if isinstance(returns, pd.DataFrame) else returns.to_frame()
    vol = compute_trailing_volatility(df, lookback=lookback, method=method, annualize=True)
    last_vol = vol.iloc[-1]
    bad = (last_vol <= 0) | (~np.isfinite(last_vol))
    if bad.all():
        n = len(df.columns)
        return {c: 1.0 / n for c in df.columns}
    inv_vol = (1.0 / last_vol).where(~bad, other=np.nan)
    total = inv_vol.sum()
    weights = (inv_vol / total).fillna(0.0)
    return dict(weights)


def realized_tracking_error(portfolio_returns, benchmark_returns, periods_per_year=252):
    """Annualized standard deviation of active returns (portfolio − benchmark).

    Args:
        portfolio_returns: Series of periodic portfolio returns.
        benchmark_returns: Series of benchmark periodic returns (same frequency).
        periods_per_year: annualization factor.

    Returns:
        float: annualized tracking error.  NaN if fewer than 2 overlapping observations.
    """
    port = pd.Series(portfolio_returns)
    bench = pd.Series(benchmark_returns)
    active = (port - bench).dropna()
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def kelly_fraction(mean_excess_return, variance):
    """Full Kelly fraction: f* = μ_excess / σ².

    Args:
        mean_excess_return: expected excess return per period (arithmetic mean).
        variance: return variance per period.

    Returns:
        float: Kelly fraction.  0.0 if variance ≤ 0.
    """
    if variance <= 0:
        return 0.0
    return float(mean_excess_return / variance)


def cppi_weights(floor_value, multiplier, portfolio_value, risky_weight=1.0):
    """CPPI cushion → risky/safe allocation fractions.

    Cushion = portfolio_value − floor_value.
    Risky exposure = multiplier × cushion, capped at `risky_weight` × portfolio_value.
    Safe fraction = 1 − risky fraction.

    Args:
        floor_value: minimum acceptable portfolio value (absolute).
        multiplier: CPPI multiplier m (typically 3–5).
        portfolio_value: current total portfolio value.
        risky_weight: max fraction allocable to the risky asset (default 1.0).

    Returns:
        dict: {'risky': float, 'safe': float} — fractions summing to 1.0.
    """
    if portfolio_value <= 0:
        return {"risky": 0.0, "safe": 1.0}
    cushion = max(0.0, portfolio_value - floor_value)
    risky_frac = min(multiplier * cushion / portfolio_value, max(0.0, float(risky_weight)))
    risky_frac = min(risky_frac, 1.0)
    return {"risky": float(risky_frac), "safe": float(1.0 - risky_frac)}
