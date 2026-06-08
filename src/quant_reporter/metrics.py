import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownResult:
    curve: pd.Series
    max_dd: float


def compute_drawdown(cumulative_returns):
    """Underwater curve + scalar max drawdown from one cumulative (Growth-of-$1) series."""
    peak = cumulative_returns.cummax()
    curve = (cumulative_returns - peak) / peak
    return DrawdownResult(curve=curve, max_dd=float(curve.min()))


def calculate_max_drawdown(cumulative_returns):
    """Backward-compatible scalar max drawdown (delegates to compute_drawdown)."""
    return compute_drawdown(cumulative_returns).max_dd

def calculate_var_cvar(daily_returns, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    if daily_returns.empty:
        return np.nan, np.nan
    var = daily_returns.quantile(1 - confidence_level)
    cvar = daily_returns[daily_returns <= var].mean()

    return var, cvar


# ---------------------------------------------------------------------------
# SP-Strategy: consolidated measurement surface.
# All functions take SIMPLE periodic returns and return plain floats.
# Edge cases (empty / single-obs / zero-vol) return NaN (or 0.0), never raise.
# ---------------------------------------------------------------------------
TRADING_DAYS = 252


def _series(returns):
    return pd.Series(returns).dropna()


def _underwater(returns):
    r = _series(returns)
    if len(r) == 0:
        return pd.Series([], dtype=float)
    growth = pd.concat([pd.Series([1.0]), (1.0 + r).cumprod().reset_index(drop=True)],
                       ignore_index=True)
    peak = growth.cummax()
    uw = (growth - peak) / peak
    return uw.iloc[1:]   # drop the synthetic starting row


def cagr(returns, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return float("nan")
    years = len(r) / periods_per_year
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(returns, mar=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 1:
        return float("nan")
    downside = r[r < mar] - mar
    if len(downside) < 1:
        return float("nan")
    # Divide by total N (semi-deviation), not just downside obs count,
    # so that downside_deviation <= annual_volatility always holds.
    return float(np.sqrt((downside ** 2).sum() / len(r)) * np.sqrt(periods_per_year))


def sortino(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    dd = downside_deviation(r, mar=risk_free_rate / periods_per_year,
                            periods_per_year=periods_per_year)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    excess_ann = (r.mean() - risk_free_rate / periods_per_year) * periods_per_year
    return float(excess_ann / dd)


def max_drawdown(returns):
    uw = _underwater(returns)
    return float(uw.min()) if len(uw) else float("nan")


def avg_drawdown(returns):
    uw = _underwater(returns)
    dd = uw[uw < 0]
    return float(dd.mean()) if len(dd) else 0.0


def ulcer_index(returns):
    uw = _underwater(returns)
    return float(np.sqrt((uw ** 2).mean())) if len(uw) else float("nan")


def calmar(returns, periods_per_year=TRADING_DAYS):
    mdd = max_drawdown(returns)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def value_at_risk(returns, confidence=0.95):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    return float(-r.quantile(1.0 - confidence))


def conditional_var(returns, confidence=0.95):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    threshold = r.quantile(1.0 - confidence)
    tail = r[r <= threshold]
    return float(-tail.mean()) if len(tail) else float("nan")


def omega(returns, threshold=0.0):
    r = _series(returns)
    excess = r - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    if losses == 0:
        return float("nan")
    return float(gains / losses)


def hit_rate(returns):
    r = _series(returns)
    return float((r > 0).mean()) if len(r) else float("nan")


def win_loss_ratio(returns):
    r = _series(returns)
    wins, losses = r[r > 0], r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def tail_ratio(returns):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    left = abs(r.quantile(0.05))
    if left == 0:
        return float("nan")
    return float(abs(r.quantile(0.95)) / left)


def tracking_error(returns, benchmark, periods_per_year=TRADING_DAYS):
    active = (pd.Series(returns) - pd.Series(benchmark)).dropna()
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year=TRADING_DAYS):
    active = (pd.Series(returns) - pd.Series(benchmark)).dropna()
    if len(active) < 2:
        return float("nan")
    sd = active.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def skewness(returns):
    r = _series(returns)
    return float(r.skew()) if len(r) >= 3 else float("nan")


def kurtosis(returns):
    r = _series(returns)
    return float(r.kurtosis()) if len(r) >= 4 else float("nan")


def summary_metrics(returns, benchmark=None, risk_free_rate=0.02,
                    periods_per_year=TRADING_DAYS):
    """Ordered, named metric dict the backtest report consumes."""
    r = _series(returns)
    out = {
        "CAGR": cagr(r, periods_per_year),
        "Volatility": annual_volatility(r, periods_per_year),
        "Sharpe": sharpe(r, risk_free_rate, periods_per_year),
        "Sortino": sortino(r, risk_free_rate, periods_per_year),
        "Calmar": calmar(r, periods_per_year),
        "Max Drawdown": max_drawdown(r),
        "Avg Drawdown": avg_drawdown(r),
        "Ulcer Index": ulcer_index(r),
        "VaR (95%)": value_at_risk(r, 0.95),
        "CVaR (95%)": conditional_var(r, 0.95),
        "Downside Dev": downside_deviation(r, 0.0, periods_per_year),
        "Omega": omega(r, 0.0),
        "Hit Rate": hit_rate(r),
        "Win/Loss": win_loss_ratio(r),
        "Tail Ratio": tail_ratio(r),
        "Skew": skewness(r),
        "Kurtosis": kurtosis(r),
    }
    if benchmark is not None:
        out["Tracking Error"] = tracking_error(r, benchmark, periods_per_year)
        out["Information Ratio"] = information_ratio(r, benchmark, periods_per_year)
    return out

