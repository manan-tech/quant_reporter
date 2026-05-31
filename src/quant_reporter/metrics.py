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

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.02):
    """Calculates the annualized Sortino Ratio."""
    annualized_mean_return = daily_returns.mean() * 252
    
    downside_returns = daily_returns[daily_returns < 0]
    annualized_downside_std = downside_returns.std() * np.sqrt(252)
    
    if annualized_downside_std == 0:
        return np.nan
        
    sortino_ratio = (annualized_mean_return - risk_free_rate) / annualized_downside_std
    return sortino_ratio

def calculate_var_cvar(daily_returns, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    if daily_returns.empty:
        return np.nan, np.nan
    var = daily_returns.quantile(1 - confidence_level)
    cvar = daily_returns[daily_returns <= var].mean()
    
    return var, cvar

