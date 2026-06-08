"""Canonical analytics core — the single source of truth for portfolio returns,
growth, drawdown, and realized metrics. Pure functions (standalone) + a memoized
PortfolioAnalytics accessor attached to ReportContext as ctx.analytics."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .rebalancing import simulate_rebalanced_portfolio
from .metrics import compute_drawdown, calculate_var_cvar
from .metrics import sortino as sortino_ratio


@dataclass(frozen=True)
class ReturnsBundle:
    daily: pd.DataFrame                 # ['Portfolio', 'Benchmark'] simple daily returns
    growth: pd.DataFrame               # ['Portfolio', 'Benchmark'] Growth-of-$1 (start 1.0)
    weights_history: "pd.DataFrame | None"

    @property
    def terminal(self) -> float:
        return float(self.growth["Portfolio"].iloc[-1] - 1.0)


def portfolio_returns(price_data, weights_dict, benchmark_col, rebalance_freq=None):
    """Single producer of the portfolio Growth-of-$1 and daily returns.

    Routes through simulate_rebalanced_portfolio for ALL frequencies;
    rebalance_freq=None is buy-and-hold (matches the closed-form get_portfolio_price).

    weights_dict is treated as proportions of a unit portfolio (assumed to sum to 1);
    the buy-and-hold/closed-form equivalence holds for normalized weights.
    """
    asset_cols = [c for c in weights_dict if c in price_data.columns]
    asset_prices = price_data[asset_cols]
    sub_weights = {k: weights_dict[k] for k in asset_cols}

    wealth, weights_history = simulate_rebalanced_portfolio(asset_prices, sub_weights, rebalance_freq)
    wealth = wealth.rename("Portfolio")

    bench = price_data[benchmark_col]
    bench_growth = (bench / bench.iloc[0]).rename("Benchmark")

    growth = pd.concat([wealth, bench_growth], axis=1).dropna()
    daily = growth.pct_change().dropna()
    return ReturnsBundle(daily=daily, growth=growth, weights_history=weights_history)


_PCT_KEYS = {
    "Realized CAGR", "Realized Volatility", "Alpha (CAPM, ann.)",
    "Max Drawdown", "VaR (95%, daily)", "CVaR (95%, daily)",
}


def compute_metrics(bundle, risk_free_rate):
    """The REALIZED metrics block (numeric) computed once from a ReturnsBundle."""
    pr = bundle.daily["Portfolio"]
    br = bundle.daily["Benchmark"]
    growth = bundle.growth["Portfolio"]
    ann = np.sqrt(252)

    vol = float(pr.std() * ann)
    excess = pr - risk_free_rate / 252
    ex_std = excess.std()
    sharpe = float((excess.mean() * 252) / (ex_std * ann)) if (np.isfinite(ex_std) and ex_std > 0) else 0.0
    # Canonical Sortino: semi-deviation downside (MAR = risk-free rate), shared
    # with the strategy/metrics library so every surface reports one definition.
    sortino = float(sortino_ratio(pr, risk_free_rate=risk_free_rate))
    max_dd = compute_drawdown(growth).max_dd

    n_years = max((growth.index[-1] - growth.index[0]).days / 365.25, 1)
    cagr = float(growth.iloc[-1] ** (1 / n_years) - 1)
    calmar = float(cagr / abs(max_dd)) if max_dd else float("nan")

    if not (np.isfinite(br.std()) and np.isfinite(pr.std())) or br.std() == 0 or pr.std() == 0:
        beta, alpha = 0.0, 0.0
    else:
        lr = stats.linregress(br, pr)
        beta, alpha = float(lr.slope), float(lr.intercept * 252)

    var95, cvar95 = calculate_var_cvar(pr, 0.95)

    return {
        "Realized CAGR": cagr,
        "Realized Volatility": vol,
        "Realized Sharpe": sharpe,
        "Realized Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Beta (CAPM)": beta,
        "Alpha (CAPM, ann.)": alpha,
        "Skew": float(pr.skew()),
        "Kurtosis": float(pr.kurtosis()),
        "VaR (95%, daily)": float(var95),
        "CVaR (95%, daily)": float(cvar95),
    }


def format_metrics(metrics):
    """Display formatter: % for rate-like keys, 2dp otherwise."""
    return {k: (f"{v:.2%}" if k in _PCT_KEYS else f"{v:.2f}") for k, v in metrics.items()}


class PortfolioAnalytics:
    """Lazy, memoized accessor attached as ctx.analytics. Delegates to the pure
    functions and caches, so every report section reads identical, compute-once values."""

    def __init__(self, ctx):
        self._ctx = ctx

    @cached_property
    def returns(self) -> ReturnsBundle:
        c = self._ctx
        return portfolio_returns(
            c.price_data_full, c.user_friendly_weights, c.friendly_benchmark,
            getattr(c, "rebalance_freq", None),
        )

    @cached_property
    def drawdown(self):
        return compute_drawdown(self.returns.growth["Portfolio"])

    @cached_property
    def metrics(self):
        return compute_metrics(self.returns, self._ctx.risk_free_rate)

    @cached_property
    def model_stats(self):
        from .opt_core import get_portfolio_stats
        c = self._ctx
        w = pd.Series(c.user_friendly_weights).reindex(c.mean_returns.index).fillna(0.0).values
        ret, vol, sharpe = get_portfolio_stats(w, c.mean_returns, c.cov_matrix, c.risk_free_rate)
        return {"Expected Return": float(ret), "Expected Volatility": float(vol), "Expected Sharpe": float(sharpe)}
