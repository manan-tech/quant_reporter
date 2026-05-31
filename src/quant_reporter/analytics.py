"""Canonical analytics core — the single source of truth for portfolio returns,
growth, drawdown, and realized metrics. Pure functions (standalone) + a memoized
PortfolioAnalytics accessor attached to ReportContext as ctx.analytics."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .rebalancing import simulate_rebalanced_portfolio
from .metrics import compute_drawdown, calculate_sortino_ratio, calculate_var_cvar


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
