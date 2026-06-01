# src/quant_reporter/strategy.py
"""Strategy abstraction + backtest runner + result (SP-Strategy).

A strategy is any callable (prices, **params) -> weights (dict or DataFrame),
a Strategy wrapper, or raw weights. backtest() is a thin orchestration over the
SP1 simulate_strategy engine (which it does NOT replace).
"""
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .backtest import simulate_strategy
from .metrics import summary_metrics
from .performance_stats import probabilistic_sharpe_ratio, deflated_sharpe_ratio


@dataclass
class Strategy:
    name: str
    fn: Callable
    params: dict = field(default_factory=dict)

    def weights(self, prices):
        return self.fn(prices, **self.params)


def _resolve_weights(strategy, prices):
    if isinstance(strategy, Strategy):
        return strategy.weights(prices), strategy.name
    if isinstance(strategy, (dict, pd.DataFrame)):
        return strategy, "custom"
    if callable(strategy):
        return strategy(prices), getattr(strategy, "__name__", "strategy")
    raise TypeError(f"Unsupported strategy type: {type(strategy)!r}")


@dataclass
class BacktestResult:
    """Backtest output. Treat as immutable after construction: `returns`,
    `metrics`, and `oos_stats` are cached on first access, so mutating the
    underlying fields afterward yields stale derived values."""
    name: str
    wealth: pd.Series
    weights: pd.DataFrame
    blotter: pd.DataFrame
    turnover: pd.Series
    cost_drag: float
    benchmark: Optional[pd.Series]
    summary: dict
    n_trials: int = 1

    @cached_property
    def returns(self) -> pd.Series:
        return self.wealth.pct_change(fill_method=None).dropna()

    @cached_property
    def _benchmark_returns(self):
        if self.benchmark is None:
            return None
        return self.benchmark.pct_change(fill_method=None).dropna()

    @cached_property
    def metrics(self) -> dict:
        return summary_metrics(self.returns, benchmark=self._benchmark_returns)

    @cached_property
    def oos_stats(self) -> dict:
        r = self.returns
        return {
            "psr": probabilistic_sharpe_ratio(r),
            "dsr": deflated_sharpe_ratio(r, n_trials=self.n_trials),
        }

    def report(self, path="backtest_report.html", open_browser=False, recommendation=None):
        from .backtest_report import create_backtest_report
        return create_backtest_report(self, path=path, open_browser=open_browser,
                                      recommendation=recommendation)

    def plot_wealth(self):
        from .backtest_report import plot_wealth
        return plot_wealth(self)

    def plot_drawdown(self):
        from .backtest_report import plot_drawdown
        return plot_drawdown(self)

    def plot_weights(self):
        from .backtest_report import plot_weights
        return plot_weights(self)

    def plot_turnover(self):
        from .backtest_report import plot_turnover
        return plot_turnover(self)

    def plot_rolling(self):
        from .backtest_report import plot_rolling
        return plot_rolling(self)


def backtest(strategy, prices, *, rebalance="M", cost_model=None, benchmark=None,
             initial_value=1.0, name=None, n_trials=1) -> BacktestResult:
    bench_series = None
    asset_prices = prices
    if benchmark is not None:
        if isinstance(benchmark, str):
            if benchmark not in prices.columns:
                raise KeyError(
                    f"benchmark column {benchmark!r} not found in prices. "
                    f"Available columns: {list(prices.columns)}"
                )
            b = prices[benchmark]
            bench_series = (b / b.iloc[0]).rename("Benchmark")
            asset_prices = prices.drop(columns=[benchmark])
        else:
            b = pd.Series(benchmark)
            bench_series = (b / b.iloc[0]).rename("Benchmark")

    weights, resolved_name = _resolve_weights(strategy, asset_prices)
    sim = simulate_strategy(asset_prices, weights, cost_model=cost_model,
                            rebalance=rebalance, initial_value=initial_value)

    if bench_series is not None:
        bench_series = bench_series.reindex(sim["wealth"].index).ffill()
        if not bench_series.notna().any():
            warnings.warn(
                "benchmark has no overlap with the simulation window; benchmark "
                "metrics (tracking error, information ratio) will be NaN. Pass a "
                "Series indexed like `prices` (DatetimeIndex) or a benchmark column name.",
                stacklevel=2,
            )

    return BacktestResult(
        name=name or resolved_name,
        wealth=sim["wealth"],
        weights=sim["weights"],
        blotter=pd.DataFrame(sim["blotter"]),
        turnover=sim["turnover"],
        cost_drag=sim["cost_drag"],
        benchmark=bench_series,
        summary=sim["summary"],
        n_trials=n_trials,
    )


def backtest_many(strategies, prices, **kwargs) -> dict:
    """Backtest several strategies. Sets n_trials = number of strategies on every
    result (for DSR deflation). Any `name` or `n_trials` passed in kwargs are
    overridden. Other kwargs (benchmark, cost_model, rebalance, ...) pass through."""
    kwargs.pop("name", None)
    kwargs.pop("n_trials", None)
    n = len(strategies)
    return {nm: backtest(strat, prices, name=nm, n_trials=n, **kwargs)
            for nm, strat in strategies.items()}
