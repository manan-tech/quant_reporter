"""Shared internal helpers for the report generators.

Consolidates logic that was previously copy-pasted across
``optimization_report.py`` and ``validation_report.py`` so the two reports
cannot drift apart:

* :func:`bundle_from_growth` — build a :class:`ReturnsBundle` from two
  Growth-of-$1 series (portfolio and benchmark).
* :func:`build_standard_constraints` — the unconstrained, balanced (40% cap)
  and sector bounds/constraints both reports set up identically.
* :func:`run_standard_optimizers` — the Equal / Min-Vol / Max-Sharpe /
  Balanced (and optional Sector Balanced) weight *arrays* from a
  ``ReportContext``. Callers map these to their own display names so the
  existing report labels and column order stay unchanged.
* :func:`compute_drawdown_frame` — the one canonical per-strategy drawdown
  path, replacing three divergent inline implementations.
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from .analytics import ReturnsBundle
from .metrics import compute_drawdown
from .opt_core import (
    build_constraints,
    find_optimal_portfolio,
    objective_min_variance,
    objective_neg_sharpe,
)


def bundle_from_growth(strategy_growth, benchmark_growth):
    """Build a ReturnsBundle from two Growth-of-$1 series (portfolio and benchmark)."""
    growth = pd.concat(
        {"Portfolio": strategy_growth, "Benchmark": benchmark_growth}, axis=1
    ).dropna()
    return ReturnsBundle(
        daily=growth.pct_change().dropna(), growth=growth, weights_history=None
    )


StandardConstraints = namedtuple(
    "StandardConstraints",
    ["bounds_uncon", "cons_uncon", "bounds_bal", "cons_bal", "cons_sector"],
)


def build_standard_constraints(ctx):
    """Build the standard bounds/constraints shared by the optimization and
    validation reports.

    Returns a :class:`StandardConstraints` with the unconstrained (0-100%)
    bounds + constraints, the balanced (0-40% cap) bounds + constraints, and
    the sector-aware constraint set (caps/mins from ``ctx``).
    """
    num_assets = len(ctx.tickers)
    return StandardConstraints(
        bounds_uncon=tuple((0, 1) for _ in range(num_assets)),
        cons_uncon=build_constraints(num_assets, ctx.tickers),
        bounds_bal=tuple((0, 0.40) for _ in range(num_assets)),
        cons_bal=build_constraints(num_assets, ctx.tickers),
        cons_sector=build_constraints(
            num_assets, ctx.tickers, ctx.sector_map, ctx.sector_caps, ctx.sector_mins
        ),
    )


def run_standard_optimizers(ctx, constraints=None):
    """Run the standard optimizers on the in-sample inputs in ``ctx`` and return
    raw weight arrays (ordered like ``ctx.friendly_tickers``).

    Returns a dict with keys ``equal``, ``min_vol``, ``max_sharpe`` and
    ``balanced``; the ``sector`` key is present only when a sector map with
    caps or mins is configured (matching the prior guard in both reports).
    Callers map these arrays to their own display names so the existing report
    labels and ordering are preserved.
    """
    if constraints is None:
        constraints = build_standard_constraints(ctx)
    num_assets = len(ctx.tickers)

    arrays = {
        "equal": np.array([1.0 / num_assets] * num_assets),
        "min_vol": find_optimal_portfolio(
            objective_min_variance, ctx.mean_returns, ctx.cov_matrix,
            constraints.bounds_uncon, constraints.cons_uncon, ctx.risk_free_rate),
        "max_sharpe": find_optimal_portfolio(
            objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix,
            constraints.bounds_uncon, constraints.cons_uncon, ctx.risk_free_rate),
        "balanced": find_optimal_portfolio(
            objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix,
            constraints.bounds_bal, constraints.cons_bal, ctx.risk_free_rate),
    }
    if ctx.sector_map and (ctx.sector_caps or ctx.sector_mins):
        arrays["sector"] = find_optimal_portfolio(
            objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix,
            constraints.bounds_uncon, constraints.cons_sector, ctx.risk_free_rate)
    return arrays


def compute_drawdown_frame(growth_df):
    """The canonical per-strategy drawdown path: an underwater curve per column.

    Replaces three divergent inline implementations across the report
    generators with a single :func:`compute_drawdown` call per column,
    preserving the input column order.
    """
    return pd.concat(
        {col: compute_drawdown(growth_df[col]).curve for col in growth_df.columns},
        axis=1,
    )
