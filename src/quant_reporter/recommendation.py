# src/quant_reporter/recommendation.py
"""Recommendation layer (SP4) — the only opinionated surface in quant_reporter.

Opt-in. Four standalone functions (recommend_weights, rebalance_trades,
risk_alerts, compare_verdict) each return a structured object carrying a
human-readable `rationale` and a machine-readable `evidence` dict; `recommend()`
bundles them into a `Recommendation`. SP4 CONSUMES existing primitives
(SP0-SP-Strategy) — it never re-optimizes or re-backtests. Opinions (vol target,
drawdown limit, concentration cap, selection metric) live ONLY here, all
overridable. `prices` are asset prices only (exclude any benchmark column).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .opt_core import (
    get_optimization_inputs, get_portfolio_stats, find_optimal_portfolio,
    build_constraints, get_portfolio_price, risk_contributions,
)
from .objectives import neg_sharpe
from .backtest import portfolio_turnover, transaction_cost_model
from .sizing import forecast_portfolio_vol
from .metrics import compute_drawdown
from .performance_stats import compare_strategies_oos
from .asset_info import compute_asset_factor_exposures


@dataclass
class RecommendedWeights:
    weights: dict
    objective: str
    rationale: str
    evidence: dict = field(default_factory=dict)


def recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None,
                      risk_free_rate=0.02):
    """Point-in-time optimal target weights from a single objective.

    Optimizes `objective` over all columns of `prices` on the canonical
    (get_optimization_inputs) basis. Distinct from compare_verdict's
    backtest-driven pick.
    """
    mean, cov, _ = get_optimization_inputs(prices)
    cols = list(cov.columns)
    n = len(cols)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    if constraints is None:
        constraints = build_constraints(n, cols)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w = find_optimal_portfolio(objective, mean, cov, bounds, constraints, risk_free_rate)
        port_ret, port_vol, sharpe = get_portfolio_stats(w, mean, cov, risk_free_rate)
    weights = {c: float(wi) for c, wi in zip(cols, np.asarray(w, dtype=float))}
    obj_name = getattr(objective, "__name__", str(objective))
    rationale = (f"Weights chosen by optimizing the {obj_name} objective; resulting "
                 f"Sharpe {sharpe:.2f} (return {port_ret:.2%}, vol {port_vol:.2%}).")
    evidence = {"objective": obj_name, "sharpe": float(sharpe),
                "expected_return": float(port_ret), "expected_vol": float(port_vol)}
    return RecommendedWeights(weights=weights, objective=obj_name,
                              rationale=rationale, evidence=evidence)


@dataclass
class Trade:
    ticker: str
    side: str                 # 'buy' | 'sell'
    current_weight: float
    target_weight: float
    delta: float              # target - current
    rationale: str
    evidence: dict = field(default_factory=dict)


@dataclass
class RebalancePlan:
    orders: list              # list[Trade]
    turnover: float           # one-way
    est_cost: float           # cost_frac on executed deltas
    held: list                # tickers inside the no-trade band
    rationale: str
    evidence: dict = field(default_factory=dict)


def rebalance_trades(current_weights, target_weights, *, cost_model=None, threshold=0.0):
    """Trade list from current -> target weights, with a no-trade band, turnover,
    and an estimated cost on the executed deltas."""
    to = portfolio_turnover(current_weights, target_weights, convention="one_way")
    deltas = to["trades"]     # signed Series, union-aligned (missing -> 0)
    cur = pd.Series(current_weights, dtype=float).reindex(deltas.index).fillna(0.0)
    tgt = pd.Series(target_weights, dtype=float).reindex(deltas.index).fillna(0.0)

    orders, held, executed = [], [], {}
    for tk in deltas.index:
        d = float(deltas[tk])
        if abs(d) == 0:
            continue                      # no change for this ticker
        if abs(d) < threshold:
            held.append(str(tk))          # inside the no-trade band
            continue
        executed[tk] = d
        side = "buy" if d > 0 else "sell"
        orders.append(Trade(
            ticker=str(tk), side=side,
            current_weight=float(cur[tk]), target_weight=float(tgt[tk]), delta=d,
            rationale=f"{side.title()} {tk}: {cur[tk]:.2%} -> {tgt[tk]:.2%} (delta {d:+.2%}).",
            evidence={"abs_delta": abs(d), "threshold": threshold},
        ))

    cost_fn = cost_model or transaction_cost_model
    est_cost = float(cost_fn(pd.Series(executed, dtype=float))["cost_frac"])
    turnover = float(to["turnover"])
    rationale = (f"{len(orders)} order(s); one-way turnover {turnover:.2%}, "
                 f"est. cost {est_cost * 1e4:.1f} bps." +
                 (f" {len(held)} position(s) held inside the {threshold:.2%} band." if held else ""))
    return RebalancePlan(
        orders=orders, turnover=turnover, est_cost=est_cost, held=held,
        rationale=rationale,
        evidence={"turnover": turnover, "est_cost": est_cost,
                  "n_orders": len(orders), "n_held": len(held), "threshold": threshold},
    )
