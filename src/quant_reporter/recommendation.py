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
    objective_min_variance,
)
from .advanced_optimizers import optimize_risk_parity, optimize_max_diversification
from .robust_estimators import ledoit_wolf_covariance
from .objectives import neg_sharpe
from .backtest import portfolio_turnover, transaction_cost_model
from .sizing import forecast_portfolio_vol
from .metrics import compute_drawdown
from .analytics import ReturnsBundle, compute_metrics
from .performance_stats import compare_strategies_oos
from .asset_info import compute_asset_factor_exposures
from .planning import check_suitability, SuitabilityReport


@dataclass
class RecommendedWeights:
    weights: dict
    objective: str
    rationale: str
    evidence: dict = field(default_factory=dict)


COV_METHODS = ("sample", "ledoit_wolf", "denoise")


def _covariance_inputs(prices, cov_method="sample", n_components=3):
    """Optimization inputs (mean, cov, log_returns) for the chosen covariance
    estimator, plus an ``info`` dict of estimator diagnostics for `evidence`.

    - ``"sample"``      — the raw sample covariance (current/default behavior).
    - ``"ledoit_wolf"`` — Ledoit-Wolf constant-correlation shrinkage
      (:func:`robust_estimators.ledoit_wolf_covariance`); the mean is unchanged.
    - ``"denoise"``     — eigenvalue-clipping denoised sample covariance
      (:func:`opt_core.get_optimization_inputs` with ``denoise_cov=True``).

    Only the covariance is affected; expected returns stay the raw annualized
    mean (issue #6 targets the covariance; the return estimate is issue #7/#8).
    """
    if cov_method == "sample":
        mean, cov, log_returns = get_optimization_inputs(prices)
        return mean, cov, log_returns, {}
    if cov_method == "denoise":
        mean, cov, log_returns = get_optimization_inputs(
            prices, denoise_cov=True, n_components=n_components)
        return mean, cov, log_returns, {"n_components": n_components}
    if cov_method == "ledoit_wolf":
        mean, _, log_returns = get_optimization_inputs(prices)
        lw = ledoit_wolf_covariance(log_returns)
        return mean, lw["cov_matrix"], log_returns, {
            "shrinkage": float(lw["shrinkage"]), "target": lw["target"]}
    raise ValueError(
        f"Unknown cov_method {cov_method!r}; expected one of {COV_METHODS}.")


# Allocators that ignore expected returns entirely (the "we don't forecast
# returns" stance). "optimize" (the default) is the objective-based path.
NO_FORECAST_METHODS = ("min_variance", "risk_parity", "max_diversification")
METHODS = ("optimize",) + NO_FORECAST_METHODS


def _optimize_weights(method, objective, mean, cov, bounds, constraints, risk_free_rate):
    """Weight vector for the chosen allocation `method`.

    ``"optimize"`` runs the objective-based optimizer (uses expected returns via
    `objective`). The no-forecast methods use only the covariance: ``"min_variance"``
    honors `bounds`/`constraints` through the shared SLSQP path, while
    ``"risk_parity"`` and ``"max_diversification"`` use their own risk-based
    allocation and therefore do NOT honor `bounds`/`constraints`/`profile`.
    """
    if method == "optimize":
        return find_optimal_portfolio(objective, mean, cov, bounds, constraints, risk_free_rate)
    if method == "min_variance":
        return find_optimal_portfolio(objective_min_variance, mean, cov, bounds,
                                      constraints, risk_free_rate)
    if method == "risk_parity":
        return optimize_risk_parity(cov)
    if method == "max_diversification":
        return optimize_max_diversification(cov)
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")


def recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None,
                      profile=None, sector_map=None, risk_free_rate=0.02,
                      cov_method="sample", n_components=3, method="optimize"):
    """Point-in-time optimal target weights.

    `method` selects the allocator (default ``"optimize"`` — the objective-based
    path that uses expected returns). The **no-forecast** methods ``"min_variance"``,
    ``"risk_parity"``, and ``"max_diversification"`` allocate from the covariance
    alone, sidestepping the fragile expected-return estimate. `objective` applies
    only to ``"optimize"``. When `profile` is given and `bounds`/`constraints` are
    not supplied explicitly, they are derived from the profile via
    planning.apply_constraints (explicit bounds/constraints always win); note that
    ``"risk_parity"``/``"max_diversification"`` do not honor those constraints.

    `cov_method` selects the covariance estimator fed to the optimizer
    (``"sample"`` (default), ``"ledoit_wolf"``, or ``"denoise"``); `n_components`
    is the eigenvalue-clipping cutoff used only by ``"denoise"``. Both switches are
    opt-in and preserve prior behavior — see :func:`_covariance_inputs`.
    """
    mean, cov, _, cov_info = _covariance_inputs(prices, cov_method, n_components)
    cols = list(cov.columns)
    n = len(cols)
    if profile is not None and bounds is None and constraints is None:
        from .planning import apply_constraints
        bounds, constraints = apply_constraints(profile, cols, sector_map=sector_map)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    if constraints is None:
        constraints = build_constraints(n, cols)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        w = _optimize_weights(method, objective, mean, cov, bounds, constraints, risk_free_rate)
        port_ret, port_vol, sharpe = get_portfolio_stats(w, mean, cov, risk_free_rate)
    weights = {c: float(wi) for c, wi in zip(cols, np.asarray(w, dtype=float))}
    obj_name = getattr(objective, "__name__", str(objective))
    no_forecast = method in NO_FORECAST_METHODS
    display_name = method if no_forecast else obj_name

    if no_forecast:
        rationale = (f"No-forecast allocation via {method}: expected returns are not "
                     f"forecast; weights come from the {cov_method} covariance alone. "
                     f"Resulting Sharpe {sharpe:.2f} (return {port_ret:.2%}, "
                     f"vol {port_vol:.2%}; annualized log-return basis).")
    else:
        rationale = (f"Weights chosen by optimizing the {obj_name} objective on the "
                     f"{cov_method} covariance; resulting Sharpe {sharpe:.2f} "
                     f"(return {port_ret:.2%}, vol {port_vol:.2%}; "
                     f"annualized log-return basis).")
    if "shrinkage" in cov_info:
        rationale += f" Ledoit-Wolf shrinkage delta={cov_info['shrinkage']:.2f}."

    evidence = {"objective": display_name, "method": method,
                "uses_return_forecast": not no_forecast, "sharpe": float(sharpe),
                "expected_return": float(port_ret), "expected_vol": float(port_vol),
                "basis": "annualized_log_return", "cov_method": cov_method}
    if cov_info:
        evidence["cov_info"] = cov_info
    return RecommendedWeights(weights=weights, objective=display_name,
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
        if abs(d) <= 1e-12:
            continue                      # no change (incl. float-dust deltas)
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


@dataclass
class RiskAlert:
    kind: str                 # 'vol_breach'|'drawdown_breach'|'concentration'|'sector_cap'|'factor_drift'
    severity: str             # 'warning' | 'breach'  ('ok' checks emit nothing)
    rationale: str
    evidence: dict = field(default_factory=dict)


def _severity(value, limit, warn_band=0.9):
    if value > limit:
        return "breach"
    if value >= warn_band * limit:
        return "warning"
    return "ok"


def risk_alerts(weights, prices, *, vol_target=0.10, max_drawdown_limit=0.20,
                max_weight=0.40, max_risk_contribution=0.40,
                sector_map=None, sector_caps=None,
                factor_returns=None, factor_loading_limit=None):
    """Limit-breach checks on a weight vector. Returns a (possibly empty) list of
    RiskAlert; only 'warning'/'breach' checks emit. `weights` tickers must be a
    subset of `prices` columns. Opinions (limits) are overridable params."""
    w = pd.Series(weights, dtype=float)
    missing = [t for t in w.index if t not in prices.columns]
    if missing:
        raise ValueError(f"weights contain tickers not in prices columns: {missing}")
    asset_prices = prices[list(w.index)]
    alerts = []

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        _, cov, _ = get_optimization_inputs(asset_prices)

        # 1. forecast vol vs target
        fvol = forecast_portfolio_vol(w.to_dict(), cov)
        sev = _severity(fvol, vol_target)
        if sev != "ok":
            alerts.append(RiskAlert("vol_breach", sev,
                f"Forecast annualized vol {fvol:.2%} vs target {vol_target:.2%}.",
                {"metric": "forecast_vol", "value": float(fvol),
                 "threshold": float(vol_target), "comparator": ">"}))

        # 2. historical max drawdown of holding `weights`
        wealth = get_portfolio_price(asset_prices, w.to_dict())
        mdd = float(compute_drawdown(wealth).max_dd)   # negative
        sev = _severity(abs(mdd), max_drawdown_limit)
        if sev != "ok":
            alerts.append(RiskAlert("drawdown_breach", sev,
                f"Historical max drawdown {mdd:.2%} vs limit {-max_drawdown_limit:.2%}.",
                {"metric": "max_drawdown", "value": mdd,
                 "threshold": -float(max_drawdown_limit), "comparator": "<"}))

        # 3a. single-name weight concentration
        max_w = float(w.abs().max())
        top_w = str(w.abs().idxmax())
        sev = _severity(max_w, max_weight)
        if sev != "ok":
            alerts.append(RiskAlert("concentration", sev,
                f"Largest position {top_w} at {max_w:.2%} vs cap {max_weight:.2%}.",
                {"metric": "max_weight", "asset": top_w, "value": max_w,
                 "threshold": float(max_weight), "comparator": ">"}))

        # 3b. risk-contribution concentration
        rc = risk_contributions(w.to_dict(), cov)
        if len(rc) and rc.notna().any() and float(rc.max()) > 0:
            max_rc = float(rc.max())
            top_rc = str(rc.idxmax())
            sev = _severity(max_rc, max_risk_contribution)
            if sev != "ok":
                alerts.append(RiskAlert("concentration", sev,
                    f"{top_rc} drives {max_rc:.2%} of portfolio risk vs cap {max_risk_contribution:.2%}.",
                    {"metric": "risk_contribution", "asset": top_rc, "value": max_rc,
                     "threshold": float(max_risk_contribution), "comparator": ">"}))

    # 4. sector caps (only when both provided)
    if sector_map is not None and sector_caps is not None:
        sector_w = {}
        for tk, wi in w.items():
            sec = sector_map.get(tk)
            if sec is not None:
                sector_w[sec] = sector_w.get(sec, 0.0) + float(wi)
        for sec, cap in sector_caps.items():
            val = sector_w.get(sec, 0.0)
            sev = _severity(val, cap)
            if sev != "ok":
                alerts.append(RiskAlert("sector_cap", sev,
                    f"Sector '{sec}' at {val:.2%} vs cap {cap:.2%}.",
                    {"metric": "sector_weight", "sector": sec, "value": val,
                     "threshold": float(cap), "comparator": ">"}))

    # 5. factor drift (only when factor_returns AND a limit are provided)
    if factor_returns is not None and factor_loading_limit is not None:
        rets = asset_prices.pct_change(fill_method=None).dropna()
        betas = compute_asset_factor_exposures(rets, factor_returns)   # N x K
        aligned_w = w.reindex(betas.index).fillna(0.0)
        port_load = betas.mul(aligned_w, axis=0).sum(axis=0)           # per-factor
        for fac, load in port_load.items():
            sev = _severity(abs(float(load)), factor_loading_limit)
            if sev != "ok":
                alerts.append(RiskAlert("factor_drift", sev,
                    f"Factor '{fac}' loading {float(load):.2f} exceeds limit "
                    f"+/-{factor_loading_limit:.2f}.",
                    {"metric": "factor_loading", "factor": str(fac), "value": float(load),
                     "threshold": float(factor_loading_limit), "comparator": "abs>"}))

    return alerts


@dataclass
class StrategyVerdict:
    winner: Optional[str]
    ranking: list             # ordered [{'name','sharpe','psr','dsr'}, ...]
    rationale: str
    evidence: dict = field(default_factory=dict)


def compare_verdict(results, *, select_by="dsr", benchmark=None):
    """Rank a dict[str, BacktestResult] by deflated Sharpe (consumes them; does
    not re-backtest). `benchmark` is an optional periodic-returns Series used only
    for 'vs Benchmark' p-values."""
    if not results:
        return StrategyVerdict(None, [], "No strategies to compare.", {"summary": {}})
    returns_dict = {nm: res.returns for nm, res in results.items()}
    bench_returns = None
    if benchmark is not None:
        bench_returns = benchmark.dropna() if isinstance(benchmark, pd.Series) else pd.Series(benchmark)
    cmp = compare_strategies_oos(returns_dict, benchmark_returns=bench_returns,
                                 n_trials=len(results))
    summary = cmp["summary"]

    def _key(name):
        v = summary[name].get(select_by, float("nan"))
        return v if np.isfinite(v) else float("-inf")

    if select_by == "dsr":
        winner = cmp["best_by_dsr"]
    else:
        usable = [k for k in summary if np.isfinite(summary[k].get(select_by, float("nan")))]
        winner = max(usable, key=_key) if usable else None
    ranking = sorted(({"name": nm, **summary[nm]} for nm in summary),
                     key=lambda d: _key(d["name"]), reverse=True)

    if winner is None:
        ranking = []  # no well-defined metric to order by; keep ranking[0]==winner invariant
        rationale = "No strategy had a well-defined selection metric; no winner."
    elif len(results) == 1:
        rationale = f"Only one strategy ('{winner}'); no comparison performed."
    else:
        wm = summary[winner]
        rationale = (f"'{winner}' wins by {select_by.upper()} (DSR {wm['dsr']:.2f}, "
                     f"PSR {wm['psr']:.2f}, Sharpe {wm['sharpe']:.2f}) across "
                     f"{len(results)} strategies.")
    return StrategyVerdict(
        winner=winner, ranking=ranking, rationale=rationale,
        evidence={"select_by": select_by, "summary": summary,
                  "pvalues": cmp["sharpe_diff_pvalues"], "n_trials": len(results)})


@dataclass
class RecommendationValidation:
    in_sample_sharpe: float
    oos_sharpe: float
    degradation: float
    n_windows: int
    verdict: str              # 'holds up' | 'fragile (overfit)' | 'inconclusive'
    rationale: str
    baseline_oos_sharpe: Optional[float] = None
    per_window: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "in_sample_sharpe": self.in_sample_sharpe,
            "oos_sharpe": self.oos_sharpe,
            "degradation": self.degradation,
            "n_windows": self.n_windows,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "baseline_oos_sharpe": self.baseline_oos_sharpe,
            "per_window": self.per_window,
            "evidence": self.evidence,
        }

    def to_text(self):
        lines = [f"Validation (walk-forward): {self.verdict.upper()}",
                 f"  {self.rationale}",
                 f"  in-sample Sharpe {self.in_sample_sharpe:.2f} | "
                 f"OOS Sharpe {self.oos_sharpe:.2f} | {self.n_windows} windows"]
        if self.baseline_oos_sharpe is not None:
            lines.append(f"  current portfolio OOS Sharpe {self.baseline_oos_sharpe:.2f}")
        return "\n".join(lines)


@dataclass
class Recommendation:
    target_weights: RecommendedWeights
    trades: Optional[RebalancePlan]
    alerts: list              # list[RiskAlert]
    verdict: Optional[StrategyVerdict]
    suitability: Optional[SuitabilityReport] = None
    validation: Optional[RecommendationValidation] = None

    def to_dict(self):
        return {
            "target_weights": {
                "weights": self.target_weights.weights,
                "objective": self.target_weights.objective,
                "rationale": self.target_weights.rationale,
                "evidence": self.target_weights.evidence,
            },
            "trades": None if self.trades is None else {
                "orders": [vars(o) for o in self.trades.orders],
                "turnover": self.trades.turnover, "est_cost": self.trades.est_cost,
                "held": self.trades.held, "rationale": self.trades.rationale,
                "evidence": self.trades.evidence,
            },
            "alerts": [vars(a) for a in self.alerts],
            "verdict": None if self.verdict is None else {
                "winner": self.verdict.winner, "ranking": self.verdict.ranking,
                "rationale": self.verdict.rationale, "evidence": self.verdict.evidence,
            },
            "suitability": None if self.suitability is None else self.suitability.to_dict(),
            "validation": None if self.validation is None else self.validation.to_dict(),
        }

    def to_text(self):
        lines = ["RECOMMENDATION", "=" * 40,
                 f"Target weights ({self.target_weights.objective}):"]
        for tk, wt in self.target_weights.weights.items():
            lines.append(f"  {tk:8s} {wt:7.2%}")
        lines.append(f"  -> {self.target_weights.rationale}")
        if self.trades is not None:
            lines += ["", f"Rebalance: {self.trades.rationale}"]
            for o in self.trades.orders:
                lines.append(f"  {o.side.upper():4s} {o.ticker:8s} "
                             f"{o.current_weight:7.2%} -> {o.target_weight:7.2%}")
        lines.append("")
        if self.alerts:
            lines.append("Risk alerts:")
            for a in self.alerts:
                lines.append(f"  [{a.severity.upper()}] {a.kind}: {a.rationale}")
        else:
            lines.append("Risk alerts: none")
        if self.verdict is not None:
            lines += ["", f"Verdict: {self.verdict.rationale}"]
        if self.suitability is not None:
            lines += ["", self.suitability.to_text()]
        if self.validation is not None:
            lines += ["", self.validation.to_text()]
        return "\n".join(lines)

    def to_html(self, path=None, open_browser=False):
        from .recommendation_report import create_recommendation_report
        return create_recommendation_report(self, path=path, open_browser=open_browser)


def recommend(prices, *, current_weights=None, objective=neg_sharpe, results=None,
              cost_model=None, threshold=0.0, profile=None, vol_target=None,
              max_drawdown_limit=None, max_weight=None, max_risk_contribution=0.40,
              sector_map=None, sector_caps=None, factor_returns=None,
              factor_loading_limit=None, risk_free_rate=0.02,
              validate=False, window_years=1, step_months=3, max_degradation=0.5,
              cov_method="sample", n_components=3, method="optimize"):
    """Opt-in recommendation bundle. `prices` are asset prices only. Alerts run on
    `current_weights` when given, else on the recommended target.

    `method` (allocator) and `cov_method`/`n_components` (covariance estimator)
    select how the recommended weights — and, when `validate=True`, the
    walk-forward scoreboard — are computed; the defaults (``"optimize"`` /
    ``"sample"``) preserve prior behavior.

    When `profile` is supplied: it constrains the optimizer (via recommend_weights),
    fills the alert thresholds (vol_target, max_drawdown_limit, max_weight,
    sector_caps) unless those are passed explicitly, and a SuitabilityReport is
    attached. `profile=None` reproduces the legacy behavior exactly.
    """
    if profile is not None:
        if vol_target is None:
            vol_target = profile.max_volatility
        if max_drawdown_limit is None:
            max_drawdown_limit = profile.max_drawdown_tolerance
        if max_weight is None:
            max_weight = profile.max_position_weight
        if sector_caps is None:
            sector_caps = profile.sector_caps
    # historical defaults when still unset (legacy behavior)
    if vol_target is None:
        vol_target = 0.10
    if max_drawdown_limit is None:
        max_drawdown_limit = 0.20
    if max_weight is None:
        max_weight = 0.40

    target = recommend_weights(prices, objective=objective, profile=profile,
                               sector_map=sector_map, risk_free_rate=risk_free_rate,
                               cov_method=cov_method, n_components=n_components,
                               method=method)
    trades = None
    if current_weights is not None:
        trades = rebalance_trades(current_weights, target.weights,
                                  cost_model=cost_model, threshold=threshold)
    alert_weights = current_weights if current_weights is not None else target.weights
    alerts = risk_alerts(alert_weights, prices, vol_target=vol_target,
                         max_drawdown_limit=max_drawdown_limit, max_weight=max_weight,
                         max_risk_contribution=max_risk_contribution,
                         sector_map=sector_map, sector_caps=sector_caps,
                         factor_returns=factor_returns,
                         factor_loading_limit=factor_loading_limit)
    verdict = compare_verdict(results) if results else None
    suitability = None
    if profile is not None:
        suitability = check_suitability(target, profile, prices=prices,
                                        sector_map=sector_map)
    validation = None
    if validate:
        validation = walk_forward_recommendation(
            prices, objective=objective, profile=profile,
            current_weights=current_weights, sector_map=sector_map,
            window_years=window_years, step_months=step_months,
            risk_free_rate=risk_free_rate, max_degradation=max_degradation,
            cov_method=cov_method, n_components=n_components, method=method)
    return Recommendation(target_weights=target, trades=trades, alerts=alerts,
                          verdict=verdict, suitability=suitability,
                          validation=validation)


def _realized_metrics(prices, weights, risk_free_rate):
    """Realized metrics of `weights` held over `prices` (benchmark = self, so
    benchmark-relative fields are inert; we read Realized Sharpe/CAGR)."""
    growth = get_portfolio_price(prices, weights)
    g = pd.concat({"Portfolio": growth, "Benchmark": growth}, axis=1).dropna()
    bundle = ReturnsBundle(daily=g.pct_change().dropna(), growth=g, weights_history=None)
    return compute_metrics(bundle, risk_free_rate)


def walk_forward_recommendation(prices, *, objective=neg_sharpe, profile=None,
                                current_weights=None, sector_map=None,
                                window_years=1, step_months=3,
                                risk_free_rate=0.02, max_degradation=0.5,
                                cov_method="sample", n_components=3,
                                method="optimize"):
    """Walk-forward OOS validation of the recommendation on raw asset prices.

    At each rolling window the constrained recommendation is re-derived on the
    train slice (honoring `objective` + `profile`) and applied forward; the user's
    `current_weights` (if given) are validated as an OOS baseline. Returns a
    RecommendationValidation; verdict is degradation-based.

    `method` and `cov_method`/`n_components` select the same allocator and
    covariance estimator as :func:`recommend_weights`, so the OOS scoreboard
    reflects what the recommendation would actually use. The defaults
    (``"optimize"`` / ``"sample"``) reproduce the prior behavior exactly.
    """
    from .validation_report import _rolling_oos_sharpe, calculate_overfitting_score

    cols = list(prices.columns)
    n = len(cols)
    obj = objective or neg_sharpe
    if profile is not None:
        from .planning import apply_constraints
        r_bounds, r_cons = apply_constraints(profile, cols, sector_map=sector_map)
    else:
        r_bounds = tuple((0.0, 1.0) for _ in range(n))
        r_cons = build_constraints(n, cols)

    def _rec(train, mean, cov):
        # For robust estimators, re-derive (mean, cov) from the train slice; the
        # sample path uses the inputs _rolling_oos_sharpe already computed so the
        # default is byte-for-byte unchanged.
        if cov_method != "sample":
            mean, cov, _, _ = _covariance_inputs(train[cols], cov_method, n_components)
        arr = _optimize_weights(method, obj, mean, cov, r_bounds, r_cons, risk_free_rate)
        return {t: float(w) for t, w in zip(cols, arr)}

    strategies = {"Recommended": _rec}
    if current_weights is not None:
        def _cur(train, mean, cov):
            return current_weights
        strategies["Current"] = _cur

    rolling_df = _rolling_oos_sharpe(
        prices, cols, strategies, window_years=window_years, step_months=step_months,
        risk_free_rate=risk_free_rate, benchmark_col=None)

    n_windows = len(rolling_df)
    if n_windows < 1:
        return RecommendationValidation(
            in_sample_sharpe=float("nan"), oos_sharpe=float("nan"),
            degradation=float("nan"), n_windows=0, verdict="inconclusive",
            rationale="Not enough data for a single walk-forward window.",
            baseline_oos_sharpe=None, per_window=[],
            evidence={"window_years": window_years, "step_months": step_months})

    oos_sharpe = float(rolling_df["Recommended Sharpe"].mean())
    baseline_oos_sharpe = (float(rolling_df["Current Sharpe"].mean())
                           if "Current Sharpe" in rolling_df.columns else None)

    mean, cov, _, _ = _covariance_inputs(prices, cov_method, n_components)
    final_arr = _optimize_weights(method, obj, mean, cov, r_bounds, r_cons, risk_free_rate)
    final_w = {t: float(w) for t, w in zip(cols, final_arr)}
    is_metrics = _realized_metrics(prices, final_w, risk_free_rate)
    in_sample_sharpe = float(is_metrics.get("Realized Sharpe", float("nan")))

    score = calculate_overfitting_score(
        {"R": {"Realized Sharpe": in_sample_sharpe,
               "Realized CAGR": is_metrics.get("Realized CAGR", 0.0)}},
        {"R": {"Realized Sharpe": oos_sharpe}})
    degradation = float(score.loc["R", "Overfitting Score"])

    if oos_sharpe > 0 and degradation <= max_degradation:
        verdict = "holds up"
        rationale = (f"OOS Sharpe {oos_sharpe:.2f} over {n_windows} windows; "
                     f"degradation {degradation:.0%} <= {max_degradation:.0%} "
                     f"(in-sample {in_sample_sharpe:.2f}).")
    else:
        verdict = "fragile (overfit)"
        rationale = (f"OOS Sharpe {oos_sharpe:.2f} over {n_windows} windows; "
                     f"degradation {degradation:.0%} (in-sample {in_sample_sharpe:.2f}). "
                     f"Advice did not generalize out-of-sample.")

    per_window = []
    has_current = "Current Sharpe" in rolling_df.columns
    for period, row in rolling_df.iterrows():
        per_window.append({
            "period": period,
            "recommended_sharpe": float(row.get("Recommended Sharpe", float("nan"))),
            "current_sharpe": (float(row["Current Sharpe"]) if has_current else None),
        })

    return RecommendationValidation(
        in_sample_sharpe=in_sample_sharpe, oos_sharpe=oos_sharpe,
        degradation=degradation, n_windows=n_windows, verdict=verdict,
        rationale=rationale, baseline_oos_sharpe=baseline_oos_sharpe,
        per_window=per_window,
        evidence={"max_degradation": max_degradation, "window_years": window_years,
                  "step_months": step_months})
