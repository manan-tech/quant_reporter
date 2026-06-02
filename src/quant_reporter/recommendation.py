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


def recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None,
                      profile=None, sector_map=None, risk_free_rate=0.02):
    """Point-in-time optimal target weights from a single objective.

    Optimizes `objective` over all columns of `prices` on the canonical
    (get_optimization_inputs) basis. Distinct from compare_verdict's
    backtest-driven pick. When `profile` is given and `bounds`/`constraints`
    are not supplied explicitly, they are derived from the profile via
    planning.apply_constraints (explicit bounds/constraints always win).
    """
    mean, cov, _ = get_optimization_inputs(prices)
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
        w = find_optimal_portfolio(objective, mean, cov, bounds, constraints, risk_free_rate)
        port_ret, port_vol, sharpe = get_portfolio_stats(w, mean, cov, risk_free_rate)
    weights = {c: float(wi) for c, wi in zip(cols, np.asarray(w, dtype=float))}
    obj_name = getattr(objective, "__name__", str(objective))
    rationale = (f"Weights chosen by optimizing the {obj_name} objective; resulting "
                 f"Sharpe {sharpe:.2f} (return {port_ret:.2%}, vol {port_vol:.2%}; "
                 f"annualized log-return basis).")
    evidence = {"objective": obj_name, "sharpe": float(sharpe),
                "expected_return": float(port_ret), "expected_vol": float(port_vol),
                "basis": "annualized_log_return"}
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
              validate=False, window_years=1, step_months=3, max_degradation=0.5):
    """Opt-in recommendation bundle. `prices` are asset prices only. Alerts run on
    `current_weights` when given, else on the recommended target.

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
                               sector_map=sector_map, risk_free_rate=risk_free_rate)
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
            risk_free_rate=risk_free_rate, max_degradation=max_degradation)
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
                                risk_free_rate=0.02, max_degradation=0.5):
    """Walk-forward OOS validation of the recommendation on raw asset prices.

    At each rolling window the constrained recommendation is re-derived on the
    train slice (honoring `objective` + `profile`) and applied forward; the user's
    `current_weights` (if given) are validated as an OOS baseline. Returns a
    RecommendationValidation; verdict is degradation-based.
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
        arr = find_optimal_portfolio(obj, mean, cov, r_bounds, r_cons, risk_free_rate)
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

    mean, cov, _ = get_optimization_inputs(prices)
    final_arr = find_optimal_portfolio(obj, mean, cov, r_bounds, r_cons, risk_free_rate)
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
