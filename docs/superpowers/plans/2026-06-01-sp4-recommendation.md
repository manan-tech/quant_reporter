# SP4 — Recommendation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the opt-in recommendation layer (SP4): four standalone functions (`recommend_weights`, `rebalance_trades`, `risk_alerts`, `compare_verdict`) each returning a structured object with `rationale`+`evidence`, a `recommend()` orchestrator that bundles them into a `Recommendation`, a self-rendered transparent HTML section, and an additive report hook.

**Architecture:** Two new flat modules. `recommendation.py` holds the dataclasses and the five functions; each CONSUMES existing primitives (`get_optimization_inputs`/`find_optimal_portfolio`/`get_portfolio_stats`, `portfolio_turnover`/`transaction_cost_model`, `forecast_portfolio_vol`/`risk_contributions`/`compute_drawdown`, `compare_strategies_oos`, `compute_asset_factor_exposures`) — it never re-optimizes or re-backtests. `recommendation_report.py` renders a `Recommendation` via the existing `html_builder.generate_html_report`, lazy-imported by `Recommendation.to_html()` (the `BacktestResult`→`backtest_report` pattern). The report hook is additive kwargs on `create_backtest_report` and `BacktestResult.report`. All changes are additive; the 294 existing tests must stay green.

**Tech Stack:** numpy, pandas, scipy, sklearn, plotly (all already pinned). No new deps.

**Verification convention:** Always run the suite as
`PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" <args>` and trust the exit code.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/quant_reporter/recommendation.py` (create) | Dataclasses (`RecommendedWeights`, `Trade`, `RebalancePlan`, `RiskAlert`, `StrategyVerdict`, `Recommendation`) + `recommend_weights`, `rebalance_trades`, `risk_alerts`, `compare_verdict`, `recommend`; `to_dict`/`to_text`/`to_html`. Built across Tasks 1–5 (sequential, append). |
| `src/quant_reporter/recommendation_report.py` (create) | `build_recommendation_section`, `create_recommendation_report`, panel HTML builders. Task 6. |
| `src/quant_reporter/backtest_report.py` (modify) | Add `recommendation=None` kwarg to `create_backtest_report`. Task 7. |
| `src/quant_reporter/strategy.py` (modify) | `BacktestResult.report(..., recommendation=None)`. Task 7. |
| `src/quant_reporter/__init__.py` (modify) | Wire new exports (LAST, sequential). Task 8. |
| `test/test_recommendation.py` (create) | Tests for `recommendation.py` — appended across Tasks 1–5. |
| `test/test_recommendation_report.py` (create) | Rendering + report-hook tests. Tasks 6–7. |
| `examples/example_recommendation.py` (create) | Offline end-to-end demo → `examples/Recommendation_Report.html` (gitignored). Task 8. |

**Build order (dependency spine):** `recommendation.py` is built bottom-up across Tasks 1–5 (each task appends one dataclass + one function with its tests). Task 6 renders the objects. Task 7 wires the additive report hook into the two existing re-exported modules. Task 8 wires `__init__` and adds the example. **`__init__.py` is edited only in Task 8**, and the two existing-module edits (Task 7) are done sequentially — the cross-contamination rule.

**Deviations from the spec (YAGNI, recorded here):** `risk_alerts` drops the spec's unused `benchmark` and `risk_free_rate` params — no implemented check consumes them. `recommend` likewise drops `benchmark` (no check uses it; `compare_verdict` still accepts an optional `benchmark` returns Series directly). `recommend`/`recommend_weights`/`risk_alerts` assume `prices` are **asset prices only** (exclude any benchmark column), matching how `get_optimization_inputs` consumes all columns.

---

## Task 1: `recommend_weights` + `RecommendedWeights`

**Files:**
- Create: `src/quant_reporter/recommendation.py`
- Create: `test/test_recommendation.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_recommendation.py`:

```python
# test/test_recommendation.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend_weights, RecommendedWeights
from quant_reporter.objectives import neg_sharpe
from quant_reporter.opt_core import get_optimization_inputs, get_portfolio_stats
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


def test_recommend_weights_sums_to_one_and_keys():
    rec = recommend_weights(_prices())
    assert isinstance(rec, RecommendedWeights)
    assert sum(rec.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(rec.weights) == {"AAA", "BBB", "CCC"}
    assert all(v >= -1e-9 for v in rec.weights.values())


def test_recommend_weights_evidence_matches_portfolio_stats():
    prices = _prices()
    rec = recommend_weights(prices, objective=neg_sharpe, risk_free_rate=0.02)
    mean, cov, _ = get_optimization_inputs(prices)
    w = np.array([rec.weights[c] for c in cov.columns])
    pr, pv, sh = get_portfolio_stats(w, mean, cov, 0.02)
    assert rec.evidence["sharpe"] == pytest.approx(sh, rel=1e-9)
    assert rec.evidence["expected_vol"] == pytest.approx(pv, rel=1e-9)
    assert rec.evidence["expected_return"] == pytest.approx(pr, rel=1e-9)
    assert rec.objective == "neg_sharpe"


def test_recommend_weights_rationale_nonempty():
    assert recommend_weights(_prices()).rationale
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.recommendation'`.

- [ ] **Step 3: Create `recommendation.py`** (full import block + first dataclass + function)

```python
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
    rationale = (f"Weights chosen by minimizing {obj_name}; resulting Sharpe "
                 f"{sharpe:.2f} (return {port_ret:.2%}, vol {port_vol:.2%}).")
    evidence = {"objective": obj_name, "sharpe": float(sharpe),
                "expected_return": float(port_ret), "expected_vol": float(port_vol)}
    return RecommendedWeights(weights=weights, objective=obj_name,
                              rationale=rationale, evidence=evidence)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation.py
git commit -m "feat(recommendation): recommend_weights + RecommendedWeights"
```

---

## Task 2: `rebalance_trades` + `Trade` + `RebalancePlan`

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (append)
- Modify: `test/test_recommendation.py` (append)

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `test/test_recommendation.py`:

```python
from quant_reporter.recommendation import rebalance_trades, Trade, RebalancePlan
```

Append to `test/test_recommendation.py`:

```python
def test_rebalance_trades_basic_deltas():
    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.3, "BBB": 0.7})
    assert isinstance(plan, RebalancePlan)
    byt = {o.ticker: o for o in plan.orders}
    assert byt["AAA"].side == "sell" and byt["AAA"].delta == pytest.approx(-0.2)
    assert byt["BBB"].side == "buy" and byt["BBB"].delta == pytest.approx(0.2)
    assert plan.turnover == pytest.approx(0.2)  # one-way: 0.5*(0.2+0.2)


def test_rebalance_trades_union_missing_as_zero():
    plan = rebalance_trades({"AAA": 1.0}, {"AAA": 0.5, "BBB": 0.5})
    byt = {o.ticker: o for o in plan.orders}
    assert byt["BBB"].current_weight == 0.0 and byt["BBB"].side == "buy"
    assert byt["BBB"].delta == pytest.approx(0.5)


def test_rebalance_threshold_band_holds_small_trades():
    plan = rebalance_trades({"AAA": 0.50, "BBB": 0.50},
                            {"AAA": 0.49, "BBB": 0.51}, threshold=0.05)
    assert plan.orders == []
    assert set(plan.held) == {"AAA", "BBB"}


def test_rebalance_cost_uses_default_model():
    # executed deltas AAA -0.5, BBB +0.5 -> two-way notional 1.0;
    # default model: 1bps commission + 2.5bps half-spread -> cost_frac 3.5bps
    plan = rebalance_trades({"AAA": 0.5, "BBB": 0.5}, {"AAA": 0.0, "BBB": 1.0})
    assert plan.est_cost == pytest.approx((1.0 / 1e4) + (2.5 / 1e4), rel=1e-9)


def test_rebalance_no_trades_when_identical():
    w = {"AAA": 0.5, "BBB": 0.5}
    plan = rebalance_trades(w, w)
    assert plan.orders == [] and plan.turnover == pytest.approx(0.0)
    assert plan.est_cost == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: FAIL with `ImportError: cannot import name 'rebalance_trades'`.

- [ ] **Step 3: Append to `recommendation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: PASS (all, 8 total).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation.py
git commit -m "feat(recommendation): rebalance_trades + Trade + RebalancePlan"
```

---

## Task 3: `risk_alerts` + `RiskAlert`

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (append)
- Modify: `test/test_recommendation.py` (append)

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `test/test_recommendation.py`:

```python
from quant_reporter.recommendation import risk_alerts, RiskAlert
```

Append to `test/test_recommendation.py`:

```python
_EQ = {"AAA": 1 / 3, "BBB": 1 / 3, "CCC": 1 / 3}
_OFF = dict(vol_target=99, max_drawdown_limit=99, max_weight=0.99, max_risk_contribution=0.99)


def test_no_alerts_when_within_all_limits():
    assert risk_alerts(_EQ, _prices(), **_OFF) == []


def test_concentration_breach_fires():
    alerts = risk_alerts({"AAA": 0.8, "BBB": 0.1, "CCC": 0.1}, _prices(),
                         vol_target=99, max_drawdown_limit=99,
                         max_weight=0.40, max_risk_contribution=0.99)
    conc = [a for a in alerts if a.evidence.get("metric") == "max_weight"]
    assert conc and conc[0].kind == "concentration" and conc[0].severity == "breach"
    assert conc[0].evidence["asset"] == "AAA"
    assert conc[0].evidence["value"] == pytest.approx(0.8)


def test_concentration_warning_in_band():
    # max weight 0.38, cap 0.40, warn band 0.36 -> warning
    alerts = risk_alerts({"AAA": 0.38, "BBB": 0.32, "CCC": 0.30}, _prices(),
                         vol_target=99, max_drawdown_limit=99,
                         max_weight=0.40, max_risk_contribution=0.99)
    conc = [a for a in alerts if a.evidence.get("metric") == "max_weight"]
    assert conc and conc[0].severity == "warning"


def test_vol_breach_fires_with_tiny_target():
    alerts = risk_alerts(_EQ, _prices(), vol_target=1e-6, max_drawdown_limit=99,
                         max_weight=0.99, max_risk_contribution=0.99)
    vb = [a for a in alerts if a.kind == "vol_breach"]
    assert vb and vb[0].severity == "breach"
    assert vb[0].evidence["value"] > vb[0].evidence["threshold"]


def test_drawdown_breach_fires_with_tiny_limit():
    alerts = risk_alerts(_EQ, _prices(), max_drawdown_limit=1e-6, vol_target=99,
                         max_weight=0.99, max_risk_contribution=0.99)
    assert any(a.kind == "drawdown_breach" for a in alerts)


def test_sector_cap_breach():
    alerts = risk_alerts({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, _prices(),
                         sector_map={"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"},
                         sector_caps={"Tech": 0.5}, **_OFF)
    sect = [a for a in alerts if a.kind == "sector_cap"]
    assert sect and sect[0].evidence["sector"] == "Tech"
    assert sect[0].evidence["value"] == pytest.approx(0.8)


def test_factor_drift_silent_without_inputs():
    assert not any(a.kind == "factor_drift" for a in risk_alerts(_EQ, _prices(), **_OFF))


def test_factor_drift_fires_with_tiny_limit():
    rng = np.random.default_rng(0)
    fac = pd.DataFrame(rng.normal(0, 0.01, (len(_prices()) - 1, 2)),
                       index=_prices().index[1:], columns=["MKT", "SMB"])
    alerts = risk_alerts(_EQ, _prices(), factor_returns=fac, factor_loading_limit=1e-9, **_OFF)
    assert any(a.kind == "factor_drift" for a in alerts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: FAIL with `ImportError: cannot import name 'risk_alerts'`.

- [ ] **Step 3: Append to `recommendation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: PASS (all, 16 total).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation.py
git commit -m "feat(recommendation): risk_alerts + RiskAlert (5 limit-breach checks)"
```

---

## Task 4: `compare_verdict` + `StrategyVerdict`

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (append)
- Modify: `test/test_recommendation.py` (append)

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `test/test_recommendation.py`:

```python
from quant_reporter.recommendation import compare_verdict, StrategyVerdict
from quant_reporter.strategy import backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from quant_reporter.performance_stats import compare_strategies_oos
```

Append to `test/test_recommendation.py`:

```python
def _results(n=700, seed=3):
    prices = make_synthetic_prices(seed=seed, n_days=n)   # AAA/BBB/CCC + BMK
    return backtest_many({"EW": equal_weight, "RP": risk_parity}, prices, benchmark="BMK")


def test_compare_verdict_picks_best_by_dsr():
    results = _results()
    v = compare_verdict(results)
    assert isinstance(v, StrategyVerdict)
    cmp = compare_strategies_oos({k: r.returns for k, r in results.items()}, n_trials=2)
    assert v.winner == cmp["best_by_dsr"]
    assert v.ranking[0]["name"] == v.winner


def test_compare_verdict_evidence_has_summary():
    v = compare_verdict(_results())
    assert set(v.evidence["summary"]) == {"EW", "RP"}
    assert v.evidence["select_by"] == "dsr"
    assert v.evidence["n_trials"] == 2


def test_compare_verdict_single_result_no_comparison():
    prices = make_synthetic_prices(n_days=700)
    results = backtest_many({"EW": equal_weight}, prices)
    v = compare_verdict(results)
    assert v.winner == "EW"
    assert "no comparison" in v.rationale.lower()


def test_compare_verdict_empty():
    v = compare_verdict({})
    assert v.winner is None and v.ranking == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: FAIL with `ImportError: cannot import name 'compare_verdict'`.

- [ ] **Step 3: Append to `recommendation.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: PASS (all, 20 total).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation.py
git commit -m "feat(recommendation): compare_verdict + StrategyVerdict (DSR-ranked)"
```

---

## Task 5: `recommend` orchestrator + `Recommendation` (to_dict / to_text)

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (append)
- Modify: `test/test_recommendation.py` (append)

- [ ] **Step 1: Write the failing tests**

Add to the import block at the top of `test/test_recommendation.py`:

```python
from quant_reporter.recommendation import recommend, Recommendation
```

Append to `test/test_recommendation.py`:

```python
def test_recommend_bundle_full():
    prices = _prices(n=700)
    results = backtest_many({"EW": equal_weight, "RP": risk_parity},
                            make_synthetic_prices(n_days=700), benchmark="BMK")
    rec = recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                    results=results, **_OFF)
    assert isinstance(rec, Recommendation)
    assert isinstance(rec.target_weights, RecommendedWeights)
    assert isinstance(rec.trades, RebalancePlan)
    assert isinstance(rec.alerts, list)
    assert isinstance(rec.verdict, StrategyVerdict)


def test_recommend_no_current_weights_no_trades_no_verdict():
    rec = recommend(_prices(n=700), **_OFF)
    assert rec.trades is None
    assert rec.verdict is None


def test_recommend_alerts_target_when_no_current():
    # no current_weights -> alerts run on the recommended target; tiny name cap
    # almost surely trips a concentration alert
    rec = recommend(_prices(n=700), vol_target=99, max_drawdown_limit=99,
                    max_weight=1e-6, max_risk_contribution=0.99)
    assert any(a.kind == "concentration" for a in rec.alerts)


def test_recommendation_to_dict_keys():
    rec = recommend(_prices(n=700), current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, **_OFF)
    d = rec.to_dict()
    assert set(d) == {"target_weights", "trades", "alerts", "verdict"}
    assert d["trades"]["orders"] and isinstance(d["alerts"], list)
    assert d["verdict"] is None


def test_recommendation_to_text_is_string():
    rec = recommend(_prices(n=700), current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, **_OFF)
    txt = rec.to_text()
    assert isinstance(txt, str) and "RECOMMENDATION" in txt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: FAIL with `ImportError: cannot import name 'recommend'`.

- [ ] **Step 3: Append to `recommendation.py`**

```python
@dataclass
class Recommendation:
    target_weights: RecommendedWeights
    trades: Optional[RebalancePlan]
    alerts: list              # list[RiskAlert]
    verdict: Optional[StrategyVerdict]

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
        return "\n".join(lines)

    def to_html(self, path=None, open_browser=False):
        from .recommendation_report import create_recommendation_report
        return create_recommendation_report(self, path=path, open_browser=open_browser)


def recommend(prices, *, current_weights=None, objective=neg_sharpe, results=None,
              cost_model=None, threshold=0.0, vol_target=0.10, max_drawdown_limit=0.20,
              max_weight=0.40, max_risk_contribution=0.40, sector_map=None,
              sector_caps=None, factor_returns=None, factor_loading_limit=None,
              risk_free_rate=0.02):
    """Opt-in recommendation bundle. `prices` are asset prices only. Alerts run on
    `current_weights` when given, else on the recommended target."""
    target = recommend_weights(prices, objective=objective, risk_free_rate=risk_free_rate)
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
    return Recommendation(target_weights=target, trades=trades, alerts=alerts, verdict=verdict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation.py -q`
Expected: PASS (all, 25 total).

- [ ] **Step 5: Verify the full suite is still green**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: PASS, count = 294 + 25 (= 319), no failures. (`__init__` not yet wired, so the new module is reached only via the test's direct import — that's fine.)

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation.py
git commit -m "feat(recommendation): recommend() orchestrator + Recommendation bundle"
```

---

## Task 6: Recommendation report rendering

**Files:**
- Create: `src/quant_reporter/recommendation_report.py`
- Create: `test/test_recommendation_report.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_recommendation_report.py`:

```python
# test/test_recommendation_report.py
import os
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend
from quant_reporter.recommendation_report import (
    build_recommendation_section, create_recommendation_report,
)
from quant_reporter.strategy import backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from conftest import make_synthetic_prices

_OFF = dict(vol_target=99, max_drawdown_limit=99, max_weight=0.99, max_risk_contribution=0.99)


def _full_rec():
    prices = make_synthetic_prices(n_days=700)[["AAA", "BBB", "CCC"]]
    results = backtest_many({"EW": equal_weight, "RP": risk_parity},
                            make_synthetic_prices(n_days=700), benchmark="BMK")
    return recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                     results=results, **_OFF)


def test_build_section_has_four_panels():
    section = build_recommendation_section(_full_rec())
    titles = [item["title"] for item in section["main_content"]]
    for expected in ("Recommended Target Weights", "Rebalance Plan",
                     "Risk Alerts", "Strategy Verdict"):
        assert expected in titles


def test_create_report_writes_file(tmp_path):
    path = str(tmp_path / "reco.html")
    out = create_recommendation_report(_full_rec(), path=path)
    assert out == path and os.path.exists(path)
    html = open(path, encoding="utf-8").read()
    assert "Recommended Target Weights" in html and "Risk Alerts" in html
    assert len(html) > 1000


def test_to_html_delegates(tmp_path):
    path = str(tmp_path / "reco2.html")
    _full_rec().to_html(path)
    assert os.path.exists(path)


def test_report_handles_missing_panels(tmp_path):
    # no current_weights -> no trades; no results -> no verdict
    rec = recommend(make_synthetic_prices(n_days=700)[["AAA", "BBB", "CCC"]], **_OFF)
    path = str(tmp_path / "reco3.html")
    create_recommendation_report(rec, path=path)
    html = open(path, encoding="utf-8").read()
    assert "No trades" in html and "No strategy comparison" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.recommendation_report'`.

- [ ] **Step 3: Create `recommendation_report.py`**

```python
# src/quant_reporter/recommendation_report.py
"""Recommendation report rendering (SP4).

Renders a Recommendation as a transparent HTML section via html_builder.
Lazy-imported by Recommendation.to_html(). Numbers come only from the
recommendation objects; nothing is recomputed.
"""
import os

import pandas as pd

from .html_builder import generate_html_report

_SEV_COLOR = {"breach": "#c0392b", "warning": "#e67e22", "ok": "#27ae60"}


def _weights_table_html(rw):
    rows = "".join(f"<tr><td>{tk}</td><td>{wt:.2%}</td></tr>"
                   for tk, wt in rw.weights.items())
    return (f'<table class="metrics-table"><tr><th>Asset</th><th>Weight</th></tr>'
            f'{rows}</table>')


def _trades_table_html(plan):
    if plan is None or not plan.orders:
        return "<p>No trades.</p>"
    rows = "".join(
        f"<tr><td>{o.side.upper()}</td><td>{o.ticker}</td><td>{o.current_weight:.2%}</td>"
        f"<td>{o.target_weight:.2%}</td><td>{o.delta:+.2%}</td></tr>" for o in plan.orders)
    return ('<table class="metrics-table"><tr><th>Side</th><th>Asset</th><th>Current</th>'
            f'<th>Target</th><th>Delta</th></tr>{rows}</table>')


def _alerts_html(alerts):
    if not alerts:
        return "<p>No risk alerts &mdash; all checks within limits.</p>"
    items = "".join(
        f'<li style="color:{_SEV_COLOR.get(a.severity, "#333")}">'
        f'<b>[{a.severity.upper()}] {a.kind}</b>: {a.rationale}</li>' for a in alerts)
    return f"<ul>{items}</ul>"


def _verdict_table_html(verdict):
    if verdict is None or not verdict.ranking:
        return "<p>No strategy comparison.</p>"
    df = pd.DataFrame(verdict.ranking).set_index("name")
    return df.to_html(classes="metrics-table", float_format=lambda x: f"{x:.3f}")


def build_recommendation_section(rec):
    return {
        "title": "Recommendations",
        "main_content": [
            {"type": "table_html", "title": "Recommended Target Weights",
             "data": _weights_table_html(rec.target_weights),
             "description": rec.target_weights.rationale},
            {"type": "table_html", "title": "Rebalance Plan",
             "data": _trades_table_html(rec.trades),
             "description": (rec.trades.rationale if rec.trades is not None
                             else "No current weights provided.")},
            {"type": "table_html", "title": "Risk Alerts",
             "data": _alerts_html(rec.alerts)},
            {"type": "table_html", "title": "Strategy Verdict",
             "data": _verdict_table_html(rec.verdict),
             "description": (rec.verdict.rationale if rec.verdict is not None
                             else "No strategies compared.")},
        ],
    }


def create_recommendation_report(rec, path=None, open_browser=False):
    path = path or "recommendation_report.html"
    generate_html_report([build_recommendation_section(rec)],
                          title="Recommendation Report", filename=path)
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation_report.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation_report.py test/test_recommendation_report.py
git commit -m "feat(recommendation-report): transparent HTML section + create_recommendation_report"
```

---

## Task 7: Report hook (additive kwargs on existing modules)

**Files:**
- Modify: `src/quant_reporter/backtest_report.py` (`create_backtest_report` signature + body)
- Modify: `src/quant_reporter/strategy.py` (`BacktestResult.report`)
- Modify: `test/test_recommendation_report.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `test/test_recommendation_report.py`:

```python
from quant_reporter.strategy import backtest


def test_backtest_report_embeds_recommendation(tmp_path):
    prices = make_synthetic_prices(n_days=700)
    res = backtest(equal_weight, prices, benchmark="BMK")
    rec = recommend(prices[["AAA", "BBB", "CCC"]],
                    current_weights={"AAA": 0.4, "BBB": 0.3, "CCC": 0.3}, **_OFF)
    path = str(tmp_path / "bt_with_reco.html")
    res.report(path, recommendation=rec)
    html = open(path, encoding="utf-8").read()
    assert "Recommendations" in html and "Growth of $1" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation_report.py::test_backtest_report_embeds_recommendation -q`
Expected: FAIL with `TypeError: report() got an unexpected keyword argument 'recommendation'`.

- [ ] **Step 3a: Modify `create_backtest_report` in `backtest_report.py`**

Change its signature and append the recommendation section before `generate_html_report`. The current function is:

```python
def create_backtest_report(result_or_results, path="backtest_report.html", open_browser=False):
    if isinstance(result_or_results, dict):
        sections = []
        for res in result_or_results.values():
            sections.extend(build_sections(res))
        sections.append(_comparison_section(result_or_results))
        title = "Strategy Comparison Report"
    else:
        sections = build_sections(result_or_results)
        title = f"Backtest Report — {result_or_results.name}"
    generate_html_report(sections, title=title, filename=path)
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{path}")
    return path
```

Replace it with (adds `recommendation=None` and the section append):

```python
def create_backtest_report(result_or_results, path="backtest_report.html",
                           open_browser=False, recommendation=None):
    if isinstance(result_or_results, dict):
        sections = []
        for res in result_or_results.values():
            sections.extend(build_sections(res))
        sections.append(_comparison_section(result_or_results))
        title = "Strategy Comparison Report"
    else:
        sections = build_sections(result_or_results)
        title = f"Backtest Report — {result_or_results.name}"
    if recommendation is not None:
        from .recommendation_report import build_recommendation_section
        sections.append(build_recommendation_section(recommendation))
    generate_html_report(sections, title=title, filename=path)
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{path}")
    return path
```

- [ ] **Step 3b: Modify `BacktestResult.report` in `strategy.py`**

The current method is:

```python
    def report(self, path="backtest_report.html", open_browser=False):
        from .backtest_report import create_backtest_report
        return create_backtest_report(self, path=path, open_browser=open_browser)
```

Replace it with:

```python
    def report(self, path="backtest_report.html", open_browser=False, recommendation=None):
        from .backtest_report import create_backtest_report
        return create_backtest_report(self, path=path, open_browser=open_browser,
                                      recommendation=recommendation)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_recommendation_report.py -q`
Expected: PASS (all 5).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/backtest_report.py src/quant_reporter/strategy.py test/test_recommendation_report.py
git commit -m "feat(recommendation): additive report hook on backtest report + BacktestResult.report"
```

---

## Task 8: Wire exports + example + full-suite green

**Files:**
- Modify: `src/quant_reporter/__init__.py` (append before `__version__`)
- Create: `examples/example_recommendation.py`

- [ ] **Step 1: Append exports to `__init__.py`**

Insert immediately BEFORE the final `__version__ = "2.0.0"` line:

```python
# --- SP4: recommendation layer (opt-in opinions) ---
from .recommendation import (
    recommend, recommend_weights, rebalance_trades, risk_alerts, compare_verdict,
    Recommendation, RecommendedWeights, RebalancePlan, Trade, RiskAlert, StrategyVerdict,
)
from .recommendation_report import create_recommendation_report
```

- [ ] **Step 2: Verify the package imports and the full suite is green**

Run: `PYTHONPATH=src:test .venv/bin/python -c "import quant_reporter as qr; print(qr.recommend, qr.risk_alerts, qr.RecommendedWeights)"`
Expected: prints the three objects, no error.

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: PASS, count = 294 + 25 (Tasks 1–5) + 5 (Tasks 6–7) = 324, no failures.

- [ ] **Step 3: Create `examples/example_recommendation.py`**

```python
"""example_recommendation.py — opt-in recommendation bundle + transparent report.

Offline (synthetic fixture). Run:
    python examples/example_recommendation.py
Produces examples/Recommendation_Report.html and embeds the section into a
backtest report (examples/Backtest_With_Reco.html).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import functools
from conftest import make_synthetic_prices
import quant_reporter as qr


def main():
    prices = make_synthetic_prices(seed=42, n_days=900)        # AAA/BBB/CCC + BMK
    assets = prices[["AAA", "BBB", "CCC"]]
    current = {"AAA": 0.50, "BBB": 0.30, "CCC": 0.20}
    cost = functools.partial(qr.transaction_cost_model, commission_bps=1.0, spread_bps=5.0)

    results = qr.backtest_many(
        {"EqualWeight": qr.equal_weight, "RiskParity": qr.risk_parity,
         "MaxSharpe": qr.max_sharpe},
        prices, benchmark="BMK", rebalance="M", cost_model=cost)

    rec = qr.recommend(assets, current_weights=current, results=results,
                       cost_model=cost, vol_target=0.10, max_drawdown_limit=0.20,
                       max_weight=0.40)

    print(rec.to_text())

    out = os.path.join(os.path.dirname(__file__), "Recommendation_Report.html")
    qr.create_recommendation_report(rec, path=out)
    print(f"\nRecommendation report: {out}")

    bt = os.path.join(os.path.dirname(__file__), "Backtest_With_Reco.html")
    results["RiskParity"].report(bt, recommendation=rec)
    print(f"Backtest report with recommendation: {bt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the example end-to-end**

Run: `PYTHONPATH=src:test .venv/bin/python examples/example_recommendation.py`
Expected: prints the text digest and two `... report: .../*.html` lines; both files exist and are non-empty.

- [ ] **Step 5: Confirm generated HTML is gitignored**

Run: `git status --porcelain examples/`
Expected: shows `examples/example_recommendation.py` (new source) but NOT `examples/Recommendation_Report.html` / `examples/Backtest_With_Reco.html` (already covered by the `examples/*.html` rule in `.gitignore`). If the HTML appears, add `examples/*.html` to `.gitignore`.

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/__init__.py examples/example_recommendation.py
git commit -m "feat(sp4): wire recommendation exports + end-to-end example"
```

---

## Self-Review

**Spec coverage:**
- §2 API surface (parts + bundle) → `recommend_weights` (T1), `rebalance_trades` (T2), `risk_alerts` (T3), `compare_verdict` (T4), `recommend` (T5).
- §3 object schemas (rationale+evidence; severity on alerts) → dataclasses introduced in T1–T5, each carrying `rationale`+`evidence`; `RiskAlert` adds `severity`.
- §4 exact wiring → T1 uses `get_optimization_inputs`/`find_optimal_portfolio`/`get_portfolio_stats`/`build_constraints`; T2 uses `portfolio_turnover`/`transaction_cost_model`; T3 uses `forecast_portfolio_vol`/`get_portfolio_price`/`compute_drawdown`/`risk_contributions`/`compute_asset_factor_exposures`; T4 uses `compare_strategies_oos`.
- §5 reporting (self-section + report hook) → T6 (`recommendation_report.py`, `to_html` delegate) + T7 (`create_backtest_report(..., recommendation=)`, `BacktestResult.report(..., recommendation=)`).
- §6 look-ahead safety → point-in-time, no schedules; no future-shuffle test needed (documented in spec). T3 uses trailing data only via existing primitives.
- §7 files & build order → recommendation.py across T1–T5, recommendation_report.py T6, existing-module edits T7 (sequential), `__init__` T8 (last).
- §8 testing → every task ends with its file green; T5/T8 add a full-suite gate; edges covered (no current_weights, single strategy, healthy book → no alerts, union/missing, threshold band, missing benchmark, missing factor inputs).
- §9 defaults → encoded in `risk_alerts`/`recommend` signatures (vol 0.10, dd 0.20, weight 0.40, risk-contrib 0.40, threshold 0.0, select_by 'dsr', warn band 0.9).
- §10 out of scope → no live trading / tax / new deps / re-backtesting.

**Placeholder scan:** No TBD/TODO; every code step shows full code. Step "if the HTML appears, add ..." (T8 S5) is a verified-conditional with the exact fix, not a placeholder.

**Type consistency:** Dataclass field names defined in T1–T5 (`RecommendedWeights.weights/objective/rationale/evidence`, `Trade.ticker/side/current_weight/target_weight/delta`, `RebalancePlan.orders/turnover/est_cost/held`, `RiskAlert.kind/severity/rationale/evidence`, `StrategyVerdict.winner/ranking/rationale/evidence`, `Recommendation.target_weights/trades/alerts/verdict`) are exactly the fields read by the renderer in T6 (`_weights_table_html` reads `.weights`; `_trades_table_html` reads `o.side/.ticker/.current_weight/.target_weight/.delta`; `_alerts_html` reads `a.severity/.kind/.rationale`; `_verdict_table_html` reads `verdict.ranking` rows `{name,sharpe,psr,dsr}`) and by `to_dict`/`to_text` in T5. `create_recommendation_report(rec, path=None, open_browser=False)` (T6) matches the `to_html` delegate (T5) and the call in T8's example. `create_backtest_report(..., recommendation=None)` (T7) matches `BacktestResult.report(..., recommendation=None)` (T7) and the example (T8). `risk_alerts` evidence always carries `metric`/`value`/`threshold` (asserted in T3). `compare_verdict` ranking rows use key `name` (set in T4, read in T6's `set_index("name")`).
