# SP4 — Recommendation layer

**Date:** 2026-06-01
**Branch:** `v2.1`
**Status:** Design approved; ready for implementation plan.
**Depends on:** SP0 (analytics core), SP1 (`simulate_strategy`, `portfolio_turnover`, `transaction_cost_model`, `compare_strategies_oos`/PSR/DSR), SP2 (`forecast_portfolio_vol`, `risk_contributions`, `portfolio_cvar`), SP3 (`compute_asset_factor_exposures`), SP-Strategy (`backtest_many`, `BacktestResult`, `summary_metrics`, objectives).
**Feeds:** the v2.1 release (final SP). After this: merge `v2.1` → `main`, bump `__version__`, user runs `twine upload`.

---

## 1. Goal & posture

Add the **only** opinionated surface in the library: an opt-in recommendation layer that
produces **structured, transparent** recommendation objects, each carrying the metric and
rationale that drove it. Everything else stays opinion-free with explicit params; the
opinions (vol target, drawdown limit, concentration cap, selection metric) live here and
here only, all overridable.

SP4 is a **thin consumer** of SP0–SP-Strategy. It never re-optimizes a frontier it could
read, never re-backtests a strategy already run. It reads existing primitives and packages
their output as recommendations with provenance.

No new dependencies (numpy/pandas/scipy/sklearn/plotly already pinned). Flat package layout
preserved; new modules mirror the established `backtest.py` / `backtest_report.py` split.

## 2. API surface — parts + bundle orchestrator

Four standalone functions, each returning a structured object usable on its own, **plus** a
`recommend()` orchestrator that runs all four and returns a `Recommendation` bundle.

```python
import quant_reporter as qr

# Each piece standalone (opinion-free inputs, explicit params):
tgt    = qr.recommend_weights(prices, objective=qr.neg_sharpe)
trades = qr.rebalance_trades(current_w, target_w, cost_model=qr.transaction_cost_model)
alerts = qr.risk_alerts(weights, prices, vol_target=0.10, max_drawdown_limit=0.20)
verd   = qr.compare_verdict(results)              # results = qr.backtest_many(...)

# Or the whole bundle in one opt-in call:
rec = qr.recommend(prices, current_weights=w, results=results)
rec.target_weights     # RecommendedWeights
rec.trades             # RebalancePlan | None
rec.alerts             # list[RiskAlert]
rec.verdict            # StrategyVerdict | None
rec.to_html("reco.html", open_browser=True)       # transparent section
```

## 3. Object schemas

All dataclasses; `rationale` (human-readable str) + `evidence` (machine-readable dict of the
metrics that drove the recommendation) are the through-line on every object. Alerts add
`severity`.

```python
@dataclass
class RecommendedWeights:
    weights: dict            # ticker -> weight
    objective: str           # name of the objective/optimizer chosen
    rationale: str
    evidence: dict           # {'objective':'neg_sharpe','sharpe':..,'vol':..,'ret':..}

@dataclass
class Trade:
    ticker: str
    side: str                # 'buy' | 'sell'
    current_weight: float
    target_weight: float
    delta: float             # target - current
    rationale: str
    evidence: dict           # {'abs_delta':..,'threshold':..}

@dataclass
class RebalancePlan:
    orders: list[Trade]
    turnover: float          # one-way, from portfolio_turnover
    est_cost: float          # cost_frac from the cost model on executed deltas
    held: list[str]          # tickers below the no-trade band (intentionally not traded)
    rationale: str
    evidence: dict           # {'turnover':..,'est_cost':..,'n_orders':..,'n_held':..}

@dataclass
class RiskAlert:
    kind: str                # 'vol_breach' | 'drawdown_breach' | 'concentration'
                             #   | 'sector_cap' | 'factor_drift'
    severity: str            # 'ok' | 'warning' | 'breach'
    rationale: str
    evidence: dict           # {'metric':..,'value':..,'threshold':..,'comparator':'>'}

@dataclass
class StrategyVerdict:
    winner: str | None
    ranking: list            # ordered [{'name':..,'sharpe':..,'psr':..,'dsr':..}, ...]
    rationale: str
    evidence: dict           # {'select_by':'dsr','summary':{...},'pvalues':{...}}

@dataclass
class Recommendation:        # the bundle
    target_weights: RecommendedWeights
    trades: RebalancePlan | None
    alerts: list[RiskAlert]
    verdict: StrategyVerdict | None

    def to_dict(self) -> dict
    def to_text(self) -> str
    def to_html(self, path=None, open_browser=False) -> str   # lazy-imports recommendation_report
```

**Refinement note:** an earlier sketch had `rec.trades` as a bare `list[Trade]`. It is a
`RebalancePlan` carrying the order list **plus** the turnover/cost estimate the roadmap
requires; `rec.trades.orders` is the list.

## 4. The four functions — exact wiring

Each consumes existing primitives; nothing is re-derived.

### `recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None, risk_free_rate=0.02)` → `RecommendedWeights`
- `mean, cov = get_optimization_inputs(prices)` (one canonical basis).
- `weights = find_optimal_portfolio(objective, mean, cov, bounds, constraints, risk_free_rate)`.
- `evidence = get_portfolio_stats(weights, mean, cov, rfr)` → `{objective, sharpe, vol, ret}`.
- Point-in-time optimizer pick on all available data; **distinct** from the backtest-driven
  verdict (§4 `compare_verdict`).

### `rebalance_trades(current_weights, target_weights, *, cost_model=None, threshold=0.0, portfolio_value=1.0)` → `RebalancePlan`
- `to = portfolio_turnover(current_weights, target_weights, convention="one_way")` →
  union-aligned signed deltas (`to['trades']`), `to['turnover']`.
- **No-trade band:** emit a `Trade` for each ticker with `|delta| >= threshold`; tickers
  below the band go to `held` (intentionally not traded). Default `threshold=0.0` → trade all.
- `side = 'buy' if delta > 0 else 'sell'`.
- `est_cost = (cost_model or transaction_cost_model)(executed_deltas)['cost_frac']` — cost is
  charged on the **executed** deltas (after the band), not the raw deltas.
- `evidence = {'turnover':..,'est_cost':..,'n_orders':..,'n_held':..,'threshold':..}`.

### `risk_alerts(weights, prices, *, vol_target=0.10, max_drawdown_limit=0.20, max_weight=0.40, max_risk_contribution=0.40, sector_map=None, sector_caps=None, factor_returns=None, factor_loading_limit=None, benchmark=None, risk_free_rate=0.02)` → `list[RiskAlert]`

Five checks; each emits a `RiskAlert` with `evidence={'metric','value','threshold','comparator'}`:

| kind | source primitive | breach condition | gated on |
|------|------------------|------------------|----------|
| `vol_breach` | `forecast_portfolio_vol(w, cov)`, `cov` from `get_optimization_inputs` | forecast vol `>` `vol_target` | always |
| `drawdown_breach` | portfolio returns (weights × prices) → `compute_drawdown` / `max_drawdown` | trough beyond `max_drawdown_limit` | always |
| `concentration` | weights; `risk_contributions(w, cov)` | any name weight `>` `max_weight`, OR any risk-contribution share `>` `max_risk_contribution` | always |
| `sector_cap` | `sector_map` + `sector_caps` | sector weight sum `>` cap | only if both provided |
| `factor_drift` | `compute_asset_factor_exposures` (portfolio-level) | `\|loading\|` `>` `factor_loading_limit` | only if `factor_returns` **and** `factor_loading_limit` provided |

- **Severity:** `breach` when over the limit; `warning` when within the warning band
  (`0.9 × limit`); otherwise no alert is emitted for that check. A healthy book yields an
  empty list (an optional `ok` summary alert may be included for the report — decided in plan).
- Portfolio returns for the drawdown check come from aligning `weights` to `prices` and
  compounding simple returns (reuse the analytics-core returns helper; trailing-only, no
  look-ahead).

### `compare_verdict(results, *, select_by="dsr", benchmark=None)` → `StrategyVerdict`
- `results`: `dict[str, BacktestResult]` (from `backtest_many`). **Consumes** them — does not
  re-backtest.
- `returns_dict = {name: res.returns for name, res in results.items()}`.
- `cmp = compare_strategies_oos(returns_dict, benchmark_returns, n_trials=len(results))` →
  already returns `{'summary': {name:{sharpe,psr,dsr}}, 'sharpe_diff_pvalues':..., 'best_by_dsr':...}`.
- `winner = cmp['best_by_dsr']` when `select_by='dsr'` (honest, multiple-testing-deflated);
  other `select_by` values rank on the corresponding `summary` metric.
- `ranking` = strategies ordered by the selection metric; `evidence` = full `summary` +
  `sharpe_diff_pvalues`.
- **Edge:** single-result input → names it as winner, `rationale` notes "no comparison".

### `recommend(prices, *, current_weights=None, objective=neg_sharpe, results=None, cost_model=None, threshold=0.0, vol_target=0.10, max_drawdown_limit=0.20, max_weight=0.40, max_risk_contribution=0.40, sector_map=None, sector_caps=None, factor_returns=None, factor_loading_limit=None, benchmark=None, risk_free_rate=0.02)` → `Recommendation`
- `target = recommend_weights(prices, objective=..., risk_free_rate=...)`.
- `trades = rebalance_trades(current_weights, target.weights, cost_model=..., threshold=...)`
  **if** `current_weights` is given, else `None`.
- `alerts = risk_alerts(alert_weights, prices, ...)` where `alert_weights = current_weights if
  current_weights is not None else target.weights` — alert on the **live** book when known,
  else on the recommended target (documented behavior).
- `verdict = compare_verdict(results, ...)` **if** `results` is given, else `None`.

## 5. Reporting

`recommendation_report.py` (rendering only; lazy-imported by `Recommendation.to_html()`,
exactly like `BacktestResult` → `backtest_report`):

- `build_recommendation_section(rec) -> html` — a self-contained section via `html_builder`
  + plotly tables, four panels:
  1. **Recommended target weights** — table (ticker, weight) + the objective evidence.
  2. **Rebalance plan** — trade blotter (ticker, side, current→target, delta) + turnover/cost.
  3. **Risk alerts** — list with severity coloring; each row shows metric/value/threshold.
  4. **Strategy verdict** — ranking table (sharpe/psr/dsr) with the winner highlighted.
  Every panel surfaces `rationale` + `evidence`. Panels with no data (e.g. no `current_weights`
  → no plan) are omitted or shown as "n/a".
- `create_recommendation_report(rec, path=None, open_browser=False) -> str` — standalone page
  wrapper (naming parity with the other `create_*_report` generators).
- **Report hook (additive kwargs on existing modules):**
  - `create_backtest_report(..., recommendation=None)` appends the section when passed.
  - `BacktestResult.report(..., recommendation=None)` threads it through.

The report is one consumer of the recommendation objects — every number comes from object
fields; it never recomputes.

## 6. Look-ahead safety

SP4 produces **point-in-time** recommendations (optimize/alert on the latest available data)
and **consumes already-honest backtests**. It generates **no dated weight schedule**, so the
future-shuffle causality property tests SP1/SP2 use do not apply here. The OOS-honesty risk
lives in `compare_strategies_oos` (SP1, already pinned with its own tests). The spec states
this explicitly so the absence of a causality test is understood as correct, not an omission.

## 7. Files & build order

**New modules (brand-new — parallelizable until the `__init__`/existing-module wire-up):**
- `src/quant_reporter/recommendation.py` — objects + `recommend_weights`, `rebalance_trades`,
  `risk_alerts`, `compare_verdict`, `recommend`; `Recommendation.to_dict/to_text/to_html`.
- `src/quant_reporter/recommendation_report.py` — `build_recommendation_section`,
  `create_recommendation_report`, panel builders.

**Edit (sequential — shared re-export surface; orchestrator wires these LAST, one at a time):**
- `src/quant_reporter/backtest_report.py` — add `recommendation=None` kwarg.
- `src/quant_reporter/strategy.py` — `BacktestResult.report(..., recommendation=None)`.
- `src/quant_reporter/__init__.py` — export new names.

**Tests (one file per module, offline via `make_synthetic_prices`):**
- `test/test_recommendation.py` — the four functions + orchestrator + objects.
- `test/test_recommendation_report.py` — HTML rendering + the report hook.

**Example:**
- `examples/example_recommendation.py` — offline, build a `Recommendation`, write
  `examples/Recommendation_Report.html` (gitignored) and embed it in a backtest report.

**Dependency spine:** `recommendation.py` (objects + functions, leaf consumer of existing
primitives) → `recommendation_report.py` (renders the objects) → existing-module hooks +
`__init__` exports + example.

## 8. Testing & safety

- **All additive; the 294 existing tests must stay green** (exit-0 + temp-file verification,
  not raw stdout).
- **`recommend_weights`**: weights sum to 1, respect bounds/constraints; evidence matches
  `get_portfolio_stats`; reproducible.
- **`rebalance_trades`**: deltas = target − current on the ticker union (missing → 0); buy/sell
  sides correct; no-trade band moves sub-threshold tickers to `held`; turnover matches
  `portfolio_turnover`; cost matches the cost model on executed deltas.
- **`risk_alerts`**: each kind fires exactly at its threshold (golden values); warning band at
  `0.9×limit`; healthy book → no breach alerts; sector/factor checks silent when inputs absent;
  evidence carries `{metric,value,threshold,comparator}`.
- **`compare_verdict`**: winner = `best_by_dsr` under `select_by='dsr'`; ranking ordered;
  single-result edge; benchmark p-values when provided.
- **`recommend`**: bundle wires the four correctly; `trades=None` when no `current_weights`;
  alerts target the live book when given else the target; `verdict=None` when no `results`.
- **Report**: generated offline; HTML non-empty, contains each panel title/anchor; absent
  panels handled; the `recommendation=` hook on `create_backtest_report` /
  `BacktestResult.report` embeds the section.
- **`to_dict`/`to_text`**: round-trip-stable, evidence preserved, no display strings inside
  numeric fields.
- **No BLAS-warning leakage** — wrap cov-heavy blocks (`forecast_portfolio_vol`,
  optimization, `risk_contributions`) in `np.errstate(divide="ignore", over="ignore",
  invalid="ignore")` per the established pattern.
- **No new dependencies.**

## 9. Default opinions (the only opinionated values in the library)

All overridable; defaults live only in SP4 function signatures.

| Param | Default | Meaning |
|-------|---------|---------|
| `vol_target` | 0.10 | annualized portfolio vol ceiling |
| `max_drawdown_limit` | 0.20 | drawdown depth ceiling |
| `max_weight` | 0.40 | single-name weight cap |
| `max_risk_contribution` | 0.40 | single-name risk-contribution share cap |
| `threshold` (rebalance) | 0.0 | no-trade band (trade all by default) |
| `factor_loading_limit` | `None` | off unless set; absolute portfolio factor-loading cap |
| `select_by` (verdict) | `"dsr"` | selection metric (deflated Sharpe) |
| warning band | `0.9 × limit` | severity `warning` vs `breach` cutoff |

## 10. Out of scope (YAGNI)

- Live trading / order routing — recommendations only.
- Tax-aware trade lists (SP1 roadmap Phase 7, deferred).
- New heavy dependencies — none.
- Re-optimizing or re-backtesting inside SP4 — it consumes existing primitives.

## 11. Gotchas (carried from the program)

- **`__init__`/existing-module cross-contamination**: `import quant_reporter` pulls every
  module, so concurrent agents editing re-exported modules break each other's import in
  pytest. The two new modules can parallelize; edits to `backtest_report.py`, `strategy.py`,
  and `__init__.py` are wired **sequentially and last**.
- **`conftest` edits additive only** — `make_synthetic_prices` is used everywhere.
- **Verify the suite via exit code + temp file**, not raw bash stdout.
- **No new dependencies** — CVaR-style work uses `scipy.linprog`/`scipy.optimize`, never cvxpy.
- **macOS BLAS spurious RuntimeWarnings** — `np.errstate` wrap on matmul-heavy blocks.
- **pandas 2.x** — `pct_change(fill_method=None)`.
- **Do NOT push `v2.1` to `main` or bump `__version__`** until SP4 is done and the user okays
  release; verify the live PyPI version before cutting.
