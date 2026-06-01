# SP-Strategy — Strategy layer, backtest reporting, shared metrics/objectives

**Date:** 2026-06-01
**Branch:** `v2.1`
**Status:** Design approved; ready for implementation plan.
**Depends on:** SP0 (analytics core), SP1 (`simulate_strategy`, PSR/DSR), SP2 (signals/sizing/risk/tilts), SP3 (per-asset info).
**Feeds:** SP4 (recommendation layer consumes `BacktestResult`).

---

## 1. Goal & posture

Give users a first-class **strategy → backtest → report** loop, plus a consolidated
library of common quant functions. Three coupled deliverables in one SP:

1. **Strategy abstraction** — define a strategy once, backtest it uniformly.
2. **Interactive backtest report** — wealth, rebalances, weights-over-time, turnover,
   costs, drawdown, rolling stats, full metrics, OOS stats.
3. **Shared library** — performance/risk/loss **metrics** + optimization **objectives**,
   usable by both retail and institutional users.

Posture is unchanged from the v2.1 roadmap: **primitives-first, no black box, no new
heavy deps** (plotly/numpy/pandas/scipy/sklearn already pinned). Opinions stay out of the
primitives; SP4 owns `recommend=True`.

## 2. The strategy model — function-first, optional wrapper

A strategy is **any callable** `(prices, **params) -> weights`, where `weights` is either
a `dict` (static one-shot) or a `DataFrame` (dated schedule). This keeps the pure-function
house style and lets institutional users write anything.

An optional thin wrapper provides naming/discoverability without forcing classes:

```python
@dataclass
class Strategy:
    name: str
    fn: Callable                      # (prices, **params) -> weights
    params: dict = field(default_factory=dict)
    def weights(self, prices):
        return self.fn(prices, **self.params)
```

`backtest()` accepts **any** of: a bare callable, a `Strategy`, or a raw `dict`/`DataFrame`
of weights (static). This is the load-bearing flexibility point.

## 3. Backtest runner + result

`backtest()` is a thin orchestration over the existing SP1 `simulate_strategy` — it does
**not** replace it. It resolves the strategy to weights, runs `simulate_strategy`, aligns
the benchmark, and wraps everything in a result object.

```python
def backtest(strategy, prices, *, rebalance="M", cost_model=None,
             benchmark=None, initial_value=1.0, name=None) -> BacktestResult:
    """
    strategy : callable (prices, **params)->weights | Strategy | dict | DataFrame
    prices   : DataFrame of asset prices (may include `benchmark` column)
    benchmark: column name in `prices` OR a price Series; optional
    """
```

```python
@dataclass
class BacktestResult:
    name: str
    wealth: pd.Series              # Growth-of-$1 (from simulate_strategy)
    weights: pd.DataFrame          # realized weights over time (rebalances)
    blotter: pd.DataFrame          # trades at each rebalance
    turnover: pd.Series
    cost_drag: float
    benchmark: pd.Series | None    # Growth-of-$1 of the benchmark, aligned

    @cached_property
    def returns(self) -> pd.Series          # wealth.pct_change().dropna()
    @cached_property
    def metrics(self) -> dict               # summary_metrics(returns, benchmark, rfr)
    @cached_property
    def oos_stats(self) -> dict             # {'psr':..., 'dsr':...} on realized returns

    def report(self, path=None, open_browser=False) -> str   # interactive HTML (returns html str)
    def plot_wealth(self) -> go.Figure
    def plot_drawdown(self) -> go.Figure
    def plot_weights(self) -> go.Figure
    def plot_turnover(self) -> go.Figure
    def plot_rolling(self) -> go.Figure
```

`backtest()` returns a single result; a helper `backtest_many(strategies: dict, prices, ...)
-> dict[str, BacktestResult]` runs several and is what the multi-strategy comparison panel
and `compare_strategies_oos` consume.

## 4. Interactive report (`backtest_report.py`)

`build_backtest_report(result | results, path=None) -> html` assembles a plotly +
`html_builder` page. Also exposed as `create_backtest_report(...)` for naming parity with
the existing `create_*_report` generators. Panels (all reuse the established plotly_white
style):

1. **Summary KPI strip** — terminal wealth, CAGR, ann. vol, Sharpe, Sortino, Calmar,
   max drawdown, turnover, cost drag, PSR, DSR.
2. **Growth-of-$1** — strategy vs benchmark.
3. **Underwater drawdown** — from `compute_drawdown`.
4. **Weights-over-time** — stacked area; rebalances visible.
5. **Turnover per rebalance** (bar) + **cumulative cost drag**.
6. **Rolling Sharpe & rolling vol** (e.g. 63-day windows).
7. **Monthly-returns heatmap + distribution**.
8. **Full metrics table** — the shared `summary_metrics` output.
9. **Trade blotter table** — `blotter` rows.
10. **Multi-strategy OOS comparison** — only when `results` is a dict of >1; shows the
    `compare_strategies_oos` summary (SR/PSR/DSR) and overlaid wealth curves.

The report is one consumer of the primitives — every number comes from `BacktestResult`
fields or the shared metrics library; the report never recomputes independently.

## 5. Shared library

### 5a. `metrics.py` (extend — additive, no breakage)

Add pure functions, each `(...) -> float`, on **simple periodic returns**, with
`periods_per_year=252` and `risk_free_rate`/`mar` where relevant:

| Function | Definition |
|----------|-----------|
| `cagr(returns)` | geometric annualized growth |
| `annual_volatility(returns)` | `std(ddof=1) * sqrt(ppy)` |
| `sharpe(returns, rfr)` | annualized excess mean / ann. vol |
| `sortino(returns, rfr)` | annualized excess mean / downside deviation |
| `calmar(returns)` | `cagr / abs(max_drawdown)` |
| `omega(returns, threshold=0)` | Σ gains above / Σ losses below threshold |
| `max_drawdown(returns)` | min of underwater curve (scalar; from growth) |
| `avg_drawdown(returns)` | mean of underwater curve over drawdown periods |
| `ulcer_index(returns)` | `sqrt(mean(drawdown²))` |
| `value_at_risk(returns, c=0.95)` | historical VaR (positive loss) |
| `conditional_var(returns, c=0.95)` | historical CVaR (positive loss) |
| `downside_deviation(returns, mar=0)` | ann. std of returns below `mar` |
| `tracking_error(returns, benchmark)` | ann. std of active returns |
| `information_ratio(returns, benchmark)` | ann. active mean / tracking error |
| `hit_rate(returns)` | fraction of periods > 0 |
| `win_loss_ratio(returns)` | mean(wins) / abs(mean(losses)) |
| `tail_ratio(returns)` | `abs(q95) / abs(q05)` |
| `skewness(returns)` / `kurtosis(returns)` | sample skew / kurtosis |

Plus a convenience aggregator the report uses:

```python
def summary_metrics(returns, benchmark=None, risk_free_rate=0.02,
                    periods_per_year=252) -> dict   # ordered, named metric dict
```

Existing `compute_drawdown`, `DrawdownResult`, `calculate_sortino_ratio`,
`calculate_var_cvar`, `calculate_max_drawdown` stay **unchanged**. `analytics.compute_metrics`
and the existing reports are untouched. New functions are the clean canonical surface;
internal sharing (e.g. `max_drawdown` delegating to `compute_drawdown`) avoids duplication.

### 5b. `objectives.py` (new module)

Minimize-ready objective/loss functions for `find_optimal_portfolio` and custom fits:

```python
neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02) -> float
variance(weights, mean_returns, cov_matrix, risk_free_rate=0.02) -> float
cvar_objective(weights, returns_matrix, confidence=0.95) -> float
tracking_error_objective(weights, returns_matrix, benchmark_returns) -> float
mean_squared_error(y_true, y_pred) -> float
mean_absolute_error(y_true, y_pred) -> float
```

Existing `opt_core.objective_neg_sharpe` / `objective_min_variance` keep working
(unchanged signatures); the new module is the discoverable home and adds the CVaR / TE /
regression objectives. Where it makes sense, `objectives.neg_sharpe` may delegate to the
existing `opt_core` helper to avoid divergence.

## 6. Prebuilt strategies (`strategies.py`)

Each is a function `(prices, **params) -> weights` composing SP1/SP2 primitives, plus a
`REGISTRY: dict[str, Callable]` for discoverability:

| Name | Composition |
|------|-------------|
| `equal_weight` | `1/N` static |
| `inverse_vol` | `inverse_volatility_weights` |
| `min_variance` | Ledoit-Wolf cov → SLSQP min-variance |
| `risk_parity` | `optimize_risk_budget` on Ledoit-Wolf cov |
| `max_sharpe` | `get_optimization_inputs` → `find_optimal_portfolio(neg_sharpe)` |
| `trend_following` | TS-momentum + MA-crossover ensemble → `volatility_target_positions` |
| `cross_sectional_momentum` | `cross_sectional_momentum_score` → long top, vol-target sizing |
| `vol_target_overlay(base_fn, target_vol)` | higher-order wrapper sizing any base strategy |

Strategies that produce a **dated schedule** must be look-ahead safe (signals lagged ≥1,
schedule decided on data ≤ d−1) — same discipline SP1/SP2 pin with hypothesis tests.

## 7. Files & build order

**New modules** (brand-new — parallelizable until the `__init__` wire-up):
- `src/quant_reporter/strategy.py` — `Strategy`, `backtest`, `backtest_many`, `BacktestResult`
- `src/quant_reporter/backtest_report.py` — `build_backtest_report` / `create_backtest_report` + panel figures
- `src/quant_reporter/strategies.py` — prebuilt strategies + `REGISTRY`
- `src/quant_reporter/objectives.py` — objective/loss surface

**Edit (sequential — shared re-export surface):**
- `src/quant_reporter/metrics.py` — add the measurement functions + `summary_metrics`
- `src/quant_reporter/__init__.py` — wire all new exports (orchestrator does this last)

**Tests** (mirror existing one-file-per-module style, offline via `make_synthetic_prices`):
- `test/test_metrics_lib.py`, `test/test_objectives.py`, `test/test_strategies.py`,
  `test/test_strategy.py` (runner + result), `test/test_backtest_report.py`.

**Example:**
- `examples/example_strategy_report.py` — build a strategy, `backtest()`, write the HTML report.

Dependency spine: `metrics.py` + `objectives.py` first (leaf deps) → `strategies.py`
(uses primitives) → `strategy.py` (uses `simulate_strategy` + metrics) → `backtest_report.py`
(uses result + metrics + plotting) → `__init__` wire-up + example.

## 8. Testing & safety

- **All additive; the 233 existing tests must stay green** (exit-0 + temp-file verification,
  not raw stdout).
- **Metrics**: golden-value tests against hand-computed references; edge cases (empty,
  single-obs, all-positive/all-negative, zero-vol) return NaN/0 sensibly, never raise.
- **Objectives**: pinned values + "is minimizable by scipy" smoke tests.
- **Strategies**: weights sum to 1 (long-only ones), all tickers present, reproducible;
  **hypothesis future-shuffle causality** on any schedule-producing strategy.
- **Runner/result**: frictionless `backtest()` wealth matches `simulate_strategy` directly;
  `metrics`/`oos_stats` cached and consistent; accepts callable / `Strategy` / dict / DataFrame.
- **Report**: generated offline; assert HTML non-empty, contains each panel's title/anchor,
  and that figures have data; multi-strategy path exercises the comparison panel.
- **No BLAS-warning leakage** — wrap any matmul-heavy block in `np.errstate(...)` per the
  established pattern.

## 9. Out of scope (YAGNI)

- Live / intraday / order-routing — backtest only.
- SP4 recommendation verdicts, trade lists, alerts — separate SP that consumes this.
- New heavy dependencies — none.
- Tax lots (SP1 roadmap Phase 7, optional, deferred).

## 10. Gotchas (carried from the program)

- **`__init__` cross-contamination**: concurrent agents editing re-exported modules break
  each other's `import quant_reporter` in pytest. Brand-new modules can parallelize; the
  orchestrator wires `__init__`/`metrics.py` edits **sequentially** at the end.
- **`conftest` edits additive only** — `make_synthetic_prices` is used everywhere.
- **Look-ahead bias is the dominant risk** — every schedule-producing strategy gets a
  future-shuffle property test.
- **Do NOT push `v2.1` or bump `__version__`** — that's the final release step (user's PyPI token).
