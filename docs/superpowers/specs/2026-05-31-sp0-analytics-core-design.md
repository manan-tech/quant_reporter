# SP0 — Analytics Core & Correctness (design spec)

**Date:** 2026-05-31 · **Branch:** `v2.1` · **Parent:** `2026-05-31-v2.1-roadmap.md`
**Status:** Approved design — ready for implementation plan.
**Maps to:** item 1 (modularity/redundancy) + the folded-in factor/attribution correctness fixes.

---

## 1. Goal

Establish a **single source of truth** for every computed quantity so the same metric can never be
computed two ways, and fix the latent correctness bugs the audit surfaced. Reports become **pure
assemblers** that read from the core and never recompute. The core is **offline-testable** (pure
functions taking injected `price_data`).

**Design approach (approved):** *Hybrid* — canonical **pure functions** are the source of truth
(usable standalone, primitives-first), plus a **lazy memoized `ctx.analytics` accessor** that
guarantees compute-once and identical values across report sections.

## 2. Non-goals (out of scope for SP0)

New strategies/primitives (SP1+), per-asset info layer (SP3), recommendation layer (SP4). SP0 ships
no new user-facing strategy — it is a correctness + structure release.

## 3. Module layout

| Module | Change | Canonical home for |
|--------|--------|--------------------|
| **`analytics.py`** *(new)* | `portfolio_returns()`, `compute_drawdown()`, `compute_metrics()`/`format_metrics()`, `PortfolioAnalytics` | portfolio returns/growth/terminal, drawdown, realized metrics, the `ctx.analytics` accessor |
| `metrics.py` | `compute_drawdown→(curve,scalar)`; `calculate_metrics` **clean-break** → numerics; keep `calculate_var_cvar`, `calculate_sortino_ratio` | drawdown + realized risk metrics (numeric) |
| `opt_core.py` | thread `risk_free_rate` everywhere; `get_portfolio_price` becomes a shim onto `portfolio_returns`; `get_optimization_inputs` stays the moments home | expected/model moments (log basis) |
| `monte_carlo.py` | absorb the `monte_carlo_report.py` fork; **seed**; accept `stress_shock` + `actual_path` | the one MC engine |
| `factor_models.py` | one OLS engine for static + rolling, excess returns, thread `risk_free_rate` | factor regression |
| `attribution.py` | Brinson honesty: real sector Brinson when benchmark weights supplied, else labeled "vs equal-weight baseline" | Brinson attribution |
| `report_context.py` | attach `ctx.analytics`; single rfr resolution | the accessor + rfr |
| `*_report.py`, `combined_report.py` | consume `ctx.analytics`; stop recomputing | pure assemblers |

## 4. Canonical functions (signatures)

```python
# analytics.py
@dataclass(frozen=True)
class ReturnsBundle:
    daily:  pd.DataFrame              # ['Portfolio','Benchmark'] simple daily returns
    growth: pd.DataFrame              # ['Portfolio','Benchmark'] Growth-of-$1 (starts at 1.0)
    weights_history: pd.DataFrame | None   # drifted weights from the rebalancer (None=closed-form)
    @property
    def terminal(self) -> float: ...  # growth['Portfolio'].iloc[-1] - 1

def portfolio_returns(price_data: pd.DataFrame, weights_dict: dict,
                      benchmark_col: str, rebalance_freq=None) -> ReturnsBundle:
    """Single home for portfolio daily returns + Growth-of-$1. Routes ALL freqs through
    simulate_rebalanced_portfolio; rebalance_freq=None == buy-and-hold."""

@dataclass(frozen=True)
class DrawdownResult:
    curve:  pd.Series
    max_dd: float
def compute_drawdown(cumulative: pd.Series) -> DrawdownResult: ...   # scalar == curve.min()

def compute_metrics(bundle: ReturnsBundle, risk_free_rate: float) -> dict[str, float]:
    """The REALIZED block, numeric. CAGR from growth; vol/Sharpe/Sortino from daily['Portfolio'];
    beta/alpha CAPM regression; VaR/CVaR daily historical; skew/kurt; rolling-60d Sharpe series."""
def format_metrics(metrics: dict[str, float]) -> dict[str, str]: ...

class PortfolioAnalytics:                       # attached as ctx.analytics
    def __init__(self, ctx): self._ctx = ctx
    @cached_property
    def returns(self)     -> ReturnsBundle: ...   # portfolio_returns(ctx..., ctx.rebalance_freq)
    @cached_property
    def drawdown(self)    -> DrawdownResult: ...   # compute_drawdown(self.returns.growth['Portfolio'])
    @cached_property
    def metrics(self)     -> dict[str, float]: ... # compute_metrics(self.returns, ctx.risk_free_rate)
    @cached_property
    def model_stats(self) -> dict[str, float]: ... # get_portfolio_stats(w, ctx.mean_returns, ctx.cov_matrix, ctx.risk_free_rate)
```

`calculate_max_drawdown(cum)` is retained as `compute_drawdown(cum).max_dd` (back-compat scalar).

## 5. Canonical basis — two labeled blocks (the dual-basis fix)

The bug is one label over two bases computed 5+ ways — not that model ≠ realized. Fix = one basis
**per concept**, computed once, **explicitly labeled**:

| Block | Basis | Quantities | Surfaced as |
|-------|-------|-----------|-------------|
| **Expected (model)** | log moments (`ctx.mean_returns/cov`, ×252) | expected return, vol, Sharpe | frontier ★, risk plots — **"Expected …"** |
| **Realized** | actual path, **simple** returns | realized vol, Sharpe, Sortino, Calmar, CAGR (geometric), β/α (CAPM), skew, kurt, VaR/CVaR (daily) | metrics card — **"Realized …"** |
| **Horizon (MC)** | simulated total returns | horizon VaR/CVaR | MC figures — **"Horizon (simulated)"** |

No report shows two bare-labeled "Sharpe Ratio" values again. Annualization uses **one** constant
(`×252` already in `get_optimization_inputs`); realized metrics annualize once with the same factor.

## 6. `rebalance_freq` honored

`portfolio_returns()` is the only producer of the Growth-of-$1 and routes through
`simulate_rebalanced_portfolio` for all frequencies. `None` = buy-and-hold (matches today's
`get_portfolio_price` within float tolerance → default reports unchanged). `'M'/'Q'/'Y'/int`
rebalances and yields a non-trivial `weights_history` (turnover lands in SP1).
`simulate_rebalanced_portfolio`'s `(Series, DataFrame)` signature is **unchanged**.

## 7. Factor / attribution correctness (folded in)

- **Brinson** (`attribution.py` / `factor_report.py:219`): the bug is that equal weights over the
  portfolio's own tickers are silently passed as `benchmark_weights` while the section claims to be
  benchmark-relative. A single benchmark **ETF** has no per-asset weights over the user's tickers, so
  a *true* benchmark-relative Brinson needs benchmark **sector/constituent weights** the library does
  not fetch. SP0 fix = **honesty + opt-in**: (a) if the user supplies benchmark sector weights (new
  optional `benchmark_weights`/sector-weight input on the ctx), run a real sector Brinson against them;
  (b) otherwise **relabel** the current output as a *"vs equal-weight baseline"* attribution — never
  present it as benchmark-relative. (True ETF-holdings attribution is a documented follow-up, not SP0.)
- **One OLS engine** (`factor_models.py`): static and rolling factor regressions share one
  implementation, on **excess** returns, threading `ctx.risk_free_rate`. Drop the hand-rolled
  `(XᵀX)⁻¹XᵀY` rolling path on raw returns.
- Factor regression runs on the **canonical** portfolio returns, not a renormalized Growth-of-$1
  `pct_change`.

## 8. Combined report — faithful & fail-loud

- Replace silent `try/except → skip` with a **visible** "⚠ Module failed: <reason>" section
  (deterministic section count). Add `strict=False` (`True` re-raises).
- Forward `num_simulations/time_horizon/initial_investment` + `actual_path` to
  `compute_monte_carlo_analysis` (no more locked 5000/252/$10k).
- Remove dead `desc=` kwarg. De-duplicate the correlation heatmap rendered twice
  (`portfolio_report:137` + `optimization_report:269`).
- Strict superset of all standalone sections it includes.

## 9. Risk-free rate unification

One resolution path with **one** documented fallback (reconcile `build_context` `0.02` vs
`get_risk_free_rate` `0.06` → single value, default **0.02**). `ctx.risk_free_rate` is threaded into
`get_portfolio_stats` (Monte Carlo path) and `factor_models` — currently both ignore it.

## 10. Report-assembler refactor (per module)

- **portfolio_report:** remove local `plot_drawdown` + the `all_daily_returns/all_cumulative_returns`
  recompute; use `ctx.analytics.{returns,drawdown,metrics}`; render via `format_metrics`; remove the
  shadowing local `plot_correlation_heatmap`; label the correlation window (currently train-only,
  unlabeled — GAP 7).
- **optimization_report:** stop discarding `plot_data`; delete inline rolling-Sharpe (`241-243`) and
  inline drawdown (`247-249`); share the optimizer suite with validation (compute once).
- **validation_report:** delete `_to_num` string round-trip (consume numerics); thread
  `denoise_cov/n_components` into `run_rolling_windows`; OOS drawdown via `compute_drawdown`.
- **monte_carlo_report:** consume the merged engine; forward params; seed; pass `ctx.risk_free_rate`;
  relabel horizon VaR/CVaR.
- **factor_report:** §7 fixes.
- **combined_report:** §8 fixes.

## 11. Backward compat & versioning

- **Preserved:** all public API except the documented `calculate_metrics` break (now numeric;
  `format_metrics()` added).
- **Behavior changes (→ CHANGELOG `[2.1.0]`):** `rebalance_freq` now affects output; report numbers
  shift to consistent/labeled values (Expected vs Realized); Monte Carlo is seeded (reproducible);
  Brinson now measured vs the real benchmark; one risk-free fallback.
- `simulate_rebalanced_portfolio` signature unchanged. Version cut deferred to release.

## 12. Testing strategy (makes the fix stick)

- **Offline deterministic fixture:** synthetic multi-asset price `DataFrame` (seeded), no yfinance —
  the pure core is finally testable without network.
- **Golden/regression:** pin canonical numeric outputs of `compute_metrics`, `compute_drawdown`,
  `portfolio_returns`, model_stats on the fixture.
- **Cross-section consistency guards (anti-divergence net):**
  `drawdown.max_dd == drawdown.curve.min()`; metrics-card realized vol `==` vol derived from
  `ctx.analytics.returns`; combined-report portfolio Sharpe `==` standalone Sharpe; the heatmap is
  rendered once.
- **MC determinism:** fixed seed reproducible; `stress` vs clean differ only by the shock.
- **`rebalance_freq`:** `None` matches closed-form buy-and-hold within tol; `'M'` ⇒ `weights_history`
  varies and (SP1 will assert turnover>0).
- **Factor/attribution:** Brinson against a *supplied* sector-weight benchmark matches a hand-computed
  value; the no-benchmark-weights path is labeled "vs equal-weight baseline" (not benchmark-relative);
  static vs rolling betas agree at the overlapping window.
- Existing network-gated smoke tests stay (skip offline).

## 13. Risks & mitigations

| Risk | Mitigation |
|------|-----------|
| Report numbers change surprises users | CHANGELOG documents every behavior change; golden tests capture the new canonical values |
| `None` rebalance numerically diverges from old closed-form | Consistency test asserts within tolerance; keep `get_portfolio_price` math as the buy-and-hold reference |
| Large refactor touches all reports at once | Land core + tests first, then refactor reports module-by-module behind the green consistency guards |
| `cached_property` on a frozen/likely-immutable ctx | `PortfolioAnalytics` is a separate object holding a ctx ref, not on the frozen dataclass itself |

## 14. Definition of done

1. `analytics.py` exists with `portfolio_returns`, `compute_drawdown`, `compute_metrics`,
   `format_metrics`, `PortfolioAnalytics`; re-exported from `__init__`.
2. `ctx.analytics` available; all 5 report modules consume it and contain **no** local recompute of
   returns/growth/drawdown/rolling-Sharpe/metrics.
3. Monte Carlo fork deleted; one seeded engine; combined forwards params + actual overlay.
4. `rebalance_freq` demonstrably changes output; Brinson uses the real benchmark; one OLS engine; one
   rfr fallback.
5. Offline test suite (golden + consistency guards + MC determinism) passes with **no network**.
6. CHANGELOG `[2.1.0]` documents all behavior changes. Existing 37 tests still pass.
