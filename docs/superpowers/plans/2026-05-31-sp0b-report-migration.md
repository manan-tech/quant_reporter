# SP0 Part B — Report Migration & Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (Tasks 1, 7, 8) and a parallel agent team / Workflow (Tasks 2-6, independent files). Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the 5 report modules + combined report to consume the `ctx.analytics` core (so every report reads identical, compute-once values), fold in the factor/attribution + Monte-Carlo correctness fixes, make `combined` fail-loud, and land the `calculate_metrics` clean break — leaving zero inline recompute and the full suite green.

**Architecture:** Reports become pure assemblers over `ctx.analytics` (from Part A). An offline `build_context_from_prices` factory makes report assembly testable without yfinance. The 5 standalone migrations touch disjoint files and run in parallel; `combined` and the clean break are the sequential tail (combined depends on all five; the clean break depends on no module importing legacy `calculate_metrics`).

**Tech Stack:** Python 3.9, numpy/pandas/scipy/statsmodels, pytest. No new dependencies.

**Grounding:** Exact recompute sites are in `docs/superpowers/specs/2026-05-31-v2.1-audit-research-raw.json` (`audits[*].recomputations`, `auditSynth`) and summarized in `…-sp0-analytics-core-design.md` §10. Each migration task cites them.

**Test convention:** from repo root, `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" <args>`. Branch: `v2.1`.

**`ctx.analytics` API (from Part A — what reports now consume):**
```python
ctx.analytics.returns      # ReturnsBundle: .daily(DF['Portfolio','Benchmark']), .growth(DF), .terminal, .weights_history
ctx.analytics.drawdown     # DrawdownResult: .curve (Series), .max_dd (float)
ctx.analytics.metrics      # dict[str,float] REALIZED block; render via format_metrics(...)
ctx.analytics.model_stats  # {'Expected Return','Expected Volatility','Expected Sharpe'} from log moments
# helpers: from quant_reporter.analytics import compute_metrics, format_metrics, compute_drawdown
```

---

### Task 1: Offline context factory `build_context_from_prices` (foundational, sequential)

**Why:** report `compute_*` functions take a `ReportContext`; today the only way to build one is `build_context`, which fetches from yfinance. Extracting the post-fetch assembly lets every later task test report assembly offline with the Part A `synthetic_prices` fixture. Also a genuine power-user feature (bring-your-own price data).

**Files:**
- Modify: `src/quant_reporter/report_context.py` (extract `_assemble_context`; add `build_context_from_prices`)
- Modify: `src/quant_reporter/__init__.py` (export `build_context_from_prices`)
- Test: `test/test_context_factory.py`

- [ ] **Step 1: Write the failing test** — `test/test_context_factory.py`:
```python
import pandas as pd
from conftest import make_synthetic_prices
from quant_reporter.report_context import build_context_from_prices


def test_build_context_from_prices_offline():
    prices = make_synthetic_prices()
    ctx = build_context_from_prices(
        prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK",
        train_start="2021-01-01", train_end="2022-12-31",
    )
    # no network, fully assembled, analytics wired
    assert ctx.friendly_benchmark == "BMK"
    assert list(ctx.mean_returns.index) == ["AAA", "BBB", "CCC"]
    assert ctx.analytics.metrics["Max Drawdown"] == ctx.analytics.drawdown.max_dd
    assert ctx.price_data_train.index[-1] <= pd.to_datetime("2022-12-31")
```

- [ ] **Step 2: Run → FAIL** (`cannot import name 'build_context_from_prices'`).
Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_context_factory.py -v`

- [ ] **Step 3: Refactor `report_context.py`.** Extract everything in `build_context` AFTER the `get_data(...)` fetch (current steps 3-6: display-name mapping, column ordering / dropping missing tickers, train/test split, `get_optimization_inputs`, `ReportContext(...)` construction, `ctx.analytics = PortfolioAnalytics(ctx)`) into a new helper:
```python
def _assemble_context(price_data_full, portfolio_dict, benchmark_ticker, train_start,
                      train_end, test_start, test_end, full_start, full_end, rfr,
                      display_names, sector_map, sector_caps, sector_mins,
                      bl_views, bl_view_confidences, bl_relative_views,
                      bl_relative_view_confidences, rebalance_freq,
                      denoise_cov, n_components):
    # ... the current post-fetch body (steps 3-6) verbatim, returning the wired ctx ...
```
`build_context` keeps steps 1-2 (date resolution, rfr resolution) + the `get_data` fetch, then `return _assemble_context(price_data_full, ...)`. Add the new entry point that skips the fetch:
```python
def build_context_from_prices(price_data_full, portfolio_dict, benchmark_ticker,
                              train_start, train_end, risk_free_rate="auto", **opts):
    """Build a ReportContext from an already-fetched price DataFrame (no network).
    price_data_full must contain every portfolio ticker + the benchmark column."""
    # resolve dates (same as build_context steps 1) and rfr (step 2), then:
    return _assemble_context(price_data_full.copy(), portfolio_dict, benchmark_ticker,
                             train_start, train_end, test_start, test_end, full_start,
                             full_end, rfr, **<opts mapped to the named params>)
```
Behaviour of `build_context` must be unchanged (verified by the existing 47 tests). Export `build_context_from_prices` from `__init__.py` next to `build_context`.

- [ ] **Step 4: Run → PASS** (new test) and **full suite green** (47 + 1).
Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`  → expect `48 passed` (+ smoke skip offline).

- [ ] **Step 5: Commit** `feat: build_context_from_prices offline factory (extract _assemble_context)`.

---

## Tasks 2-6 — Report migrations (PARALLEL, one agent per module)

**Shared contract for every migration task:**
1. Read the module. Replace each audit-flagged recompute with the matching `ctx.analytics.*` value. Render numeric metrics through `format_metrics(ctx.analytics.metrics)`.
2. Do NOT change the module's public `create_*`/`compute_*` signatures or its section/HTML structure (same titles, same plots) — only the *source* of the numbers.
3. Add an **offline test** in the module's test file: build a ctx via `build_context_from_prices(make_synthetic_prices(), …)`, call `compute_<kind>_analysis(ctx)`, assert it returns a non-empty section list **and** that a representative displayed number equals the `ctx.analytics` source (e.g. the realized-Sharpe cell equals `format_metrics(ctx.analytics.metrics)["Realized Sharpe"]`). For modules that fetch external data (factor), mock that fetch (see Task 6).
4. Keep the network-gated smoke test working when network is available.
5. Commit `refactor(<module>): consume ctx.analytics; remove inline recompute`.

### Task 2: `portfolio_report.py`
**Audit targets to remove** (`audits[portfolio].recomputations`): local `plot_drawdown` (`portfolio_report.py:35-52`) → use `ctx.analytics.drawdown.curve`; `all_daily_returns`/`all_cumulative_returns` (`:125-126`) → derive from `ctx.analytics.returns`; metrics dashboard → `format_metrics(ctx.analytics.metrics)`; remove the shadowing local `plot_correlation_heatmap` (`:22-33`, use the `opt_plotting` one). Label the correlation window (currently train-only, unlabeled — GAP 7).

### Task 3: `optimization_report.py`
Stop discarding `plot_data` (`:219`); delete inline rolling-Sharpe (`:241-243`) and inline drawdown (`:247-249`) → `ctx.analytics`. Benchmark normalization (`:212`) via the shared growth. (Optimizer-suite sharing with validation is optional polish — leave the optimizer calls intact this task.)

### Task 4: `validation_report.py`
Delete the `_to_num` string round-trip (`:31-38`,`116`,`194`) — consume numeric `ctx.analytics.metrics` / `compute_metrics` directly. OOS drawdown (`:209-212`) → `compute_drawdown`. Thread `denoise_cov/n_components` into `run_rolling_windows` so IS and rolling use the same covariance treatment.

### Task 5: `monte_carlo_report.py` (+ engine merge)
Delete the forked engine in `monte_carlo_report.py:13-185`; make `monte_carlo.py` the single engine — add an optional `seed` (default e.g. 42, `np.random.seed`/`default_rng`), keep `stress_shock`, restore the `actual_path` overlay + `plot_simulation_paths`. `compute_monte_carlo_analysis` imports from `monte_carlo.py` and passes `ctx.risk_free_rate` into `get_portfolio_stats`. Relabel simulated VaR/CVaR as **"Horizon (simulated)"**. **TDD:** determinism test — same seed ⇒ identical paths; `stress` vs clean differ only by the shock.

### Task 6: `factor_report.py` (+ factor/attribution correctness)
Regress on the **canonical** `ctx.analytics.returns` (not a renormalized Growth-of-$1 `pct_change`). Unify static + rolling factor regression on ONE statsmodels excess-return engine, threading `ctx.risk_free_rate` (delete the hand-rolled raw-return rolling path `:25-71`). **Brinson honesty** (`attribution`/`factor_report.py:219`): if benchmark sector weights are supplied (new optional input), run real sector Brinson; otherwise relabel the output **"vs equal-weight baseline"** (never claim benchmark-relative). **TDD (offline, mock the Fama-French fetch):** Brinson vs a supplied sector-weight benchmark matches a hand-computed value; static vs rolling betas agree at the overlapping window.

---

### Task 7: `combined_report.py` — faithful & fail-loud (sequential, after 2-6)

**Files:** Modify `src/quant_reporter/combined_report.py`; Test `test/test_combined_report.py`.

- [ ] **Step 1 (failing test):** with an offline ctx, a delegate that raises produces a **visible** "⚠ Module failed" section (not a silent skip); `strict=True` re-raises.
```python
def test_combined_failloud(monkeypatch):
    # patch one compute_* to raise; assert a section titled like "⚠ ... failed" appears,
    # and that create_combined_report(..., strict=True) raises.
```
- [ ] **Step 2 → FAIL.**
- [ ] **Step 3:** replace silent `try/except → logger.error` (`combined_report.py:99-126`) with an error-section append; add `strict=False` param (`True` re-raises). Forward `num_simulations/time_horizon/initial_investment` + `actual_path` to `compute_monte_carlo_analysis` (no more locked 5000/252/$10k). Remove dead `desc=` kwarg. De-duplicate the correlation heatmap (now rendered once, since portfolio_report dropped its local one in Task 2).
- [ ] **Step 4 → PASS** + full suite green.
- [ ] **Step 5: Commit** `feat(combined): fail-loud sections, forward MC params, dedupe heatmap, drop dead kwarg`.

---

### Task 8: `calculate_metrics` clean break (sequential, last)

**Precondition:** Tasks 2-6 left no report importing legacy `calculate_metrics`. Verify: `grep -rn "calculate_metrics" src/` shows only the definition (and `opt_core` import if still present) — migrate any stragglers first.

**Files:** Modify `src/quant_reporter/metrics.py`, `src/quant_reporter/__init__.py`, `opt_core.py` (drop the unused `from .metrics import calculate_metrics` if present); Test: full suite.

- [ ] **Step 1 (failing test):** `test/test_metrics.py` asserts `calculate_metrics` no longer exists (or is the numeric one) — add `def test_calculate_metrics_removed(): assert not hasattr(quant_reporter, "calculate_metrics")`.
- [ ] **Step 2 → FAIL.**
- [ ] **Step 3:** remove the string-returning `calculate_metrics` from `metrics.py` and its `__init__` export (the numeric path is `analytics.compute_metrics`). Fix any remaining import.
- [ ] **Step 4 → PASS** + **full suite green**; `import quant_reporter` smoke OK.
- [ ] **Step 5: Commit** `feat!: remove string-returning calculate_metrics (clean break; use compute_metrics)`. Update CHANGELOG `[2.1.0]` Changed: "Breaking: `calculate_metrics` removed; use `compute_metrics` (numeric) + `format_metrics`."

---

## Self-Review

**Spec coverage (design §4-§10):** ✅ reports → assemblers (T2-6); ✅ drawdown/metrics/returns single source consumed (T2-6); ✅ MC merge+seed+labels (T5); ✅ factor one-OLS + Brinson honesty (T6); ✅ combined fail-loud + forward params + dedupe + drop desc (T7); ✅ rebalance_freq already honored in Part A (reports inherit it via `ctx.analytics.returns`); ✅ clean break (T8); ✅ offline testability (T1). **Deferred (noted):** optimizer-suite de-dup between optimization/validation (optional polish, not required for single-source correctness).

**Placeholder scan:** Tasks 2-6 are deliberate *contracts* (audit-line targets + ctx.analytics substitution + a concrete test contract), executed by per-module agents holding the full file — not "TODO". T1/T7/T8 have concrete code/test/commands. No "figure it out" steps.

**Type consistency:** `ctx.analytics.{returns,drawdown,metrics,model_stats}`, `ReturnsBundle`, `DrawdownResult`, `compute_metrics`/`format_metrics`, `build_context_from_prices` all match Part A's shipped names.

**Parallel-safety:** Tasks 2-6 modify disjoint files (`portfolio_report.py`, `optimization_report.py`, `validation_report.py`, `monte_carlo_report.py`+`monte_carlo.py`, `factor_report.py`+`attribution.py`+`factor_models.py`). No two tasks touch the same file → safe to run as a parallel agent team. T7 reads all five (run after). T8 last.
