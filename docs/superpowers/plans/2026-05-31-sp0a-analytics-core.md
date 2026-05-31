# SP0 Part A — Analytics Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the canonical, offline-testable analytics core (`portfolio_returns`, `compute_drawdown`, `compute_metrics`, `format_metrics`, `PortfolioAnalytics`/`ctx.analytics`) that becomes the single source of truth — **without touching the report modules yet**, so the existing 37 tests stay green.

**Architecture:** Hybrid — pure functions are the source of truth (usable standalone); a lazy `cached_property`-based `PortfolioAnalytics` accessor attached as `ctx.analytics` guarantees compute-once. `portfolio_returns` routes all rebalance frequencies through the existing `simulate_rebalanced_portfolio`. New numeric `compute_metrics` is **added alongside** the existing string-returning `calculate_metrics` (the clean break happens in Plan B once reports migrate).

**Tech Stack:** Python 3.9, numpy, pandas≥2.2, scipy, pytest (offline, deterministic fixtures).

**Scope:** Part A = the core + offline tests only. Part B (separate plan) migrates the 5 reports + combined + factor/attribution and removes the old `calculate_metrics`. See `2026-05-31-sp0-analytics-core-design.md`.

**Conventions:** Repo uses `test/` (not `tests/`). All test commands run from repo root as:
`PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" <args>` (the `-o addopts=""` disables the `setup.cfg` `--cov`/`fail_under=90` gate).

---

### Task 1: Offline deterministic price fixture

**Files:**
- Create: `test/conftest.py`
- Test: `test/test_synthetic_fixture.py`

- [ ] **Step 1: Write the fixture + the failing test**

`test/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


def make_synthetic_prices(seed=42, n_days=756, assets=("AAA", "BBB", "CCC"), benchmark="BMK"):
    """Deterministic GBM price panel: assets + a benchmark column, ~3 business years."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    cols = list(assets) + [benchmark]
    frame = {}
    for i, c in enumerate(cols):
        mu = 0.0003 + 0.0001 * i
        sig = 0.010 + 0.002 * i
        daily = rng.normal(mu, sig, n_days)
        frame[c] = 100.0 * np.exp(np.cumsum(daily))
    return pd.DataFrame(frame, index=dates)


@pytest.fixture
def synthetic_prices():
    return make_synthetic_prices()
```

`test/test_synthetic_fixture.py`:
```python
import pandas as pd
from conftest import make_synthetic_prices


def test_synthetic_prices_deterministic():
    a = make_synthetic_prices()
    b = make_synthetic_prices()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["AAA", "BBB", "CCC", "BMK"]
    assert len(a) == 756
    assert (a > 0).all().all()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_synthetic_fixture.py -v`
Expected: PASS (2 imports resolve; `conftest` importable because `test/` is on the path).

- [ ] **Step 3: Commit**

```bash
git add test/conftest.py test/test_synthetic_fixture.py
git commit -m "test: deterministic offline price fixture for SP0 core"
```

---

### Task 2: `compute_drawdown` (curve + scalar) in metrics.py

**Files:**
- Modify: `src/quant_reporter/metrics.py` (add `DrawdownResult` + `compute_drawdown`; make `calculate_max_drawdown` delegate — currently `metrics.py:5-9`)
- Test: `test/test_analytics_core.py`

- [ ] **Step 1: Write the failing test**

`test/test_analytics_core.py`:
```python
import numpy as np
import pandas as pd
import pytest
from quant_reporter.metrics import compute_drawdown, calculate_max_drawdown, DrawdownResult


def test_compute_drawdown_scalar_equals_curve_min():
    cum = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6])
    dd = compute_drawdown(cum)
    assert isinstance(dd, DrawdownResult)
    assert dd.max_dd == dd.curve.min()
    assert dd.max_dd == pytest.approx((0.6 - 1.2) / 1.2)  # -0.5 from the 1.2 peak


def test_calculate_max_drawdown_backcompat_scalar():
    cum = pd.Series([1.0, 1.2, 0.9])
    assert calculate_max_drawdown(cum) == pytest.approx((0.9 - 1.2) / 1.2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_drawdown'`.

- [ ] **Step 3: Implement in metrics.py**

Add near the top of `src/quant_reporter/metrics.py` (after the imports):
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownResult:
    curve: pd.Series
    max_dd: float


def compute_drawdown(cumulative_returns):
    """Underwater curve + scalar max drawdown from one cumulative (Growth-of-$1) series."""
    peak = cumulative_returns.cummax()
    curve = (cumulative_returns - peak) / peak
    return DrawdownResult(curve=curve, max_dd=float(curve.min()))
```

Replace the existing `calculate_max_drawdown` (`metrics.py:5-9`) body with a delegate:
```python
def calculate_max_drawdown(cumulative_returns):
    """Backward-compatible scalar max drawdown (delegates to compute_drawdown)."""
    return compute_drawdown(cumulative_returns).max_dd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/metrics.py test/test_analytics_core.py
git commit -m "feat: compute_drawdown returns curve + scalar; calculate_max_drawdown delegates"
```

---

### Task 3: `ReturnsBundle` + `portfolio_returns` in analytics.py

**Files:**
- Create: `src/quant_reporter/analytics.py`
- Test: `test/test_analytics_core.py` (append)

- [ ] **Step 1: Write the failing test (append to test/test_analytics_core.py)**

```python
from quant_reporter.analytics import portfolio_returns, ReturnsBundle
from quant_reporter.opt_core import get_portfolio_price


def test_portfolio_returns_buy_and_hold_matches_closed_form(synthetic_prices):
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    rb = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq=None)
    closed = get_portfolio_price(synthetic_prices[["AAA", "BBB", "CCC"]], w)
    assert isinstance(rb, ReturnsBundle)
    assert rb.growth["Portfolio"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    # iterative buy&hold == closed-form buy&hold
    assert rb.growth["Portfolio"].iloc[-1] == pytest.approx(closed.iloc[-1], rel=1e-6)
    assert rb.terminal == pytest.approx(closed.iloc[-1] - 1.0, rel=1e-6)


def test_portfolio_returns_rebalance_differs_from_buy_and_hold(synthetic_prices):
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    bh = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq=None)
    mo = portfolio_returns(synthetic_prices, w, "BMK", rebalance_freq="M")
    assert mo.weights_history is not None
    assert not np.isclose(mo.terminal, bh.terminal)
    assert list(mo.daily.columns) == ["Portfolio", "Benchmark"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.analytics'`.

- [ ] **Step 3: Implement analytics.py**

`src/quant_reporter/analytics.py`:
```python
"""Canonical analytics core — the single source of truth for portfolio returns,
growth, drawdown, and realized metrics. Pure functions (standalone) + a memoized
PortfolioAnalytics accessor attached to ReportContext as ctx.analytics."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from scipy import stats

from .rebalancing import simulate_rebalanced_portfolio
from .metrics import compute_drawdown, calculate_sortino_ratio, calculate_var_cvar


@dataclass(frozen=True)
class ReturnsBundle:
    daily: pd.DataFrame                 # ['Portfolio', 'Benchmark'] simple daily returns
    growth: pd.DataFrame               # ['Portfolio', 'Benchmark'] Growth-of-$1 (start 1.0)
    weights_history: "pd.DataFrame | None"

    @property
    def terminal(self) -> float:
        return float(self.growth["Portfolio"].iloc[-1] - 1.0)


def portfolio_returns(price_data, weights_dict, benchmark_col, rebalance_freq=None):
    """Single producer of the portfolio Growth-of-$1 and daily returns.

    Routes through simulate_rebalanced_portfolio for ALL frequencies;
    rebalance_freq=None is buy-and-hold (matches the closed-form get_portfolio_price).
    """
    asset_cols = [c for c in weights_dict if c in price_data.columns]
    asset_prices = price_data[asset_cols]
    sub_weights = {k: weights_dict[k] for k in asset_cols}

    wealth, weights_history = simulate_rebalanced_portfolio(asset_prices, sub_weights, rebalance_freq)
    wealth = wealth.rename("Portfolio")

    bench = price_data[benchmark_col]
    bench_growth = (bench / bench.iloc[0]).rename("Benchmark")

    growth = pd.concat([wealth, bench_growth], axis=1).dropna()
    daily = growth.pct_change().dropna()
    return ReturnsBundle(daily=daily, growth=growth, weights_history=weights_history)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/analytics.py test/test_analytics_core.py
git commit -m "feat: portfolio_returns + ReturnsBundle (single Growth-of-\$1 source, honors rebalance_freq)"
```

---

### Task 4: `compute_metrics` (numeric) + `format_metrics`

**Files:**
- Modify: `src/quant_reporter/analytics.py` (append)
- Test: `test/test_analytics_core.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
from quant_reporter.analytics import compute_metrics, format_metrics

REALIZED_KEYS = {
    "Realized CAGR", "Realized Volatility", "Realized Sharpe", "Realized Sortino",
    "Calmar", "Max Drawdown", "Beta (CAPM)", "Alpha (CAPM, ann.)",
    "Skew", "Kurtosis", "VaR (95%, daily)", "CVaR (95%, daily)",
}


def test_compute_metrics_numeric_and_consistent(synthetic_prices):
    w = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    rb = portfolio_returns(synthetic_prices, w, "BMK")
    m = compute_metrics(rb, risk_free_rate=0.02)
    assert set(m) == REALIZED_KEYS
    assert all(isinstance(v, float) for v in m.values())
    # internal consistency: realized vol == std(daily portfolio) * sqrt(252)
    assert m["Realized Volatility"] == pytest.approx(rb.daily["Portfolio"].std() * np.sqrt(252))
    # max drawdown comes from the SAME growth series
    from quant_reporter.metrics import compute_drawdown
    assert m["Max Drawdown"] == pytest.approx(compute_drawdown(rb.growth["Portfolio"]).max_dd)


def test_format_metrics_strings():
    m = {"Realized Volatility": 0.1234, "Realized Sharpe": 1.2}
    f = format_metrics(m)
    assert f["Realized Volatility"] == "12.34%"
    assert f["Realized Sharpe"] == "1.20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: FAIL with `cannot import name 'compute_metrics'`.

- [ ] **Step 3: Implement (append to analytics.py)**

```python
_PCT_KEYS = {
    "Realized CAGR", "Realized Volatility", "Alpha (CAPM, ann.)",
    "Max Drawdown", "VaR (95%, daily)", "CVaR (95%, daily)",
}


def compute_metrics(bundle, risk_free_rate):
    """The REALIZED metrics block (numeric) computed once from a ReturnsBundle."""
    pr = bundle.daily["Portfolio"]
    br = bundle.daily["Benchmark"]
    growth = bundle.growth["Portfolio"]
    ann = np.sqrt(252)

    vol = float(pr.std() * ann)
    excess = pr - risk_free_rate / 252
    sharpe = float((excess.mean() * 252) / (excess.std() * ann)) if excess.std() else 0.0
    sortino = float(calculate_sortino_ratio(pr, risk_free_rate))
    max_dd = compute_drawdown(growth).max_dd

    n_years = max((growth.index[-1] - growth.index[0]).days / 365.25, 1)
    cagr = float(growth.iloc[-1] ** (1 / n_years) - 1)
    calmar = float(cagr / abs(max_dd)) if max_dd else float("nan")

    if br.std() == 0 or pr.std() == 0:
        beta, alpha = 0.0, 0.0
    else:
        lr = stats.linregress(br, pr)
        beta, alpha = float(lr.slope), float(lr.intercept * 252)

    var95, cvar95 = calculate_var_cvar(pr, 0.95)

    return {
        "Realized CAGR": cagr,
        "Realized Volatility": vol,
        "Realized Sharpe": sharpe,
        "Realized Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Beta (CAPM)": beta,
        "Alpha (CAPM, ann.)": alpha,
        "Skew": float(pr.skew()),
        "Kurtosis": float(pr.kurtosis()),
        "VaR (95%, daily)": float(var95),
        "CVaR (95%, daily)": float(cvar95),
    }


def format_metrics(metrics):
    """Display formatter: % for rate-like keys, 2dp otherwise."""
    return {k: (f"{v:.2%}" if k in _PCT_KEYS else f"{v:.2f}") for k, v in metrics.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_analytics_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/analytics.py test/test_analytics_core.py
git commit -m "feat: compute_metrics (numeric realized block) + format_metrics"
```

---

### Task 5: `PortfolioAnalytics` accessor + `ctx.analytics` wiring

**Files:**
- Modify: `src/quant_reporter/analytics.py` (append `PortfolioAnalytics`)
- Modify: `src/quant_reporter/report_context.py` (add `analytics` field `report_context.py:40`-area; set it before `return` at `report_context.py:163-193`)
- Test: `test/test_ctx_analytics.py`

- [ ] **Step 1: Write the failing test**

`test/test_ctx_analytics.py`:
```python
import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic_prices
from quant_reporter.report_context import ReportContext
from quant_reporter.opt_core import get_optimization_inputs
from quant_reporter.analytics import PortfolioAnalytics


def _ctx_from_prices(prices, weights, benchmark="BMK", rfr=0.02, rebalance_freq=None):
    tickers = list(weights)
    mean_returns, cov_matrix, log_returns = get_optimization_inputs(prices[tickers])
    ctx = ReportContext(
        full_start="2021-01-01", full_end="2023-12-31",
        train_start="2021-01-01", train_end="2022-12-31",
        test_start="2023-01-01", test_end="2023-12-31",
        portfolio_dict=weights, benchmark_ticker=benchmark,
        display_names=None, sector_map=None, risk_free_rate=rfr,
        sector_caps=None, sector_mins=None,
        bl_views=None, bl_view_confidences=None,
        bl_relative_views=None, bl_relative_view_confidences=None,
        rebalance_freq=rebalance_freq,
        tickers=tickers, friendly_tickers=tickers, friendly_benchmark=benchmark,
        friendly_sector_map=None, user_friendly_weights=weights,
        price_data_full=prices, price_data_train=prices, price_data_test=prices,
        mean_returns=mean_returns, cov_matrix=cov_matrix, log_returns=log_returns,
    )
    ctx.analytics = PortfolioAnalytics(ctx)
    return ctx


def test_ctx_analytics_consistency_and_caching():
    prices = make_synthetic_prices()
    ctx = _ctx_from_prices(prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    # anti-divergence guard: scalar == curve.min()
    assert ctx.analytics.drawdown.max_dd == ctx.analytics.drawdown.curve.min()
    # metrics drawdown == accessor drawdown (same series, computed once)
    assert ctx.analytics.metrics["Max Drawdown"] == pytest.approx(ctx.analytics.drawdown.max_dd)
    # model_stats present and numeric
    assert set(ctx.analytics.model_stats) == {"Expected Return", "Expected Volatility", "Expected Sharpe"}
    # cached: same object on re-access
    assert ctx.analytics.returns is ctx.analytics.returns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_ctx_analytics.py -v`
Expected: FAIL with `cannot import name 'PortfolioAnalytics'`.

- [ ] **Step 3: Implement `PortfolioAnalytics` (append to analytics.py)**

```python
class PortfolioAnalytics:
    """Lazy, memoized accessor attached as ctx.analytics. Delegates to the pure
    functions and caches, so every report section reads identical, compute-once values."""

    def __init__(self, ctx):
        self._ctx = ctx

    @cached_property
    def returns(self) -> ReturnsBundle:
        c = self._ctx
        return portfolio_returns(
            c.price_data_full, c.user_friendly_weights, c.friendly_benchmark,
            getattr(c, "rebalance_freq", None),
        )

    @cached_property
    def drawdown(self):
        return compute_drawdown(self.returns.growth["Portfolio"])

    @cached_property
    def metrics(self):
        return compute_metrics(self.returns, self._ctx.risk_free_rate)

    @cached_property
    def model_stats(self):
        from .opt_core import get_portfolio_stats
        c = self._ctx
        w = pd.Series(c.user_friendly_weights).reindex(c.mean_returns.index).fillna(0.0).values
        ret, vol, sharpe = get_portfolio_stats(w, c.mean_returns, c.cov_matrix, c.risk_free_rate)
        return {"Expected Return": float(ret), "Expected Volatility": float(vol), "Expected Sharpe": float(sharpe)}
```

- [ ] **Step 4: Wire into report_context.py**

In `src/quant_reporter/report_context.py`, add the import after the existing imports (top of file):
```python
from .analytics import PortfolioAnalytics
```
Add a field to the `ReportContext` dataclass — append AFTER the existing `log_returns` field (the last field, ~`report_context.py:57`) so the defaulted field comes last:
```python
    # Memoized analytics accessor (set in build_context)
    analytics: object = None
```
In `build_context`, set it on the constructed instance — change the end of the function (currently `report_context.py:163`) from `return ReportContext(...)` to assign first:
```python
    ctx = ReportContext(
        # ... existing keyword args unchanged ...
        log_returns=log_returns,
    )
    ctx.analytics = PortfolioAnalytics(ctx)
    return ctx
```

- [ ] **Step 5: Run tests (new + import smoke)**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_ctx_analytics.py -v`
Expected: PASS.
Run: `PYTHONPATH=src .venv/bin/python -c "import quant_reporter as qr; print(qr.__version__)"`
Expected: prints `2.0.0` (no import cycle).

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/analytics.py src/quant_reporter/report_context.py test/test_ctx_analytics.py
git commit -m "feat: PortfolioAnalytics accessor wired as ctx.analytics (memoized, single-source)"
```

---

### Task 6: Unify the risk-free-rate fallback

**Files:**
- Modify: `src/quant_reporter/opt_core.py` (`get_risk_free_rate` except branch `opt_core.py:213-215`)
- Modify: `src/quant_reporter/report_context.py` (else-branch default `report_context.py:90`)
- Test: `test/test_rfr_fallback.py`

- [ ] **Step 1: Write the failing test**

`test/test_rfr_fallback.py`:
```python
import quant_reporter.opt_core as oc
from quant_reporter.opt_core import DEFAULT_RISK_FREE_RATE, get_risk_free_rate


def test_rfr_single_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(oc.yf, "download", boom)
    assert get_risk_free_rate() == DEFAULT_RISK_FREE_RATE
    assert DEFAULT_RISK_FREE_RATE == 0.02
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_rfr_fallback.py -v`
Expected: FAIL with `cannot import name 'DEFAULT_RISK_FREE_RATE'`.

- [ ] **Step 3: Implement**

In `src/quant_reporter/opt_core.py`, add a module constant after the imports (near `opt_core.py:7`):
```python
DEFAULT_RISK_FREE_RATE = 0.02
```
Change the `except` branch of `get_risk_free_rate` (`opt_core.py:213-215`) — the fallback `return 0.06` becomes:
```python
    except Exception as e:
        logger.warning("Could not fetch live risk-free rate. Defaulting to %.2f. Error: %s",
                       DEFAULT_RISK_FREE_RATE, e)
        return DEFAULT_RISK_FREE_RATE
```
In `src/quant_reporter/report_context.py`, update the existing import line (`report_context.py:8`) to also pull the constant:
```python
from .opt_core import get_risk_free_rate, get_optimization_inputs, DEFAULT_RISK_FREE_RATE
```
Then change the else-branch (`report_context.py:90`, currently `rfr = 0.02`) to:
```python
    else:
        rfr = DEFAULT_RISK_FREE_RATE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest -o addopts="" test/test_rfr_fallback.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/opt_core.py src/quant_reporter/report_context.py test/test_rfr_fallback.py
git commit -m "fix: single risk-free-rate fallback (0.06 -> 0.02) via DEFAULT_RISK_FREE_RATE"
```

---

### Task 7: Export new symbols, CHANGELOG, full-suite green

**Files:**
- Modify: `src/quant_reporter/__init__.py` (add an analytics export block near `__init__.py:19`)
- Modify: `CHANGELOG.md` (add `[2.1.0] - Unreleased`)
- Test: full suite

- [ ] **Step 1: Add exports**

In `src/quant_reporter/__init__.py`, after the `report_context` import (`__init__.py:19`) add:
```python
# --- Analytics Core (single source of truth) ---
from .analytics import (
    portfolio_returns,
    ReturnsBundle,
    compute_metrics,
    format_metrics,
    PortfolioAnalytics,
)
from .metrics import compute_drawdown, DrawdownResult
```

- [ ] **Step 2: Add CHANGELOG entry**

Prepend to `CHANGELOG.md` (under the top heading):
```markdown
## [2.1.0] - Unreleased

### Added
- Analytics core (`analytics.py`): `portfolio_returns`/`ReturnsBundle`, `compute_metrics` (numeric)
  + `format_metrics`, `compute_drawdown`/`DrawdownResult`, and the memoized `ctx.analytics` accessor —
  the single source of truth for portfolio returns, growth, drawdown, and realized metrics.

### Changed
- Risk-free-rate fetch failure now falls back to **0.02** (was 0.06) via `DEFAULT_RISK_FREE_RATE`,
  matching `build_context`'s default — one documented fallback.

### Notes
- Report modules still use the legacy `calculate_metrics`; migration to the new core (and its clean
  break to numeric output) lands in SP0 Part B.
```

- [ ] **Step 3: Run the FULL suite + import check**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: all pass — the original **37** plus the new core tests (`37 + new`), 0 failures (smoke test skips if offline).
Run: `PYTHONPATH=src .venv/bin/python -c "import quant_reporter as qr; print(qr.portfolio_returns, qr.compute_drawdown, qr.__version__)"`
Expected: prints the two functions and `2.0.0`.

- [ ] **Step 4: Commit**

```bash
git add src/quant_reporter/__init__.py CHANGELOG.md
git commit -m "feat: export analytics core; CHANGELOG 2.1.0 (rfr fallback unified)"
```

---

## Self-Review (completed by plan author)

**Spec coverage (Part A scope):** ✅ analytics core (`portfolio_returns`, `compute_drawdown`, `compute_metrics`/`format_metrics`, `PortfolioAnalytics`) — Tasks 2-5; ✅ canonical basis blocks (Expected via `model_stats`, Realized via `compute_metrics`, labeled keys) — Tasks 4-5; ✅ `rebalance_freq` honored — Task 3; ✅ offline fixture + golden + consistency guards — Tasks 1-5; ✅ rfr single fallback — Task 6. **Deferred to Part B (documented):** report-module refactor, Monte-Carlo merge+seed, factor/attribution correctness, combined fail-loud, and the `calculate_metrics` clean break.

**Placeholder scan:** none. (An earlier `if False` construct in Task 6 was replaced with a direct `rfr = DEFAULT_RISK_FREE_RATE` edit.) No TBD/TODO remain.

**Type consistency:** `ReturnsBundle.daily/growth` columns `['Portfolio','Benchmark']` used identically in Tasks 3-5; `DrawdownResult.curve/max_dd` consistent (Tasks 2,4,5); `compute_metrics` keys match `REALIZED_KEYS`/`_PCT_KEYS`/`format_metrics` (Task 4); `PortfolioAnalytics.{returns,drawdown,metrics,model_stats}` match the test (Task 5).
