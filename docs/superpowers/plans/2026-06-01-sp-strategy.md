# SP-Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a function-first strategy abstraction, a thin `backtest()` runner returning a rich `BacktestResult` with an interactive HTML report, a consolidated metrics library, an objectives/loss library, and a set of prebuilt strategies.

**Architecture:** A strategy is any callable `(prices, **params) -> weights` (dict or dated DataFrame), or a thin `Strategy` wrapper, or raw weights. `backtest()` resolves it to weights, runs the existing SP1 `simulate_strategy` (unchanged), aligns a benchmark, and wraps the output in `BacktestResult`. The result lazily computes metrics (from the new `metrics.py` surface) and OOS stats (PSR/DSR), and renders an interactive report via the existing `html_builder` + plotly. All changes are additive; the 233 existing tests must stay green.

**Tech Stack:** numpy, pandas, scipy, sklearn, plotly (all already pinned). No new deps.

**Verification convention:** Always run the suite as
`PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" <args>` and trust the exit code.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/quant_reporter/metrics.py` (modify) | Add the measurement surface (pure `(returns)->float`) + `summary_metrics`. Existing functions untouched. |
| `src/quant_reporter/objectives.py` (create) | Minimize-ready objective/loss functions. |
| `src/quant_reporter/strategies.py` (create) | Prebuilt strategy functions + `REGISTRY`. |
| `src/quant_reporter/strategy.py` (create) | `Strategy`, `backtest`, `backtest_many`, `BacktestResult`. |
| `src/quant_reporter/backtest_report.py` (create) | Plotly panels + `build_sections` + `create_backtest_report`. |
| `src/quant_reporter/__init__.py` (modify) | Wire all new exports (LAST, sequential). |
| `examples/example_strategy_report.py` (create) | End-to-end demo writing an HTML report. |
| `test/test_metrics_lib.py`, `test/test_objectives.py`, `test/test_strategies.py`, `test/test_strategy.py`, `test/test_backtest_report.py` (create) | One test file per module. |

**Build order (dependency spine):** Task 1 (metrics) and Task 2 (objectives) are leaves. Task 3 (strategies) uses existing primitives. Task 4 (strategy runner) uses `simulate_strategy` + metrics. Task 5 (report) uses the result + metrics + plotting. Task 6 wires `__init__` and adds the example. Tasks 1–3 touch only brand-new files (except metrics.py — see Task 1) and could parallelize; **`__init__.py` is edited only in Task 6** to avoid the cross-contamination gotcha.

---

## Task 1: Metrics library (measurement surface)

**Files:**
- Modify: `src/quant_reporter/metrics.py` (append; do not alter existing functions)
- Test: `test/test_metrics_lib.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_metrics_lib.py`:

```python
# test/test_metrics_lib.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.metrics import (
    cagr, annual_volatility, sharpe, sortino, calmar, omega,
    max_drawdown, avg_drawdown, ulcer_index, value_at_risk, conditional_var,
    downside_deviation, tracking_error, information_ratio, hit_rate,
    win_loss_ratio, tail_ratio, skewness, kurtosis, summary_metrics,
)
from conftest import make_synthetic_prices


def _rets(seed=1, n=504):
    return make_synthetic_prices(seed=seed, n_days=n)["AAA"].pct_change().dropna()


def test_annual_vol_matches_formula():
    r = _rets()
    assert annual_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_sharpe_zero_rfr_matches_formula():
    r = _rets()
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe(r, risk_free_rate=0.0) == pytest.approx(expected, rel=1e-9)


def test_cagr_constant_growth():
    # 1% per period for 252 periods -> (1.01**252)-1 annualized over exactly 1 year
    r = pd.Series([0.01] * 252)
    assert cagr(r) == pytest.approx((1.01 ** 252) - 1, rel=1e-9)


def test_max_drawdown_non_positive():
    assert max_drawdown(_rets()) <= 0.0


def test_max_drawdown_known_path():
    # +100% then -50% -> peak 2.0, trough 1.0 -> dd = -0.5
    r = pd.Series([1.0, -0.5])
    assert max_drawdown(r) == pytest.approx(-0.5, rel=1e-9)


def test_calmar_sign():
    assert calmar(_rets()) == pytest.approx(cagr(_rets()) / abs(max_drawdown(_rets())), rel=1e-9)


def test_value_at_risk_positive_loss():
    assert value_at_risk(_rets(), 0.95) > 0


def test_cvar_ge_var():
    r = _rets()
    assert conditional_var(r, 0.95) >= value_at_risk(r, 0.95)


def test_hit_rate_bounds():
    assert 0.0 <= hit_rate(_rets()) <= 1.0


def test_omega_above_one_for_positive_drift():
    r = pd.Series([0.02, -0.01, 0.02, -0.01, 0.03])
    assert omega(r, 0.0) > 1.0


def test_tracking_error_zero_for_identical():
    r = _rets()
    assert tracking_error(r, r) == pytest.approx(0.0, abs=1e-12)


def test_information_ratio_finite():
    r = _rets(seed=1)
    b = _rets(seed=2)
    assert np.isfinite(information_ratio(r, b))


def test_downside_deviation_le_total_vol():
    r = _rets()
    assert downside_deviation(r) <= annual_volatility(r) + 1e-9


def test_empty_returns_nan_not_raise():
    empty = pd.Series([], dtype=float)
    for fn in (cagr, annual_volatility, sharpe, sortino, max_drawdown,
               value_at_risk, hit_rate, ulcer_index):
        assert np.isnan(fn(empty)) or fn(empty) == 0.0


def test_summary_metrics_keys_no_benchmark():
    m = summary_metrics(_rets())
    for k in ("CAGR", "Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown",
              "Avg Drawdown", "Ulcer Index", "VaR (95%)", "CVaR (95%)", "Downside Dev",
              "Omega", "Hit Rate", "Win/Loss", "Tail Ratio", "Skew", "Kurtosis"):
        assert k in m
    assert "Tracking Error" not in m


def test_summary_metrics_adds_benchmark_keys():
    m = summary_metrics(_rets(seed=1), benchmark=_rets(seed=2))
    assert "Tracking Error" in m and "Information Ratio" in m
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_metrics_lib.py -q`
Expected: FAIL with `ImportError: cannot import name 'cagr'`.

- [ ] **Step 3: Append the implementation to `metrics.py`**

Append to `src/quant_reporter/metrics.py` (leave existing code above it intact):

```python
# ---------------------------------------------------------------------------
# SP-Strategy: consolidated measurement surface.
# All functions take SIMPLE periodic returns and return plain floats.
# Edge cases (empty / single-obs / zero-vol) return NaN (or 0.0), never raise.
# ---------------------------------------------------------------------------
TRADING_DAYS = 252


def _series(returns):
    return pd.Series(returns).dropna()


def _underwater(returns):
    r = _series(returns)
    if len(r) == 0:
        return pd.Series([], dtype=float)
    growth = (1.0 + r).cumprod()
    peak = growth.cummax()
    return (growth - peak) / peak


def cagr(returns, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) == 0:
        return float("nan")
    growth = float((1.0 + r).prod())
    if growth <= 0:
        return float("nan")
    years = len(r) / periods_per_year
    return float(growth ** (1.0 / years) - 1.0)


def annual_volatility(returns, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def downside_deviation(returns, mar=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    downside = r[r < mar] - mar
    if len(downside) < 1:
        return float("nan")
    return float(np.sqrt((downside ** 2).mean()) * np.sqrt(periods_per_year))


def sortino(returns, risk_free_rate=0.0, periods_per_year=TRADING_DAYS):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    dd = downside_deviation(r, mar=risk_free_rate / periods_per_year,
                            periods_per_year=periods_per_year)
    if dd == 0 or not np.isfinite(dd):
        return float("nan")
    excess_ann = (r.mean() - risk_free_rate / periods_per_year) * periods_per_year
    return float(excess_ann / dd)


def max_drawdown(returns):
    uw = _underwater(returns)
    return float(uw.min()) if len(uw) else float("nan")


def avg_drawdown(returns):
    uw = _underwater(returns)
    dd = uw[uw < 0]
    return float(dd.mean()) if len(dd) else 0.0


def ulcer_index(returns):
    uw = _underwater(returns)
    return float(np.sqrt((uw ** 2).mean())) if len(uw) else float("nan")


def calmar(returns, periods_per_year=TRADING_DAYS):
    mdd = max_drawdown(returns)
    if mdd == 0 or not np.isfinite(mdd):
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def value_at_risk(returns, confidence=0.95):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    return float(-r.quantile(1.0 - confidence))


def conditional_var(returns, confidence=0.95):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    threshold = r.quantile(1.0 - confidence)
    tail = r[r <= threshold]
    return float(-tail.mean()) if len(tail) else float("nan")


def omega(returns, threshold=0.0):
    r = _series(returns)
    excess = r - threshold
    gains = float(excess[excess > 0].sum())
    losses = float(-excess[excess < 0].sum())
    if losses == 0:
        return float("nan")
    return float(gains / losses)


def hit_rate(returns):
    r = _series(returns)
    return float((r > 0).mean()) if len(r) else float("nan")


def win_loss_ratio(returns):
    r = _series(returns)
    wins, losses = r[r > 0], r[r < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def tail_ratio(returns):
    r = _series(returns)
    if len(r) < 2:
        return float("nan")
    left = abs(r.quantile(0.05))
    if left == 0:
        return float("nan")
    return float(abs(r.quantile(0.95)) / left)


def tracking_error(returns, benchmark, periods_per_year=TRADING_DAYS):
    active = (pd.Series(returns) - pd.Series(benchmark)).dropna()
    if len(active) < 2:
        return float("nan")
    return float(active.std(ddof=1) * np.sqrt(periods_per_year))


def information_ratio(returns, benchmark, periods_per_year=TRADING_DAYS):
    active = (pd.Series(returns) - pd.Series(benchmark)).dropna()
    if len(active) < 2:
        return float("nan")
    sd = active.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return float("nan")
    return float(active.mean() / sd * np.sqrt(periods_per_year))


def skewness(returns):
    r = _series(returns)
    return float(r.skew()) if len(r) >= 3 else float("nan")


def kurtosis(returns):
    r = _series(returns)
    return float(r.kurtosis()) if len(r) >= 4 else float("nan")


def summary_metrics(returns, benchmark=None, risk_free_rate=0.02,
                    periods_per_year=TRADING_DAYS):
    """Ordered, named metric dict the backtest report consumes."""
    r = _series(returns)
    out = {
        "CAGR": cagr(r, periods_per_year),
        "Volatility": annual_volatility(r, periods_per_year),
        "Sharpe": sharpe(r, risk_free_rate, periods_per_year),
        "Sortino": sortino(r, risk_free_rate, periods_per_year),
        "Calmar": calmar(r, periods_per_year),
        "Max Drawdown": max_drawdown(r),
        "Avg Drawdown": avg_drawdown(r),
        "Ulcer Index": ulcer_index(r),
        "VaR (95%)": value_at_risk(r, 0.95),
        "CVaR (95%)": conditional_var(r, 0.95),
        "Downside Dev": downside_deviation(r, 0.0, periods_per_year),
        "Omega": omega(r, 0.0),
        "Hit Rate": hit_rate(r),
        "Win/Loss": win_loss_ratio(r),
        "Tail Ratio": tail_ratio(r),
        "Skew": skewness(r),
        "Kurtosis": kurtosis(r),
    }
    if benchmark is not None:
        out["Tracking Error"] = tracking_error(r, benchmark, periods_per_year)
        out["Information Ratio"] = information_ratio(r, benchmark, periods_per_year)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_metrics_lib.py -q`
Expected: PASS (all).

- [ ] **Step 5: Verify the full suite is still green**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: PASS, count = previous + new (234+).

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/metrics.py test/test_metrics_lib.py
git commit -m "feat(metrics): consolidated measurement surface + summary_metrics"
```

---

## Task 2: Objectives library (loss/minimize surface)

**Files:**
- Create: `src/quant_reporter/objectives.py`
- Test: `test/test_objectives.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_objectives.py`:

```python
# test/test_objectives.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.objectives import (
    neg_sharpe, variance, cvar_objective, tracking_error_objective,
    mean_squared_error, mean_absolute_error,
)


def test_neg_sharpe_matches_manual():
    w = np.array([0.5, 0.5])
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    port_ret = 0.5 * 0.10 + 0.5 * 0.20
    port_vol = np.sqrt(0.5 ** 2 * 0.04 + 0.5 ** 2 * 0.09)
    assert neg_sharpe(w, mu, cov, 0.02) == pytest.approx(-(port_ret - 0.02) / port_vol, rel=1e-9)


def test_variance_matches_quadratic_form():
    w = np.array([0.5, 0.5])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    assert variance(w, None, cov) == pytest.approx(float(w @ cov @ w), rel=1e-12)


def test_cvar_objective_positive_for_losses():
    rng = np.random.default_rng(0)
    R = rng.normal(-0.001, 0.01, (500, 2))
    assert cvar_objective(np.array([0.5, 0.5]), R, 0.95) > 0


def test_tracking_error_objective_zero_when_matched():
    R = np.array([[0.01, 0.01], [0.02, 0.02], [-0.01, -0.01]])
    bench = np.array([0.01, 0.02, -0.01])
    # weight fully on a column equal to benchmark -> zero TE
    assert tracking_error_objective(np.array([1.0, 0.0]), R, bench) == pytest.approx(0.0, abs=1e-12)


def test_mse_and_mae_zero_when_equal():
    y = np.array([1.0, 2.0, 3.0])
    assert mean_squared_error(y, y) == 0.0
    assert mean_absolute_error(y, y) == 0.0


def test_mse_known_value():
    assert mean_squared_error([0.0, 0.0], [1.0, 3.0]) == pytest.approx((1 + 9) / 2)


def test_neg_sharpe_drop_in_for_find_optimal_portfolio():
    # Must accept the (w, mean, cov, rfr) arg order find_optimal_portfolio uses.
    import scipy.optimize as sco
    mu = np.array([0.10, 0.20]); cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    res = sco.minimize(neg_sharpe, np.array([0.5, 0.5]), args=(mu, cov, 0.02),
                       method="SLSQP", bounds=[(0, 1)] * 2,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    assert res.success and 0 <= res.x[0] <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_objectives.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.objectives'`.

- [ ] **Step 3: Create `objectives.py`**

```python
# src/quant_reporter/objectives.py
"""Optimization objective / loss functions (SP-Strategy).

Each is minimize-ready. The portfolio objectives use the same
(weights, mean_returns, cov_matrix, risk_free_rate) arg order as
opt_core.find_optimal_portfolio, so they are drop-in replacements.
"""
import numpy as np


def _cov(cov_matrix):
    return cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix, dtype=float)


def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    cov = _cov(cov_matrix)
    port_return = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-30)))
    if port_vol <= 0:
        return 0.0
    return float(-(port_return - risk_free_rate) / port_vol)


def variance(weights, mean_returns=None, cov_matrix=None, risk_free_rate=0.02):
    w = np.asarray(weights, dtype=float)
    cov = _cov(cov_matrix)
    return float(w @ cov @ w)


def cvar_objective(weights, returns_matrix, confidence=0.95):
    w = np.asarray(weights, dtype=float)
    R = returns_matrix.values if hasattr(returns_matrix, "values") else np.asarray(returns_matrix, dtype=float)
    port = R @ w
    threshold = np.quantile(port, 1.0 - confidence)
    tail = port[port <= threshold]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


def tracking_error_objective(weights, returns_matrix, benchmark_returns):
    w = np.asarray(weights, dtype=float)
    R = returns_matrix.values if hasattr(returns_matrix, "values") else np.asarray(returns_matrix, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    active = R @ w - b
    if active.size < 2:
        return 0.0
    return float(np.std(active, ddof=1))


def mean_squared_error(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def mean_absolute_error(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_objectives.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/objectives.py test/test_objectives.py
git commit -m "feat(objectives): minimize-ready loss/objective surface"
```

---

## Task 3: Prebuilt strategies + REGISTRY

**Files:**
- Create: `src/quant_reporter/strategies.py`
- Test: `test/test_strategies.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_strategies.py`:

```python
# test/test_strategies.py
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from quant_reporter.strategies import (
    equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe,
    trend_following, cross_sectional_momentum, vol_target_overlay, REGISTRY,
)
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)[["AAA", "BBB", "CCC"]]


@pytest.mark.parametrize("fn", [equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe])
def test_static_strategy_weights_sum_to_one(fn):
    w = fn(_prices())
    assert isinstance(w, dict)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(w) == {"AAA", "BBB", "CCC"}
    assert all(v >= -1e-9 for v in w.values())


def test_registry_contains_expected():
    for name in ("equal_weight", "inverse_vol", "min_variance", "risk_parity",
                 "max_sharpe", "trend_following", "cross_sectional_momentum"):
        assert name in REGISTRY and callable(REGISTRY[name])


def test_trend_following_returns_valid_schedule():
    sched = trend_following(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()        # long-biased every row
    assert list(sched.columns) == ["AAA", "BBB", "CCC"]


def test_cross_sectional_momentum_returns_valid_schedule():
    sched = cross_sectional_momentum(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()


def test_vol_target_overlay_wraps_base():
    overlay = vol_target_overlay(equal_weight, target_vol=0.10)
    sched = overlay(_prices(n=800))
    assert isinstance(sched, pd.DataFrame)
    assert (sched.sum(axis=1) > 0).all()


@settings(max_examples=15, deadline=None)
@given(cut=st.integers(min_value=300, max_value=500))
def test_trend_following_is_causal(cut):
    prices = _prices(n=600, seed=5)
    base = trend_following(prices)
    shuffled = prices.copy()
    rng = np.random.default_rng(7)
    tail = np.arange(cut + 1, len(prices))
    shuffled.iloc[cut + 1:] = prices.iloc[rng.permutation(tail)].values
    shuf = trend_following(shuffled)
    common = base.index.intersection(shuf.index)
    common = [d for d in common if prices.index.get_loc(d) <= cut]
    pd.testing.assert_frame_equal(base.loc[common], shuf.loc[common])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_strategies.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.strategies'`.

- [ ] **Step 3: Create `strategies.py`**

```python
# src/quant_reporter/strategies.py
"""Prebuilt strategies (SP-Strategy).

Each is a function (prices, **params) -> weights, where weights is a dict
(static) or a DataFrame (dated, look-ahead-safe schedule). REGISTRY lists the
no-argument-factory strategies for discoverability.
"""
import numpy as np
import pandas as pd


def _long_only_schedule(sized, prices):
    """Clip to long-only, renormalize per row; equal-weight fallback on flat rows."""
    cols = list(prices.columns)
    n = len(cols)
    long_only = sized.reindex(columns=cols).clip(lower=0)
    row_sum = long_only.sum(axis=1)
    normalized = long_only.div(row_sum.where(row_sum > 0, other=1.0), axis=0)
    eq = pd.DataFrame(1.0 / n, index=normalized.index, columns=cols)
    mask = pd.DataFrame(np.repeat((row_sum > 0).values[:, None], n, axis=1),
                        index=normalized.index, columns=cols)
    return normalized.where(mask, other=eq).dropna(how="any")


def equal_weight(prices, **kwargs):
    cols = list(prices.columns)
    return {c: 1.0 / len(cols) for c in cols}


def inverse_vol(prices, lookback=63, method="ewma", **kwargs):
    from .sizing import inverse_volatility_weights
    returns = prices.pct_change().dropna()
    return inverse_volatility_weights(returns, lookback=lookback, method=method)


def min_variance(prices, **kwargs):
    from .robust_estimators import ledoit_wolf_covariance
    from .opt_core import find_optimal_portfolio, objective_min_variance, build_constraints
    returns = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(returns)["cov_matrix"]
    mean = returns.mean() * 252
    n = len(cov.columns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = build_constraints(n, list(cov.columns))
    w = find_optimal_portfolio(objective_min_variance, mean, cov, bounds, constraints)
    return dict(zip(cov.columns, np.asarray(w, dtype=float)))


def risk_parity(prices, **kwargs):
    from .robust_estimators import ledoit_wolf_covariance
    from .opt_core import optimize_risk_budget
    returns = prices.pct_change().dropna()
    cov = ledoit_wolf_covariance(returns)["cov_matrix"]
    return optimize_risk_budget(cov)["weights"]


def max_sharpe(prices, risk_free_rate=0.02, **kwargs):
    from .opt_core import (get_optimization_inputs, find_optimal_portfolio,
                           objective_neg_sharpe, build_constraints)
    mean, cov, _ = get_optimization_inputs(prices)
    n = len(cov.columns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = build_constraints(n, list(cov.columns))
    w = find_optimal_portfolio(objective_neg_sharpe, mean, cov, bounds, constraints, risk_free_rate)
    return dict(zip(cov.columns, np.asarray(w, dtype=float)))


def trend_following(prices, lookback=252, skip_recent=21, fast=50, slow=200,
                    target_vol=0.10, vol_lookback=63, max_leverage=2.0, **kwargs):
    from .signals import (time_series_momentum_signal, moving_average_crossover_signal,
                          volatility_target_positions)
    tsm = time_series_momentum_signal(prices, lookback=lookback, skip_recent=skip_recent)
    mac = moving_average_crossover_signal(prices, fast=fast, slow=slow)
    ensemble = ((tsm + mac) / 2.0).dropna(how="any")
    sized = volatility_target_positions(ensemble, prices.pct_change(), target_vol=target_vol,
                                        vol_lookback=vol_lookback, method="ewma",
                                        max_leverage=max_leverage, scaling="per_asset")
    return _long_only_schedule(sized, prices)


def cross_sectional_momentum(prices, lookback=126, skip_recent=5, target_vol=0.10,
                             vol_lookback=63, max_leverage=2.0, **kwargs):
    from .signals import cross_sectional_momentum_score, volatility_target_positions
    score = cross_sectional_momentum_score(prices, lookback=lookback, skip_recent=skip_recent)
    longs = (score > 0).astype(float)
    sized = volatility_target_positions(longs, prices.pct_change(), target_vol=target_vol,
                                        vol_lookback=vol_lookback, method="ewma",
                                        max_leverage=max_leverage, scaling="per_asset")
    return _long_only_schedule(sized, prices)


def vol_target_overlay(base_fn, target_vol=0.10, vol_lookback=63, max_leverage=2.0):
    """Higher-order: return a strategy fn that vol-targets base_fn's weights."""
    def _strategy(prices, **kwargs):
        from .signals import volatility_target_positions
        base = base_fn(prices, **kwargs)
        if isinstance(base, dict):
            sig = pd.DataFrame([base] * len(prices), index=prices.index)
        else:
            sig = base
        sig = sig.reindex(columns=prices.columns).fillna(0.0)
        sized = volatility_target_positions(sig, prices.pct_change(), target_vol=target_vol,
                                            vol_lookback=vol_lookback, method="ewma",
                                            max_leverage=max_leverage, scaling="portfolio")
        return _long_only_schedule(sized, prices)
    return _strategy


REGISTRY = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "min_variance": min_variance,
    "risk_parity": risk_parity,
    "max_sharpe": max_sharpe,
    "trend_following": trend_following,
    "cross_sectional_momentum": cross_sectional_momentum,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_strategies.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/strategies.py test/test_strategies.py
git commit -m "feat(strategies): prebuilt strategies + REGISTRY"
```

---

## Task 4: Strategy runner + BacktestResult

**Files:**
- Create: `src/quant_reporter/strategy.py`
- Test: `test/test_strategy.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_strategy.py`:

```python
# test/test_strategy.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.strategy import Strategy, backtest, backtest_many, BacktestResult
from quant_reporter.backtest import simulate_strategy
from quant_reporter.strategies import equal_weight, risk_parity
from conftest import make_synthetic_prices


def _prices(n=600, seed=3):
    return make_synthetic_prices(seed=seed, n_days=n)  # includes BMK benchmark column


def test_backtest_accepts_callable():
    res = backtest(equal_weight, _prices()[["AAA", "BBB", "CCC"]], rebalance="M")
    assert isinstance(res, BacktestResult)
    assert res.wealth.iloc[-1] > 0


def test_backtest_accepts_strategy_wrapper():
    strat = Strategy("EW", equal_weight)
    res = backtest(strat, _prices()[["AAA", "BBB", "CCC"]])
    assert res.name == "EW"


def test_backtest_accepts_static_dict():
    res = backtest({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, _prices()[["AAA", "BBB", "CCC"]])
    assert isinstance(res, BacktestResult)


def test_frictionless_backtest_matches_simulate_strategy():
    prices = _prices()[["AAA", "BBB", "CCC"]]
    w = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    res = backtest(w, prices, rebalance="M", cost_model=None)
    sim = simulate_strategy(prices, w, cost_model=None, rebalance="M")
    pd.testing.assert_series_equal(res.wealth, sim["wealth"])


def test_benchmark_column_is_split_off():
    prices = _prices()  # AAA/BBB/CCC + BMK
    res = backtest(equal_weight, prices, benchmark="BMK")
    assert res.benchmark is not None
    # Strategy must NOT have allocated to BMK
    assert "BMK" not in res.weights.columns


def test_metrics_cached_and_have_keys():
    res = backtest(risk_parity, _prices()[["AAA", "BBB", "CCC"]])
    m = res.metrics
    assert res.metrics is m  # cached identity
    assert "Sharpe" in m and "Max Drawdown" in m


def test_oos_stats_present():
    res = backtest(equal_weight, _prices()[["AAA", "BBB", "CCC"]])
    assert set(res.oos_stats) == {"psr", "dsr"}


def test_backtest_many_returns_dict_of_results():
    prices = _prices()[["AAA", "BBB", "CCC"]]
    out = backtest_many({"EW": equal_weight, "RP": risk_parity}, prices)
    assert set(out) == {"EW", "RP"}
    assert all(isinstance(r, BacktestResult) for r in out.values())
    assert out["EW"].n_trials == 2  # used for DSR deflation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_strategy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.strategy'`.

- [ ] **Step 3: Create `strategy.py`**

```python
# src/quant_reporter/strategy.py
"""Strategy abstraction + backtest runner + result (SP-Strategy).

A strategy is any callable (prices, **params) -> weights (dict or DataFrame),
a Strategy wrapper, or raw weights. backtest() is a thin orchestration over the
SP1 simulate_strategy engine (which it does NOT replace).
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .backtest import simulate_strategy
from .metrics import summary_metrics
from .performance_stats import probabilistic_sharpe_ratio, deflated_sharpe_ratio


@dataclass
class Strategy:
    name: str
    fn: Callable
    params: dict = field(default_factory=dict)

    def weights(self, prices):
        return self.fn(prices, **self.params)


def _resolve_weights(strategy, prices):
    if isinstance(strategy, Strategy):
        return strategy.weights(prices), strategy.name
    if isinstance(strategy, (dict, pd.DataFrame)):
        return strategy, "custom"
    if callable(strategy):
        return strategy(prices), getattr(strategy, "__name__", "strategy")
    raise TypeError(f"Unsupported strategy type: {type(strategy)!r}")


@dataclass
class BacktestResult:
    name: str
    wealth: pd.Series
    weights: pd.DataFrame
    blotter: pd.DataFrame
    turnover: pd.Series
    cost_drag: float
    benchmark: Optional[pd.Series]
    summary: dict
    n_trials: int = 1

    @cached_property
    def returns(self) -> pd.Series:
        return self.wealth.pct_change().dropna()

    @cached_property
    def _benchmark_returns(self):
        if self.benchmark is None:
            return None
        return self.benchmark.pct_change().dropna()

    @cached_property
    def metrics(self) -> dict:
        return summary_metrics(self.returns, benchmark=self._benchmark_returns)

    @cached_property
    def oos_stats(self) -> dict:
        r = self.returns
        return {
            "psr": probabilistic_sharpe_ratio(r),
            "dsr": deflated_sharpe_ratio(r, n_trials=self.n_trials),
        }

    def report(self, path="backtest_report.html", open_browser=False):
        from .backtest_report import create_backtest_report
        return create_backtest_report(self, path=path, open_browser=open_browser)

    def plot_wealth(self):
        from .backtest_report import plot_wealth
        return plot_wealth(self)

    def plot_drawdown(self):
        from .backtest_report import plot_drawdown
        return plot_drawdown(self)

    def plot_weights(self):
        from .backtest_report import plot_weights
        return plot_weights(self)

    def plot_turnover(self):
        from .backtest_report import plot_turnover
        return plot_turnover(self)

    def plot_rolling(self):
        from .backtest_report import plot_rolling
        return plot_rolling(self)


def backtest(strategy, prices, *, rebalance="M", cost_model=None, benchmark=None,
             initial_value=1.0, name=None, n_trials=1) -> BacktestResult:
    bench_series = None
    asset_prices = prices
    if benchmark is not None:
        if isinstance(benchmark, str):
            b = prices[benchmark]
            bench_series = (b / b.iloc[0]).rename("Benchmark")
            asset_prices = prices.drop(columns=[benchmark])
        else:
            b = pd.Series(benchmark)
            bench_series = (b / b.iloc[0]).rename("Benchmark")

    weights, resolved_name = _resolve_weights(strategy, asset_prices)
    sim = simulate_strategy(asset_prices, weights, cost_model=cost_model,
                            rebalance=rebalance, initial_value=initial_value)

    if bench_series is not None:
        bench_series = bench_series.reindex(sim["wealth"].index).ffill()

    return BacktestResult(
        name=name or resolved_name,
        wealth=sim["wealth"],
        weights=sim["weights"],
        blotter=pd.DataFrame(sim["blotter"]),
        turnover=sim["turnover"],
        cost_drag=sim["cost_drag"],
        benchmark=bench_series,
        summary=sim["summary"],
        n_trials=n_trials,
    )


def backtest_many(strategies, prices, **kwargs) -> dict:
    kwargs.pop("name", None)
    kwargs.pop("n_trials", None)
    n = len(strategies)
    return {nm: backtest(strat, prices, name=nm, n_trials=n, **kwargs)
            for nm, strat in strategies.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_strategy.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/strategy.py test/test_strategy.py
git commit -m "feat(strategy): Strategy wrapper, backtest runner, BacktestResult"
```

---

## Task 5: Interactive backtest report

**Files:**
- Create: `src/quant_reporter/backtest_report.py`
- Test: `test/test_backtest_report.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_backtest_report.py`:

```python
# test/test_backtest_report.py
import os
import pandas as pd
import plotly.graph_objects as go
import pytest

from quant_reporter.strategy import backtest, backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from quant_reporter.backtest_report import (
    plot_wealth, plot_drawdown, plot_weights, plot_turnover, plot_rolling,
    build_sections, create_backtest_report,
)
from conftest import make_synthetic_prices


def _res():
    prices = make_synthetic_prices(n_days=600)
    return backtest(equal_weight, prices, benchmark="BMK", rebalance="M")


def test_plot_functions_return_figures_with_data():
    res = _res()
    for fn in (plot_wealth, plot_drawdown, plot_weights, plot_turnover, plot_rolling):
        fig = fn(res)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_plot_wealth_includes_benchmark_trace():
    res = _res()
    names = [t.name for t in plot_wealth(res).data]
    assert any("Benchmark" in (n or "") for n in names)


def test_build_sections_has_all_panels():
    res = _res()
    sections = build_sections(res)
    titles = [item.get("title") for s in sections for item in s["main_content"]]
    for expected in ("Growth of $1", "Underwater Drawdown", "Weights Over Time",
                     "Turnover per Rebalance", "Rolling Sharpe & Vol",
                     "Performance Metrics", "Trade Blotter"):
        assert expected in titles


def test_create_report_writes_file(tmp_path):
    res = _res()
    path = str(tmp_path / "bt.html")
    out = create_backtest_report(res, path=path)
    assert out == path
    assert os.path.exists(path)
    html = open(path, encoding="utf-8").read()
    assert "Growth of $1" in html and len(html) > 1000


def test_multi_strategy_report_has_comparison(tmp_path):
    prices = make_synthetic_prices(n_days=600)
    results = backtest_many({"EW": equal_weight, "RP": risk_parity}, prices, benchmark="BMK")
    path = str(tmp_path / "cmp.html")
    create_backtest_report(results, path=path)
    html = open(path, encoding="utf-8").read()
    assert "OOS Strategy Comparison" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest_report.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'quant_reporter.backtest_report'`.

- [ ] **Step 3: Create `backtest_report.py`**

```python
# src/quant_reporter/backtest_report.py
"""Interactive backtest report (SP-Strategy).

Reuses the established plotly_white style and html_builder.generate_html_report.
Every number comes from the BacktestResult / metrics surface; nothing is recomputed.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .html_builder import generate_html_report

_PCT_KEYS = {"CAGR", "Volatility", "Max Drawdown", "Avg Drawdown", "VaR (95%)",
             "CVaR (95%)", "Downside Dev", "Tracking Error", "Hit Rate"}


def plot_wealth(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.wealth.index, y=result.wealth.values, name=result.name))
    if result.benchmark is not None:
        fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                                 name="Benchmark"))
    fig.update_layout(title="Growth of $1", template="plotly_white", hovermode="x unified")
    return fig


def plot_drawdown(result):
    growth = result.wealth
    uw = (growth - growth.cummax()) / growth.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uw.index, y=uw.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title="Underwater Drawdown", template="plotly_white")
    return fig


def plot_weights(result):
    w = result.weights
    fig = go.Figure()
    for col in w.columns:
        fig.add_trace(go.Scatter(x=w.index, y=w[col].values, stackgroup="one", name=str(col)))
    fig.update_layout(title="Weights Over Time", template="plotly_white")
    return fig


def plot_turnover(result):
    t = result.turnover
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(t.index), y=t.values, name="Turnover"))
    fig.update_layout(title="Turnover per Rebalance", template="plotly_white")
    return fig


def plot_rolling(result, window=63):
    r = result.returns
    roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
    roll_vol = r.rolling(window).std() * np.sqrt(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                             name=f"{window}d Rolling Sharpe"))
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
                             name=f"{window}d Rolling Vol", yaxis="y2"))
    fig.update_layout(title="Rolling Sharpe & Vol", template="plotly_white",
                      yaxis2=dict(overlaying="y", side="right", title="Vol"))
    return fig


def _fmt(key, val):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return f"{val:.2%}" if key in _PCT_KEYS else f"{val:.2f}"


def _metrics_table_html(result):
    rows = "".join(f"<tr><td>{k}</td><td>{_fmt(k, v)}</td></tr>"
                   for k, v in result.metrics.items())
    return f'<table class="metrics-table">{rows}</table>'


def build_sections(result):
    return [{
        "title": f"Backtest — {result.name}",
        "main_content": [
            {"type": "plot", "title": "Growth of $1", "data": plot_wealth(result)},
            {"type": "plot", "title": "Underwater Drawdown", "data": plot_drawdown(result)},
            {"type": "plot", "title": "Weights Over Time", "data": plot_weights(result)},
            {"type": "plot", "title": "Turnover per Rebalance", "data": plot_turnover(result)},
            {"type": "plot", "title": "Rolling Sharpe & Vol", "data": plot_rolling(result)},
            {"type": "table_html", "title": "Performance Metrics",
             "data": _metrics_table_html(result)},
            {"type": "table_html", "title": "Trade Blotter",
             "data": result.blotter.to_html(classes="metrics-table", index=False)},
        ],
    }]


def _comparison_section(results):
    from .performance_stats import compare_strategies_oos
    oos = {nm: res.returns for nm, res in results.items()}
    comp = compare_strategies_oos(oos, n_trials=len(results))
    summary_df = pd.DataFrame(comp["summary"]).T
    fig = go.Figure()
    for nm, res in results.items():
        fig.add_trace(go.Scatter(x=res.wealth.index, y=res.wealth.values, name=nm))
    fig.update_layout(title="Strategy Wealth Comparison", template="plotly_white")
    return {
        "title": "OOS Strategy Comparison",
        "description": f"Best by DSR: {comp['best_by_dsr']}",
        "main_content": [
            {"type": "plot", "title": "Wealth Comparison", "data": fig},
            {"type": "table_html", "title": "OOS Stats (SR / PSR / DSR)",
             "data": summary_df.to_html(classes="metrics-table")},
        ],
    }


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

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest_report.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/backtest_report.py test/test_backtest_report.py
git commit -m "feat(backtest-report): interactive HTML report + plotly panels"
```

---

## Task 6: Wire exports + example + full-suite green

**Files:**
- Modify: `src/quant_reporter/__init__.py` (append before `__version__`)
- Create: `examples/example_strategy_report.py`

- [ ] **Step 1: Append exports to `__init__.py`**

Insert immediately BEFORE the final `__version__ = "2.0.0"` line:

```python
# --- SP-Strategy: shared metrics library ---
from .metrics import (
    cagr, annual_volatility, sharpe, sortino, calmar, omega,
    max_drawdown, avg_drawdown, ulcer_index, value_at_risk, conditional_var,
    downside_deviation, tracking_error, information_ratio, hit_rate,
    win_loss_ratio, tail_ratio, skewness, kurtosis, summary_metrics,
)

# --- SP-Strategy: objectives / loss surface ---
from .objectives import (
    neg_sharpe, variance, cvar_objective, tracking_error_objective,
    mean_squared_error, mean_absolute_error,
)

# --- SP-Strategy: prebuilt strategies ---
from .strategies import (
    equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe,
    trend_following, cross_sectional_momentum, vol_target_overlay, REGISTRY,
)

# --- SP-Strategy: strategy runner + result + report ---
from .strategy import Strategy, backtest, backtest_many, BacktestResult
from .backtest_report import create_backtest_report
```

- [ ] **Step 2: Verify the package imports and the full suite is green**

Run: `PYTHONPATH=src:test .venv/bin/python -c "import quant_reporter as qr; print(qr.backtest, qr.summary_metrics, qr.REGISTRY.keys())"`
Expected: prints the functions and the registry keys, no error.

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: PASS, count ≈ 233 + (new tests from Tasks 1–5). No failures.

- [ ] **Step 3: Create `examples/example_strategy_report.py`**

```python
"""example_strategy_report.py — define strategies, backtest, write an HTML report.

Offline (synthetic fixture). Run:
    python examples/example_strategy_report.py
Produces examples/Strategy_Report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import functools
from conftest import make_synthetic_prices
import quant_reporter as qr


def main():
    prices = make_synthetic_prices(seed=42, n_days=900)  # AAA/BBB/CCC + BMK
    cost = functools.partial(qr.transaction_cost_model, commission_bps=1.0, spread_bps=5.0)

    strategies = {
        "EqualWeight": qr.equal_weight,
        "RiskParity": qr.risk_parity,
        "TrendFollowing": qr.trend_following,
    }
    results = qr.backtest_many(strategies, prices, benchmark="BMK",
                               rebalance="M", cost_model=cost)

    for name, res in results.items():
        m = res.metrics
        print(f"{name:16s} CAGR={m['CAGR']:.2%}  Sharpe={m['Sharpe']:.2f}  "
              f"MaxDD={m['Max Drawdown']:.2%}  DSR={res.oos_stats['dsr']:.2f}")

    out = os.path.join(os.path.dirname(__file__), "Strategy_Report.html")
    qr.create_backtest_report(results, path=out)
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the example end-to-end**

Run: `PYTHONPATH=src:test .venv/bin/python examples/example_strategy_report.py`
Expected: prints a metrics line per strategy and `Report written: .../Strategy_Report.html`; the file exists and is non-empty.

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/__init__.py examples/example_strategy_report.py
git commit -m "feat(sp-strategy): wire exports + end-to-end strategy report example"
```

---

## Self-Review

**Spec coverage:**
- §2 strategy model → Task 4 (`Strategy`, `_resolve_weights` accepts callable/Strategy/dict/DataFrame).
- §3 runner + result → Task 4 (`backtest`, `backtest_many`, `BacktestResult` with cached `metrics`/`oos_stats`).
- §4 report panels (10) → Task 5 (`build_sections` + `_comparison_section`; KPI summary carried in `result.summary`+`oos_stats`, surfaced via metrics table + comparison panel).
- §5a metrics → Task 1; §5b objectives → Task 2.
- §6 prebuilt strategies + REGISTRY → Task 3.
- §7 files + sequential `__init__` → Task 6 (only place `__init__` is touched).
- §8 testing/safety → every task ends with a suite-green gate; causality property test in Task 3; frictionless-equivalence pin in Task 4.

**Placeholder scan:** No TBD/TODO; every code step shows full code. The one conditional note (Task 3 Step 4) gives the exact fix.

**Type consistency:** `BacktestResult` fields (`wealth, weights, blotter, turnover, cost_drag, benchmark, summary, n_trials`) defined in Task 4 are exactly those read in Task 5. `summary_metrics` keys defined in Task 1 (`CAGR`, `Max Drawdown`, …) match `_PCT_KEYS` and the report's metric table in Task 5 and the example's `m['CAGR']` access in Task 6. `create_backtest_report(result_or_results, path, open_browser)` signature matches the call in `BacktestResult.report` (Task 4) and the example (Task 6). `objectives.neg_sharpe`/`variance` arg order matches `find_optimal_portfolio` usage (verified by a test in Task 2).
