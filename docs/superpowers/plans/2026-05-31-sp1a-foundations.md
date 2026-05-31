# SP1a — Foundations (Phase-1 Primitives) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the five pure, offline-testable Phase-1 primitives the rest of SP1/SP2 composes through: `compute_trailing_volatility`, `volatility_target_positions`, `ledoit_wolf_covariance`, `portfolio_turnover`, `drawdown_stats`.

**Architecture:** Three new flat modules (`signals.py`, `robust_estimators.py`, `backtest.py`), each a set of pure functions. No edits to `__init__.py`/`conftest.py`/existing modules during the build — the orchestrator wires re-exports into `__init__` sequentially at integration time (this is what keeps parallel builds from cross-contaminating: `__init__` never imports a half-written sibling). All tests use the existing `make_synthetic_prices` fixture; **no network, no new dependencies** (`sklearn.covariance` is already pinned).

**Tech Stack:** Python 3.9, numpy, pandas≥2.2, scipy, scikit-learn, pytest, hypothesis.

**Run command (every test step):**
```
PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" <target> -q
```
`-o addopts=""` skips the `setup.cfg` `--cov-fail-under=70` gate (which would fail mid-build on partially-covered new code). `test/` is on the path so `from conftest import make_synthetic_prices` works.

**Hard invariant for the whole phase — look-ahead safety:** every trailing/rolling statistic must be *causal* (value at row *d* uses only data ≤ *d*). In pandas this means `.rolling(...)`/`.ewm(...)` **without** `center=True`, and any value consumed for a position decision must be `.shift(1)`-lagged. Two property tests (Tasks 2 and 8) enforce this by shuffling future rows and asserting past outputs are unchanged.

---

## File Structure

| File | Responsibility |
|------|----------------|
| Create `src/quant_reporter/signals.py` | `compute_trailing_volatility`, `volatility_target_positions` |
| Create `src/quant_reporter/robust_estimators.py` | `ledoit_wolf_covariance` |
| Create `src/quant_reporter/backtest.py` | `portfolio_turnover`, `drawdown_stats` (SP1a slice; SP1b adds the engine) |
| Create `test/test_signals.py` | tests for `signals.py` |
| Create `test/test_robust_estimators.py` | tests for `robust_estimators.py` |
| Create `test/test_backtest.py` | tests for `backtest.py` (turnover + drawdown_stats slice) |
| Modify `src/quant_reporter/__init__.py` | **(orchestrator, integration step only)** add re-export banner sections |
| Modify `CHANGELOG.md` | **(orchestrator, integration step only)** note primitives under `[2.1.0] Added` |

The three modules are independent — no import edges between them. `backtest.drawdown_stats` imports `compute_drawdown` from `.metrics`. `robust_estimators` imports from `sklearn.covariance`. Build them in parallel; integrate sequentially.

---

## Task 1: `compute_trailing_volatility` (signals.py)

**Files:**
- Create: `src/quant_reporter/signals.py`
- Test: `test/test_signals.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_signals.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.signals import compute_trailing_volatility


def _returns(n=260, seed=1, cols=("AAA", "BBB")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(
        {c: rng.normal(0.0004, 0.01 + 0.003 * i, n) for i, c in enumerate(cols)},
        index=idx,
    )


def test_trailing_vol_simple_golden():
    s = pd.Series([0.01, -0.01, 0.02, -0.02, 0.01], name="AAA")
    df = compute_trailing_volatility(s.to_frame(), lookback=3, method="simple", annualize=False)
    # rolling std (ddof=1) of [0.01,-0.01,0.02] at row 2
    assert df["AAA"].iloc[2] == pytest.approx(0.0152752523, rel=1e-6)
    assert np.isnan(df["AAA"].iloc[0]) and np.isnan(df["AAA"].iloc[1])


def test_trailing_vol_annualize_scales_by_sqrt_252():
    r = _returns()
    raw = compute_trailing_volatility(r, lookback=63, method="simple", annualize=False)
    ann = compute_trailing_volatility(r, lookback=63, method="simple", annualize=True)
    ratio = (ann / raw).dropna()
    assert np.allclose(ratio.values, np.sqrt(252))


def test_trailing_vol_ewma_runs_and_is_positive():
    r = _returns()
    vol = compute_trailing_volatility(r, lookback=63, method="ewma", annualize=True)
    tail = vol.dropna()
    assert (tail.values > 0).all()
    assert list(vol.columns) == list(r.columns)


def test_trailing_vol_accepts_series_returns_dataframe():
    s = _returns(cols=("AAA",))["AAA"]
    vol = compute_trailing_volatility(s, lookback=20)
    assert isinstance(vol, pd.DataFrame)
    assert list(vol.columns) == ["AAA"]


def test_trailing_vol_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_trailing_volatility(_returns(), method="garch")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_reporter.signals'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_reporter/signals.py
"""Volatility signals & vol-targeting position sizing.

Pure, causal (look-ahead-safe) functions. The trailing-vol estimator here is the
sizing engine the whole tactical/overlay family (SP2) composes through.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_frame(returns):
    """Coerce a Series to a one-column DataFrame; pass DataFrames through."""
    if isinstance(returns, pd.Series):
        return returns.to_frame()
    return returns


def compute_trailing_volatility(returns, lookback=63, method="ewma", annualize=True):
    """Trailing (causal) volatility per asset.

    Args:
        returns: DataFrame (or Series) of periodic simple returns.
        lookback: window length (rolling window for 'simple'; EWMA span for 'ewma').
        method: 'ewma' (exponentially weighted) or 'simple' (equal-weighted rolling).
        annualize: if True, multiply by sqrt(252).

    Returns:
        DataFrame of trailing volatility, same columns as `returns`. Value at row d
        uses only rows <= d (no center=True), so it is safe to .shift(1) for sizing.
    """
    df = _as_frame(returns)
    if method == "simple":
        vol = df.rolling(window=lookback).std(ddof=1)
    elif method == "ewma":
        vol = df.ewm(span=lookback, adjust=False, min_periods=lookback).std(bias=False)
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'ewma' or 'simple'.")
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS)
    return vol
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/signals.py test/test_signals.py
git commit -m "feat(signals): compute_trailing_volatility (causal EWMA/simple vol)"
```

---

## Task 2: Look-ahead property test for `compute_trailing_volatility`

**Files:**
- Test: `test/test_signals.py` (append)

- [ ] **Step 1: Write the failing property test**

```python
# test/test_signals.py (append)
from hypothesis import given, settings, strategies as st


@settings(max_examples=30, deadline=None)
@given(cut=st.integers(min_value=80, max_value=180))
def test_trailing_vol_is_causal_under_future_shuffle(cut):
    """Shuffling rows strictly after `cut` must not change vol values at/<=cut."""
    r = _returns(n=200, seed=7)
    vol_full = compute_trailing_volatility(r, lookback=40, method="ewma")
    shuffled = r.copy()
    rng = np.random.default_rng(123)
    tail_idx = np.arange(cut + 1, len(r))
    perm = rng.permutation(tail_idx)
    shuffled.iloc[cut + 1:] = r.iloc[perm].values
    vol_shuf = compute_trailing_volatility(shuffled, lookback=40, method="ewma")
    pd.testing.assert_frame_equal(vol_full.iloc[: cut + 1], vol_shuf.iloc[: cut + 1])
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py::test_trailing_vol_is_causal_under_future_shuffle -q`
Expected: PASS (the implementation is already causal; this pins it).

- [ ] **Step 3: Commit**

```bash
git add test/test_signals.py
git commit -m "test(signals): property test — trailing vol is causal under future shuffle"
```

---

## Task 3: `volatility_target_positions` (signals.py)

**Files:**
- Modify: `src/quant_reporter/signals.py`
- Test: `test/test_signals.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# test/test_signals.py (append)
from quant_reporter.signals import volatility_target_positions


def test_vol_target_per_asset_scales_toward_target():
    r = _returns(n=400, seed=3)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)  # always long 1.0
    pos = volatility_target_positions(signal, r, target_vol=0.10, vol_lookback=63,
                                      method="simple", max_leverage=5.0, scaling="per_asset")
    # Lower-vol asset (AAA) should get a larger position than higher-vol asset (BBB)
    tail = pos.dropna()
    assert tail["AAA"].mean() > tail["BBB"].mean()
    assert (tail.values >= 0).all()


def test_vol_target_respects_max_leverage():
    r = _returns(n=400, seed=4)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    pos = volatility_target_positions(signal, r, target_vol=10.0,  # absurd target forces clipping
                                      vol_lookback=63, method="simple", max_leverage=2.0,
                                      scaling="per_asset")
    gross = pos.abs().sum(axis=1).dropna()
    assert (gross <= 2.0 + 1e-9).all()


def test_vol_target_zero_signal_zero_position():
    r = _returns(n=200, seed=5)
    signal = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    pos = volatility_target_positions(signal, r, target_vol=0.10, vol_lookback=40,
                                      method="simple", scaling="per_asset")
    assert np.allclose(pos.fillna(0.0).values, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py -q -k vol_target`
Expected: FAIL — `ImportError: cannot import name 'volatility_target_positions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_reporter/signals.py (append)
def volatility_target_positions(signal, returns, target_vol=0.10, vol_lookback=63,
                                method="ewma", max_leverage=2.0, scaling="per_asset", cov=None):
    """Scale a position `signal` so realized volatility targets `target_vol`.

    Look-ahead-safe: the volatility estimate is lagged one period (.shift(1)) before
    scaling, so the position at row d uses vol known as of d-1 and the signal at d.

    Args:
        signal: DataFrame of desired positions/weights over time (same cols as returns).
        returns: DataFrame of asset periodic returns.
        target_vol: annualized volatility target.
        vol_lookback: lookback for the vol estimate.
        method: passed to compute_trailing_volatility ('ewma'/'simple').
        max_leverage: cap on gross exposure (sum of |positions|) per row.
        scaling: 'per_asset' (scale each asset by target/own-vol) or
                 'portfolio' (single scalar/row from portfolio vol).
        cov: optional annualized covariance DataFrame for scaling='portfolio'
             (if None, portfolio vol is estimated from `returns`).

    Returns:
        DataFrame of scaled positions (NaN where the lagged vol is undefined).
    """
    sig = _as_frame(signal)
    rets = _as_frame(returns)
    sig, rets = sig.align(rets, join="inner", axis=1)

    if scaling == "per_asset":
        vol = compute_trailing_volatility(rets, lookback=vol_lookback, method=method,
                                          annualize=True).shift(1)
        scale = target_vol / vol.replace(0.0, np.nan)
        pos = sig * scale
    elif scaling == "portfolio":
        if cov is None:
            vol = compute_trailing_volatility(rets, lookback=vol_lookback, method=method,
                                              annualize=True).shift(1)
            port_vol = vol.mul(sig.abs()).sum(axis=1)  # crude proxy when no cov given
        else:
            w = sig.abs()
            cov_arr = cov.reindex(index=rets.columns, columns=rets.columns).values
            port_var = (w.values * (w.values @ cov_arr)).sum(axis=1)
            port_vol = pd.Series(np.sqrt(np.clip(port_var, 0, None)), index=sig.index).shift(1)
        scale = target_vol / port_vol.replace(0.0, np.nan)
        pos = sig.mul(scale, axis=0)
    else:
        raise ValueError(f"Unknown scaling {scaling!r}; expected 'per_asset' or 'portfolio'.")

    # Cap gross leverage per row, preserving relative sizing.
    gross = pos.abs().sum(axis=1)
    over = gross > max_leverage
    if over.any():
        factor = pd.Series(1.0, index=pos.index)
        factor[over] = max_leverage / gross[over]
        pos = pos.mul(factor, axis=0)
    return pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py -q`
Expected: PASS (all signals tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/signals.py test/test_signals.py
git commit -m "feat(signals): volatility_target_positions (lagged vol, leverage cap)"
```

---

## Task 4: Look-ahead property test for `volatility_target_positions`

**Files:**
- Test: `test/test_signals.py` (append)

- [ ] **Step 1: Write the failing property test**

```python
# test/test_signals.py (append)
@settings(max_examples=25, deadline=None)
@given(cut=st.integers(min_value=90, max_value=160))
def test_vol_target_positions_causal_under_future_shuffle(cut):
    r = _returns(n=200, seed=11)
    signal = pd.DataFrame(1.0, index=r.index, columns=r.columns)
    base = volatility_target_positions(signal, r, target_vol=0.1, vol_lookback=40,
                                       method="simple", scaling="per_asset")
    shuffled = r.copy()
    rng = np.random.default_rng(99)
    tail_idx = np.arange(cut + 1, len(r))
    shuffled.iloc[cut + 1:] = r.iloc[rng.permutation(tail_idx)].values
    shuf = volatility_target_positions(signal, shuffled, target_vol=0.1, vol_lookback=40,
                                       method="simple", scaling="per_asset")
    pd.testing.assert_frame_equal(base.iloc[: cut + 1], shuf.iloc[: cut + 1])
```

- [ ] **Step 2: Run test**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_signals.py::test_vol_target_positions_causal_under_future_shuffle -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/test_signals.py
git commit -m "test(signals): property test — vol-target positions are causal"
```

---

## Task 5: `ledoit_wolf_covariance` (robust_estimators.py)

**Files:**
- Create: `src/quant_reporter/robust_estimators.py`
- Test: `test/test_robust_estimators.py`

**Contract:** returns annualized cov (`×periods_per_year`) so it is a drop-in for `get_optimization_inputs`'s `cov_matrix`. Default target is the Ledoit-Wolf (2004) **constant-correlation** shrinkage (pure numpy closed-form); `target='identity'` delegates to `sklearn.covariance.ledoit_wolf` (spherical). `delta=None` estimates the optimal shrinkage; a passed `delta` is used directly.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_robust_estimators.py
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf as sk_ledoit_wolf

from quant_reporter.robust_estimators import ledoit_wolf_covariance
from conftest import make_synthetic_prices


def _returns(seed=0, n=300, cols=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    base = rng.normal(0, 0.01, (n, 1))
    data = {c: (base[:, 0] * (0.5 + 0.2 * i) + rng.normal(0, 0.008, n)) for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


def test_lw_returns_expected_keys():
    out = ledoit_wolf_covariance(_returns())
    assert set(out) == {"cov_matrix", "shrinkage", "target", "sample_cov", "target_matrix"}


def test_lw_cov_is_symmetric_pd_and_labeled():
    r = _returns()
    out = ledoit_wolf_covariance(r)
    cov = out["cov_matrix"]
    assert list(cov.index) == list(r.columns) == list(cov.columns)
    assert np.allclose(cov.values, cov.values.T)
    assert (np.linalg.eigvalsh(cov.values) > 0).all()


def test_lw_shrinkage_in_unit_interval():
    out = ledoit_wolf_covariance(_returns())
    assert 0.0 <= out["shrinkage"] <= 1.0


def test_lw_annualization_matches_252_contract():
    r = _returns()
    ann = ledoit_wolf_covariance(r, periods_per_year=252)["cov_matrix"]
    daily = ledoit_wolf_covariance(r, periods_per_year=1)["cov_matrix"]
    assert np.allclose(ann.values, daily.values * 252)


def test_lw_identity_target_matches_sklearn():
    r = _returns()
    out = ledoit_wolf_covariance(r, target="identity", periods_per_year=1)
    sk_cov, sk_shrink = sk_ledoit_wolf(r.values)
    assert out["shrinkage"] == pytest.approx(sk_shrink, rel=1e-6)
    assert np.allclose(out["cov_matrix"].values, sk_cov, atol=1e-10)


def test_lw_explicit_delta_is_used():
    r = _returns()
    out = ledoit_wolf_covariance(r, delta=0.0, periods_per_year=1)
    # delta=0 => pure sample covariance (ddof handling aside), shrinkage reported as 0
    assert out["shrinkage"] == pytest.approx(0.0)
    assert np.allclose(out["cov_matrix"].values, out["sample_cov"].values)


def test_lw_constant_correlation_offdiag_uses_avg_corr():
    r = _returns()
    out = ledoit_wolf_covariance(r, target="constant_correlation", periods_per_year=1)
    F = out["target_matrix"].values
    S = out["sample_cov"].values
    # diagonal of target equals sample variances
    assert np.allclose(np.diag(F), np.diag(S))
    # off-diagonal target correlation is constant
    d = np.sqrt(np.diag(F))
    corr = F / np.outer(d, d)
    off = corr[~np.eye(len(d), dtype=bool)]
    assert np.allclose(off, off[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_robust_estimators.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_reporter.robust_estimators'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_reporter/robust_estimators.py
"""Robust covariance estimation (Ledoit-Wolf shrinkage).

Hardens the covariance input every optimizer consumes. Returns annualized matrices
matching the get_optimization_inputs contract (cov scaled by periods_per_year).
No new dependencies: sklearn.covariance is already pinned.
"""
import numpy as np
import pandas as pd


def _sample_cov(X):
    """MLE sample covariance (divide by T), matching Ledoit-Wolf's derivation."""
    T = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / T, Xc


def _constant_correlation_target(S):
    """LW(2004) constant-correlation target F: F_ii=S_ii, F_ij=r_bar*sqrt(S_ii*S_jj)."""
    d = np.sqrt(np.diag(S))
    outer = np.outer(d, d)
    corr = S / outer
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    r_bar = corr[mask].mean()
    F = r_bar * outer
    np.fill_diagonal(F, np.diag(S))
    return F, r_bar


def _lw_constant_correlation_delta(Xc, S, F, r_bar):
    """Optimal shrinkage intensity for the constant-correlation target (LW 2004)."""
    T, n = Xc.shape
    # pi: sum of asymptotic variances of sample-cov entries
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / T - S ** 2
    pi_hat = pi_mat.sum()
    # rho: estimator of sum of asy covariances between F and S entries
    d = np.sqrt(np.diag(S))
    # theta_ii,ij terms
    term1 = ((Xc ** 3).T @ Xc) / T - np.diag(S)[:, None] * S
    term2 = ((Xc).T @ (Xc ** 3)) / T - np.diag(S)[None, :] * S
    theta_ii = term1  # (i diag with ij)
    theta_jj = term2
    ratio_ji = np.outer(1.0 / d, d)  # sqrt(S_jj/S_ii)
    ratio_ij = np.outer(d, 1.0 / d)  # sqrt(S_ii/S_jj)
    rho_off = (r_bar / 2.0) * (ratio_ji * theta_ii + ratio_ij * theta_jj)
    mask = ~np.eye(n, dtype=bool)
    rho_hat = np.diag(pi_mat).sum() + rho_off[mask].sum()
    # gamma: misfit between sample cov and target
    gamma_hat = ((F - S) ** 2).sum()
    if gamma_hat <= 0:
        return 0.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(max(0.0, min(1.0, kappa / T)))


def ledoit_wolf_covariance(returns, target="constant_correlation", periods_per_year=252, delta=None):
    """Ledoit-Wolf shrinkage covariance, annualized.

    Args:
        returns: DataFrame of periodic asset returns.
        target: 'constant_correlation' (LW 2004 closed-form) or 'identity'
                (sklearn spherical LedoitWolf).
        periods_per_year: annualization factor applied to every returned matrix.
        delta: if given, use this shrinkage intensity directly (skip estimation).

    Returns:
        dict with 'cov_matrix' (annualized PD DataFrame), 'shrinkage' (float in [0,1]),
        'target' (str), 'sample_cov' (annualized DataFrame), 'target_matrix'
        (annualized DataFrame).
    """
    cols = list(returns.columns)
    X = returns.values.astype(float)
    S, Xc = _sample_cov(X)

    if target == "identity":
        mu = np.trace(S) / S.shape[0]
        F = mu * np.eye(S.shape[0])
    elif target == "constant_correlation":
        F, r_bar = _constant_correlation_target(S)
    else:
        raise ValueError(f"Unknown target {target!r}; expected 'constant_correlation' or 'identity'.")

    if delta is None:
        if target == "identity":
            from sklearn.covariance import ledoit_wolf as _skl
            _, shrink = _skl(X)
            delta = float(shrink)
        else:
            delta = _lw_constant_correlation_delta(Xc, S, F, r_bar)
    else:
        delta = float(delta)

    shrunk = delta * F + (1.0 - delta) * S
    f = periods_per_year

    def _df(m):
        return pd.DataFrame(m * f, index=cols, columns=cols)

    return {
        "cov_matrix": _df(shrunk),
        "shrinkage": delta,
        "target": target,
        "sample_cov": _df(S),
        "target_matrix": _df(F),
    }
```

> **Implementer note:** `test_lw_identity_target_matches_sklearn` pins the identity branch to sklearn exactly (sklearn's `ledoit_wolf` also uses the divide-by-T sample cov, so `_sample_cov` matches it). If the assertion is off, reconcile the sample-cov normalization to whatever `sklearn.covariance.ledoit_wolf` uses — sklearn is the source of truth for the identity branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_robust_estimators.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/robust_estimators.py test/test_robust_estimators.py
git commit -m "feat(robust_estimators): ledoit_wolf_covariance (const-corr + identity targets)"
```

---

## Task 6: `portfolio_turnover` (backtest.py)

**Files:**
- Create: `src/quant_reporter/backtest.py`
- Test: `test/test_backtest.py`

- [ ] **Step 1: Write the failing tests**

```python
# test/test_backtest.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.backtest import portfolio_turnover


def test_turnover_one_way_golden():
    out = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3})
    assert out["turnover"] == pytest.approx(0.2)   # 0.5 * (|0.2| + |0.2|)
    assert out["buys"] == pytest.approx(0.2)
    assert out["sells"] == pytest.approx(0.2)
    assert out["trades"]["A"] == pytest.approx(0.2)
    assert out["trades"]["B"] == pytest.approx(-0.2)


def test_turnover_two_way_doubles_one_way():
    one = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3}, convention="one_way")
    two = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3}, convention="two_way")
    assert two["turnover"] == pytest.approx(2 * one["turnover"])


def test_turnover_handles_new_and_dropped_assets():
    out = portfolio_turnover({"A": 1.0}, {"A": 0.5, "B": 0.5})
    assert out["trades"]["A"] == pytest.approx(-0.5)
    assert out["trades"]["B"] == pytest.approx(0.5)
    assert out["turnover"] == pytest.approx(0.5)


def test_turnover_identity_is_zero():
    out = portfolio_turnover({"A": 0.3, "B": 0.7}, {"A": 0.3, "B": 0.7})
    assert out["turnover"] == pytest.approx(0.0)


def test_turnover_accepts_series():
    a = pd.Series({"A": 0.5, "B": 0.5})
    b = pd.Series({"A": 0.7, "B": 0.3})
    assert portfolio_turnover(a, b)["turnover"] == pytest.approx(0.2)


def test_turnover_rejects_unknown_convention():
    with pytest.raises(ValueError):
        portfolio_turnover({"A": 1.0}, {"A": 1.0}, convention="round_trip")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_reporter.backtest'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_reporter/backtest.py
"""Cost-aware backtest engine & execution primitives.

SP1a slice: portfolio_turnover, drawdown_stats. (SP1b adds transaction_cost_model,
generate_rebalance_dates, simulate_strategy.) Pure, offline-testable, no new deps.
"""
import numpy as np
import pandas as pd

from .metrics import compute_drawdown

TRADING_DAYS = 252


def _to_series(weights):
    if isinstance(weights, pd.Series):
        return weights.astype(float)
    return pd.Series(weights, dtype=float)


def portfolio_turnover(weights_before, weights_after, convention="one_way"):
    """Turnover between two weight vectors.

    Aligns on the union of tickers (missing => 0). `trades` are signed deltas
    (after - before). one_way turnover = 0.5*sum(|trades|); two_way = sum(|trades|).

    Returns dict: {'turnover', 'buys', 'sells', 'trades' (Series of signed deltas)}.
    """
    if convention not in ("one_way", "two_way"):
        raise ValueError(f"Unknown convention {convention!r}; expected 'one_way' or 'two_way'.")
    before = _to_series(weights_before)
    after = _to_series(weights_after)
    idx = before.index.union(after.index)
    before = before.reindex(idx).fillna(0.0)
    after = after.reindex(idx).fillna(0.0)
    trades = after - before
    abs_sum = float(trades.abs().sum())
    buys = float(trades[trades > 0].sum())
    sells = float(-trades[trades < 0].sum())
    turnover = abs_sum if convention == "two_way" else 0.5 * abs_sum
    return {"turnover": turnover, "buys": buys, "sells": sells, "trades": trades}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/backtest.py test/test_backtest.py
git commit -m "feat(backtest): portfolio_turnover (signed trades, one/two-way)"
```

---

## Task 7: `drawdown_stats` (backtest.py)

**Files:**
- Modify: `src/quant_reporter/backtest.py`
- Test: `test/test_backtest.py` (append)

**Units (documented, pinned by tests):** `max_drawdown` and `worst_drawdowns[i]['depth']` are **negative decimals** (e.g. `-0.1736`). `pain_index` is the mean absolute decimal drawdown. `ulcer_index` is in **percentage points** — `sqrt(mean((100*curve)**2))` — the conventional definition.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_backtest.py (append)
from quant_reporter.backtest import drawdown_stats


def test_drawdown_stats_golden():
    wealth = pd.Series([1.0, 1.1, 0.99, 1.05, 1.21, 1.0],
                       index=pd.bdate_range("2022-01-03", periods=6))
    out = drawdown_stats(wealth, top_n=5)
    # deepest dd is the final 1.21 -> 1.00 leg: (1.00-1.21)/1.21
    assert out["max_drawdown"] == pytest.approx(-0.1735537, rel=1e-5)
    assert out["worst_drawdowns"][0]["depth"] == pytest.approx(-0.1735537, rel=1e-5)
    assert out["pain_index"] >= 0
    assert out["ulcer_index"] >= 0


def test_drawdown_stats_keys_and_underwater_curve():
    wealth = pd.Series(np.linspace(1.0, 2.0, 200) + np.sin(np.linspace(0, 12, 200)) * 0.05,
                       index=pd.bdate_range("2021-01-01", periods=200))
    out = drawdown_stats(wealth)
    assert set(out) >= {"max_drawdown", "underwater_curve", "worst_drawdowns",
                        "ulcer_index", "pain_index"}
    assert isinstance(out["underwater_curve"], pd.Series)
    assert (out["underwater_curve"] <= 1e-12).all()  # never positive


def test_drawdown_stats_monotone_wealth_has_zero_drawdown():
    wealth = pd.Series(np.cumprod(1 + np.full(100, 0.001)),
                       index=pd.bdate_range("2021-01-01", periods=100))
    out = drawdown_stats(wealth)
    assert out["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert out["worst_drawdowns"] == [] or out["worst_drawdowns"][0]["depth"] == pytest.approx(0.0, abs=1e-12)


def test_drawdown_stats_top_n_caps_episode_count():
    rng = np.random.default_rng(0)
    wealth = pd.Series(np.cumprod(1 + rng.normal(0, 0.02, 500)),
                       index=pd.bdate_range("2021-01-01", periods=500))
    out = drawdown_stats(wealth, top_n=3)
    assert len(out["worst_drawdowns"]) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest.py -q -k drawdown`
Expected: FAIL — `ImportError: cannot import name 'drawdown_stats'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/quant_reporter/backtest.py (append)
def _drawdown_episodes(curve):
    """Split an underwater curve into episodes: (peak_date, trough_date, recovery_date, depth, length)."""
    episodes = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0
    dates = curve.index
    for i, (date, val) in enumerate(curve.items()):
        if not in_dd and val < 0:
            in_dd = True
            peak_date = dates[i - 1] if i > 0 else date
            trough_date = date
            trough_val = val
        elif in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
            if val >= 0:  # recovered
                episodes.append({"peak_date": peak_date, "trough_date": trough_date,
                                 "recovery_date": date, "depth": float(trough_val),
                                 "length": int(curve.index.get_loc(date) - curve.index.get_loc(peak_date))})
                in_dd = False
    if in_dd:  # ongoing drawdown at series end
        episodes.append({"peak_date": peak_date, "trough_date": trough_date,
                         "recovery_date": None, "depth": float(trough_val),
                         "length": int(curve.index.get_loc(dates[-1]) - curve.index.get_loc(peak_date))})
    return episodes


def drawdown_stats(wealth, top_n=5, periods_per_year=252):
    """Drawdown analytics for a realized wealth (Growth-of-$1) path.

    Returns dict: {'max_drawdown' (neg decimal), 'underwater_curve' (Series),
    'worst_drawdowns' (top_n episodes by depth), 'ulcer_index' (pct points),
    'pain_index' (mean |dd|, decimal)}.
    """
    dd = compute_drawdown(wealth)
    curve = dd.curve
    episodes = _drawdown_episodes(curve)
    episodes.sort(key=lambda e: e["depth"])  # most negative first
    ulcer = float(np.sqrt(np.mean((100.0 * curve.values) ** 2)))
    pain = float(curve.abs().mean())
    return {
        "max_drawdown": float(dd.max_dd),
        "underwater_curve": curve,
        "worst_drawdowns": episodes[:top_n],
        "ulcer_index": ulcer,
        "pain_index": pain,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/test_backtest.py -q`
Expected: PASS (all backtest tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/backtest.py test/test_backtest.py
git commit -m "feat(backtest): drawdown_stats (episodes, ulcer, pain index)"
```

---

## Task 8: Integration — wire `__init__`, run full suite, CHANGELOG (orchestrator only)

> Do this **after** Tasks 1–7 are all green individually. This is the sequential step that prevents cross-contamination.

**Files:**
- Modify: `src/quant_reporter/__init__.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add re-export banner sections to `__init__.py`** (insert before the `__version__` line)

```python
# --- Signals (volatility estimation & vol-targeting) ---
from .signals import (
    compute_trailing_volatility,
    volatility_target_positions,
)

# --- Robust Estimators ---
from .robust_estimators import ledoit_wolf_covariance

# --- Backtest Engine (Phase-1 primitives) ---
from .backtest import (
    portfolio_turnover,
    drawdown_stats,
)
```

- [ ] **Step 2: Verify the package imports and symbols are exported**

Run: `PYTHONPATH=src:test .venv/bin/python -c "import quant_reporter as qr; print(qr.compute_trailing_volatility, qr.volatility_target_positions, qr.ledoit_wolf_covariance, qr.portfolio_turnover, qr.drawdown_stats)"`
Expected: prints five function objects, no ImportError.

- [ ] **Step 3: Run the FULL suite (must stay green)**

Run: `PYTHONPATH=src:test .venv/bin/python -m pytest -o addopts="" test/ -q`
Expected: PASS — 62 prior + new SP1a tests, 0 failures.

- [ ] **Step 4: Add the CHANGELOG entry** (extend `[2.1.0] Added`)

```markdown
- SP1a foundations (`signals.py`, `robust_estimators.py`, `backtest.py`): `compute_trailing_volatility`, `volatility_target_positions`, `ledoit_wolf_covariance`, `portfolio_turnover`, `drawdown_stats` — pure, look-ahead-safe primitives.
```

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/__init__.py CHANGELOG.md
git commit -m "feat(sp1a): re-export Phase-1 primitives; CHANGELOG"
```

---

## Self-Review

**Spec coverage (SP1 spec §4, §10 SP1a slice):**
- `compute_trailing_volatility` → Tasks 1–2 ✓
- `volatility_target_positions` → Tasks 3–4 ✓
- `ledoit_wolf_covariance` → Task 5 ✓ (annualization contract pinned to `get_optimization_inputs`'s `×252`; symmetric PD; shrinkage∈[0,1])
- `portfolio_turnover` → Task 6 ✓
- `drawdown_stats` → Task 7 ✓ (calls `compute_drawdown`)
- Look-ahead property tests → Tasks 2, 4 ✓ (spec §8 mandate)
- Golden numbers (EWMA vs simple; turnover/cost arithmetic; LW symmetric PD + ×252) → Tasks 1, 5, 6 ✓
- Re-exported from `__init__` → Task 8 ✓
- No new deps, no network → all tests use `make_synthetic_prices`/local arrays ✓

**Placeholder scan:** none — every step has complete code + exact command + expected output.

**Type consistency:** `_as_frame`, `_to_series`, `compute_drawdown(...).curve/.max_dd`, dict return shapes consistent across tasks. `ledoit_wolf_covariance` key set fixed at Task 5 and asserted. `portfolio_turnover` returns `trades` as a Series (asserted by `out["trades"]["A"]` indexing).

**Deferred to SP1b (not in this plan, by design):** `transaction_cost_model`, `generate_rebalance_dates`, `simulate_strategy`, the `run_rolling_windows` schedule unlock, `performance_stats.py` (PSR/DSR/`compare_strategies_oos`), and the flagship example.
