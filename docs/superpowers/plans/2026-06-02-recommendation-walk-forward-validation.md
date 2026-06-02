# Recommendation Walk-Forward Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add profile-aware walk-forward (OOS) validation of recommendations by extending the existing `run_rolling_windows` engine and wiring `recommend(validate=True)` → `RecommendationValidation`, with the existing validation report and tests left byte-identical.

**Architecture:** Extract the per-window train/test loop from `run_rolling_windows` into a shared private `_rolling_oos_sharpe` core driven by a `{name -> fn(train, mean, cov) -> weights}` strategy dict; `run_rolling_windows` becomes a thin wrapper (behavior-identical) that can optionally add a profile-constrained "Recommended" strategy; `recommend()` calls the same core on raw prices via a `walk_forward_recommendation` helper that reuses `apply_constraints` and `calculate_overfitting_score`.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy (via `opt_core`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-recommendation-walk-forward-validation-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/quant_reporter/validation_report.py` | **MODIFY** — extract `_rolling_oos_sharpe`; refactor `run_rolling_windows` to a thin wrapper + optional `objective`/`profile` |
| `src/quant_reporter/recommendation.py` | **MODIFY** — `RecommendationValidation` dataclass, `walk_forward_recommendation` helper, `recommend(validate=…)`, `Recommendation.validation` + rendering |
| `src/quant_reporter/__init__.py` | **MODIFY** — export `RecommendationValidation`, `walk_forward_recommendation` |
| `test/test_recommendation_validation.py` | **NEW** |
| `test/test_validation_unlock.py` | **UNCHANGED** — regression gate, must stay green |

**Key reference facts (verified in the codebase):**
- `validation_report.py` imports (top): `ReturnsBundle, compute_metrics` from `.analytics`; `get_optimization_inputs, find_optimal_portfolio, objective_neg_sharpe, objective_min_variance, get_portfolio_price, build_constraints` from `.opt_core`. It does **not** import `recommendation` (so `recommendation` may import from it safely).
- Current `run_rolling_windows` builds 4 strategies (`Equal Wt`, `Min Vol`, `Max Sharpe`, `User Portfolio`) with **unconstrained** bounds, computes `get_optimization_inputs` once per window, and records each strategy's `compute_metrics(...)["Realized Sharpe"]`. Output: `rolling_df` indexed by `"Test Period"`, columns `"{name} Sharpe"`; optional `schedule` dict (per-strategy weight DataFrames).
- `calculate_overfitting_score(is_metrics, oos_metrics)` (validation_report.py:32) returns a DataFrame with `"Overfitting Score"` = `max(0, (IS_Sharpe − OOS_Sharpe)/IS_Sharpe)`. Inputs are dict-of-dicts keyed by name with `"Realized Sharpe"`/`"Realized CAGR"`.
- `recommendation.py` already imports `get_optimization_inputs, get_portfolio_stats, find_optimal_portfolio, build_constraints, get_portfolio_price, risk_contributions` from `.opt_core`, and `neg_sharpe` from `.objectives`. It does **not** yet import from `.analytics`.
- `apply_constraints(profile, columns, *, sector_map=None) -> (bounds, constraints)` exists in `planning.py` (raises `ValueError` if `n_effective * max_position_weight < invested_budget`).
- Test data: `test/conftest.py::make_synthetic_prices(seed=42, n_days=756, assets=("AAA","BBB","CCC"), benchmark="BMK")` → asset columns + a `BMK` column. `qr.build_context_from_prices(prices, weights, benchmark, train_start, train_end)` builds a `ReportContext`.

---

## Task 1: Extract `_rolling_oos_sharpe` core (behavior-preserving refactor)

**Files:**
- Modify: `src/quant_reporter/validation_report.py:64-148` (the `run_rolling_windows` function)
- Test: `test/test_recommendation_validation.py`

- [ ] **Step 1: Write the failing test**

Create `test/test_recommendation_validation.py`:

```python
import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_prices
import quant_reporter as qr


def _asset_prices(n_days=756):
    """Asset-only price panel (drop the benchmark column) for the recommend path."""
    return make_synthetic_prices(n_days=n_days)[["AAA", "BBB", "CCC"]]


def test_rolling_core_runs_on_raw_prices():
    from quant_reporter.validation_report import _rolling_oos_sharpe
    prices = make_synthetic_prices(n_days=756)
    cols = ["AAA", "BBB", "CCC"]
    strategies = {"EqualWt": lambda tr, m, c: {t: 1.0 / len(cols) for t in cols}}
    df = _rolling_oos_sharpe(prices, cols, strategies, window_years=1,
                             step_months=3, benchmark_col=None)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "EqualWt Sharpe" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py::test_rolling_core_runs_on_raw_prices -p no:cacheprovider -o addopts="" -q`
Expected: FAIL — `ImportError: cannot import name '_rolling_oos_sharpe'`

(Note: `-o addopts=""` disables the repo's coverage gate so a single-file run doesn't exit non-zero on low total coverage.)

- [ ] **Step 3: Replace `run_rolling_windows` with core + thin wrapper**

In `src/quant_reporter/validation_report.py`, replace the entire `run_rolling_windows` function (lines 64-148) with the following two functions:

```python
def _rolling_oos_sharpe(price_data, asset_cols, strategies, *, window_years=1,
                        step_months=3, risk_free_rate=0.02, benchmark_col=None,
                        denoise_cov=False, n_components=3, return_schedule=False):
    """Shared walk-forward core.

    `strategies` is an ordered dict ``{name -> fn(train_df, mean, cov) -> weights_dict}``.
    For each rolling train/test window: compute optimization inputs once on the
    train slice, build each strategy's weights, apply them to the test slice, and
    record each strategy's realized OOS Sharpe. Returns a DataFrame indexed by
    "Test Period" with columns "{name} Sharpe" (plus the per-strategy weight
    schedule when ``return_schedule=True``). When ``benchmark_col`` is None, each
    strategy's own growth is used as the bundle benchmark (Realized Sharpe is a
    portfolio-only metric, so this does not affect it).
    """
    results = []
    schedule_rows = {name: {} for name in strategies}

    start_date = price_data.index.min()
    end_date = price_data.index.max()
    current_train_start = start_date
    window_days = int(window_years * 365)
    step_days = int(step_months * 30)

    while current_train_start + pd.Timedelta(days=window_days + step_days) <= end_date:
        train_end = current_train_start + pd.Timedelta(days=window_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=step_days)

        train_data = price_data.loc[current_train_start:train_end]
        test_data = price_data.loc[test_start:test_end]
        if train_data.empty or test_data.empty:
            break

        mean_returns, cov_matrix, _ = get_optimization_inputs(
            train_data[asset_cols], denoise_cov=denoise_cov, n_components=n_components)

        try:
            w_dicts = {name: fn(train_data, mean_returns, cov_matrix)
                       for name, fn in strategies.items()}
            for _name, _w in w_dicts.items():
                schedule_rows[_name][test_start] = _w

            window_metrics = {"Test Period":
                              f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"}
            bench_growth = None
            if benchmark_col is not None:
                bench_growth = test_data[benchmark_col] / test_data[benchmark_col].iloc[0]

            for name, w_dict in w_dicts.items():
                strat_growth = get_portfolio_price(test_data[asset_cols], w_dict)
                bench = bench_growth if bench_growth is not None else strat_growth
                bundle = _bundle_from_growth(strat_growth, bench)
                m = compute_metrics(bundle, risk_free_rate)
                window_metrics[f"{name} Sharpe"] = m.get("Realized Sharpe", np.nan)

            results.append(window_metrics)
        except Exception as e:
            logger.debug(f"Rolling optimization failed for window {current_train_start}: {e}")

        current_train_start += pd.Timedelta(days=step_days)

    rolling_df = pd.DataFrame(results).set_index("Test Period") if results else pd.DataFrame()
    if return_schedule:
        schedule = {
            name: pd.DataFrame.from_dict(rows, orient="index").reindex(
                columns=asset_cols).sort_index()
            for name, rows in schedule_rows.items()
        }
        return rolling_df, schedule
    return rolling_df


def run_rolling_windows(ctx: ReportContext, window_years=1, step_months=3,
                        denoise_cov: bool = False, n_components: int = 3,
                        return_schedule: bool = False,
                        objective=None, profile=None):
    """
    Performs Rolling Window Walk-Forward Validation using a fixed window size.
    Returns a dataframe of out-of-sample Sharpe ratios across all periods.

    When ``return_schedule=True``, additionally returns a per-strategy weight
    schedule as ``(rolling_df, target_weight_schedule)`` for the strategies built
    inside the loop: ``Equal Wt``, ``Min Vol``, ``Max Sharpe``, ``User Portfolio``
    (and ``Recommended`` when ``objective`` or ``profile`` is supplied).

    When ``objective`` or ``profile`` is given, an extra ``Recommended`` strategy
    is validated using the profile-constrained optimize (honoring
    ``max_position_weight``, sector caps, exclusions) and the chosen objective.
    With both None, the strategy set is exactly the historical four (unchanged).
    """
    asset_cols = ctx.friendly_tickers
    num_assets = len(asset_cols)
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, ctx.tickers)
    rf = ctx.risk_free_rate

    def _equal(train, mean, cov):
        return {t: 1.0 / num_assets for t in asset_cols}

    def _minvol(train, mean, cov):
        arr = find_optimal_portfolio(objective_min_variance, mean, cov,
                                     bounds_uncon, cons_uncon, rf)
        return {t: w for t, w in zip(asset_cols, arr)}

    def _maxsharpe(train, mean, cov):
        arr = find_optimal_portfolio(objective_neg_sharpe, mean, cov,
                                     bounds_uncon, cons_uncon, rf)
        return {t: w for t, w in zip(asset_cols, arr)}

    def _user(train, mean, cov):
        return ctx.user_friendly_weights

    strategies = {"Equal Wt": _equal, "Min Vol": _minvol,
                  "Max Sharpe": _maxsharpe, "User Portfolio": _user}

    if objective is not None or profile is not None:
        obj = objective or objective_neg_sharpe
        if profile is not None:
            from .planning import apply_constraints
            r_bounds, r_cons = apply_constraints(
                profile, ctx.tickers, sector_map=getattr(ctx, "sector_map", None))
        else:
            r_bounds, r_cons = bounds_uncon, cons_uncon

        def _recommended(train, mean, cov, _obj=obj, _b=r_bounds, _c=r_cons):
            arr = find_optimal_portfolio(_obj, mean, cov, _b, _c, rf)
            return {t: w for t, w in zip(asset_cols, arr)}

        strategies["Recommended"] = _recommended

    return _rolling_oos_sharpe(
        ctx.price_data_full, asset_cols, strategies,
        window_years=window_years, step_months=step_months, risk_free_rate=rf,
        benchmark_col=ctx.friendly_benchmark, denoise_cov=denoise_cov,
        n_components=n_components, return_schedule=return_schedule)
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py::test_rolling_core_runs_on_raw_prices -o addopts="" -q`
Expected: PASS

- [ ] **Step 5: Run the regression gate (existing validation tests + the validation report path)**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_validation_unlock.py -o addopts="" -q`
Expected: PASS (2 passed) — proves the refactor is behavior-identical.

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/validation_report.py test/test_recommendation_validation.py
git commit -m "$(cat <<'EOF'
refactor(validation): extract _rolling_oos_sharpe core from run_rolling_windows

Behavior-preserving: run_rolling_windows becomes a thin wrapper over a shared
strategy-driven core. Existing test_validation_unlock.py is the regression gate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Optional `objective`/`profile` → "Recommended" strategy

**Files:**
- Modify: (the optional branch added in Task 1 is the implementation; this task adds its tests)
- Test: `test/test_recommendation_validation.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_recommendation_validation.py`:

```python
from quant_reporter.validation_report import run_rolling_windows


def _ctx():
    prices = make_synthetic_prices(n_days=756)
    return qr.build_context_from_prices(
        prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK", "2021-01-01", "2022-06-30")


def test_run_rolling_windows_no_profile_unchanged_columns():
    df = run_rolling_windows(_ctx())
    assert "Recommended Sharpe" not in df.columns
    assert "Max Sharpe Sharpe" in df.columns


def test_run_rolling_windows_recommended_column_with_profile():
    prof = qr.build_profile(max_position_weight=0.5)
    df, schedule = run_rolling_windows(_ctx(), return_schedule=True, profile=prof)
    assert "Recommended Sharpe" in df.columns
    assert "Recommended" in schedule
    rec = schedule["Recommended"].dropna(how="all")
    if not rec.empty:
        # every window's recommended weights respect the profile cap
        assert (rec.fillna(0.0).values <= 0.5 + 1e-6).all()
```

- [ ] **Step 2: Run tests to verify they pass (implementation already added in Task 1)**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py -k "run_rolling_windows" -o addopts="" -q`
Expected: PASS (2 passed)

(If `test_run_rolling_windows_recommended_column_with_profile` fails because the optional branch is missing, add the `if objective is not None or profile is not None:` block from Task 1 Step 3 to `run_rolling_windows`.)

- [ ] **Step 3: Commit**

```bash
git add test/test_recommendation_validation.py
git commit -m "$(cat <<'EOF'
test(validation): cover optional profile-constrained Recommended walk-forward column

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `RecommendationValidation` + `walk_forward_recommendation` + `recommend(validate=…)`

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (imports near top; add dataclass + helper; extend `Recommendation` and `recommend`)
- Test: `test/test_recommendation_validation.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_recommendation_validation.py`:

```python
def test_recommend_validate_attaches_validation():
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    rec = qr.recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                       profile=prof, validate=True)
    assert rec.validation is not None
    assert rec.validation.n_windows >= 1
    assert rec.validation.baseline_oos_sharpe is not None
    assert rec.validation.verdict in ("holds up", "fragile (overfit)", "inconclusive")


def test_recommend_validate_false_leaves_validation_none():
    prices = _asset_prices()
    rec = qr.recommend(prices, profile=qr.build_profile(max_position_weight=0.6))
    assert rec.validation is None


def test_walk_forward_inconclusive_on_short_data():
    from quant_reporter.recommendation import walk_forward_recommendation
    prices = _asset_prices().iloc[:40]
    v = walk_forward_recommendation(prices, profile=qr.build_profile(max_position_weight=0.6))
    assert v.verdict == "inconclusive"
    assert v.n_windows == 0


def test_verdict_threshold_controls_outcome():
    from quant_reporter.recommendation import walk_forward_recommendation
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    strict = walk_forward_recommendation(prices, profile=prof, max_degradation=-1.0)
    assert strict.verdict == "fragile (overfit)"   # degradation >= 0 can never be <= -1
    lenient = walk_forward_recommendation(prices, profile=prof, max_degradation=10.0)
    if lenient.oos_sharpe > 0:
        assert lenient.verdict == "holds up"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py -k "recommend_validate or walk_forward_inconclusive or verdict_threshold" -o addopts="" -q`
Expected: FAIL — `TypeError: recommend() got an unexpected keyword argument 'validate'`

- [ ] **Step 3a: Add the analytics import**

In `src/quant_reporter/recommendation.py`, after the line `from .metrics import compute_drawdown` (line 25), add:

```python
from .analytics import ReturnsBundle, compute_metrics
```

- [ ] **Step 3b: Add `RecommendationValidation` dataclass**

In `src/quant_reporter/recommendation.py`, immediately before the `@dataclass`/`class Recommendation:` definition (line 286), add:

```python
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
```

- [ ] **Step 3c: Add the `validation` field to `Recommendation`**

In the `Recommendation` dataclass (currently fields at lines 287-291), add a fifth field after `verdict`:

```python
@dataclass
class Recommendation:
    target_weights: RecommendedWeights
    trades: Optional[RebalancePlan]
    alerts: list              # list[RiskAlert]
    verdict: Optional[StrategyVerdict]
    suitability: Optional[SuitabilityReport] = None
    validation: Optional[RecommendationValidation] = None
```

- [ ] **Step 3d: Add `validate=` to `recommend` and attach the result**

In `recommend(...)`, add the new parameters and the attach logic. Change the signature to include `validate=False, window_years=1, step_months=3, max_degradation=0.5` (place them after `profile=None`), and at the end replace the final `return Recommendation(...)` with:

```python
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
```

- [ ] **Step 3e: Add `_realized_metrics` + `walk_forward_recommendation`**

At the end of `src/quant_reporter/recommendation.py`, append:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py -k "recommend_validate or walk_forward_inconclusive or verdict_threshold" -o addopts="" -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_recommendation_validation.py
git commit -m "$(cat <<'EOF'
feat(recommendation): walk_forward_recommendation + recommend(validate=)

Reuses the shared _rolling_oos_sharpe core + calculate_overfitting_score to
attach a degradation-based RecommendationValidation. Backward compatible.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rendering + exports + full suite

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (`Recommendation.to_dict`, `Recommendation.to_text`)
- Modify: `src/quant_reporter/__init__.py` (recommendation export block, ~lines 222-227)
- Test: `test/test_recommendation_validation.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_recommendation_validation.py`:

```python
def test_validation_renders_in_text_and_dict():
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    rec = qr.recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                       profile=prof, validate=True)
    assert "Validation (walk-forward)" in rec.to_text()
    d = rec.to_dict()
    assert d["validation"] is not None
    assert "oos_sharpe" in d["validation"]
    # legacy path: key present, value None
    assert qr.recommend(prices, profile=prof).to_dict()["validation"] is None


def test_validation_public_surface_exported():
    for name in ("RecommendationValidation", "walk_forward_recommendation"):
        assert hasattr(qr, name), f"{name} not exported from quant_reporter"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py -k "renders_in_text or public_surface" -o addopts="" -q`
Expected: FAIL — `KeyError: 'validation'` (to_dict) / `AssertionError` (export)

- [ ] **Step 3a: Render `validation` in `Recommendation.to_dict`**

In `Recommendation.to_dict`, add a `"validation"` entry after the `"suitability"` entry:

```python
            "validation": None if self.validation is None else self.validation.to_dict(),
```

- [ ] **Step 3b: Render `validation` in `Recommendation.to_text`**

In `Recommendation.to_text`, immediately before the final `return "\n".join(lines)`, add:

```python
        if self.validation is not None:
            lines += ["", self.validation.to_text()]
```

- [ ] **Step 3c: Export the new public surface**

In `src/quant_reporter/__init__.py`, extend the recommendation import block (the `from .recommendation import (...)` near lines 222-226) to also import the two new names — change it to:

```python
from .recommendation import (
    recommend, recommend_weights, rebalance_trades, risk_alerts, compare_verdict,
    Recommendation, RecommendedWeights, RebalancePlan, Trade, RiskAlert, StrategyVerdict,
    RecommendationValidation, walk_forward_recommendation,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && PYTHONPATH=src python3 -m pytest test/test_recommendation_validation.py -o addopts="" -q`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run the full suite + lint**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python3 -m pytest test/ -q && ruff check src/`
Expected: All tests pass (prior suite + the new validation tests), coverage gate ≥ 80% satisfied, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/recommendation.py src/quant_reporter/__init__.py test/test_recommendation_validation.py
git commit -m "$(cat <<'EOF'
feat(recommendation): render + export RecommendationValidation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes (for the implementer)

- **Regression gate:** Task 1 must keep `test/test_validation_unlock.py` green — that is the proof the `run_rolling_windows` refactor is behavior-identical. Do not edit that test.
- **Import direction:** `recommendation.py` imports `_rolling_oos_sharpe`/`calculate_overfitting_score` from `validation_report.py` *inside* `walk_forward_recommendation` (lazy), and `validation_report.py` imports `apply_constraints` from `planning.py` *inside* `run_rolling_windows` (lazy). Neither `validation_report` nor `planning` imports `recommendation`, so there is no cycle; the lazy imports mirror the existing `to_html` pattern and keep module-load light.
- **Coverage gate:** the repo's `setup.cfg`/`pyproject` adds a `--cov-fail-under` to `addopts`, so a *single-file* pytest run exits non-zero on low total coverage even when tests pass. Use `-o addopts=""` for single-file runs (Tasks 1-4 steps); the final full-suite run in Task 4 Step 5 keeps the gate on.
- **Feasibility:** `apply_constraints` raises if `n_effective * max_position_weight < invested_budget`. The 3-asset synthetic panel needs `max_position_weight ≥ 0.34`; the tests use `0.5`/`0.6`, which are safe.
- **Verdict determinism:** `test_verdict_threshold_controls_outcome` pins the gate logic without depending on the data's actual degradation — `max_degradation=-1.0` can never pass (degradation ≥ 0), and the lenient branch is only asserted when `oos_sharpe > 0`.
```
