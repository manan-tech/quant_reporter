# Design: profile-aware walk-forward validation of recommendations (Option 2)

**Date:** 2026-06-02
**Status:** Approved — ready for implementation planning
**Component:** `quant_reporter` decision-support layer (validation)

## Context

The recommendation layer's *strategy comparison* (`compare_verdict`) is already
honest — it ranks by Deflated Sharpe. But the **recommended target weights**
(`recommend_weights`) come from a single in-sample optimization with no
out-of-sample check. We want walk-forward validation of the recommendation so
users can see whether the advice would have held up OOS.

`quant_reporter` **already has** walk-forward + OOS reporting:
`run_rolling_windows()` (returns OOS Sharpe across periods for Equal Wt / Min Vol
/ Max Sharpe / User Portfolio) and `create_validation_report()`. The **only gap**
is that `run_rolling_windows` optimizes with **unconstrained** bounds — it ignores
the Profile's `max_position_weight`, sector caps, exclusions, and the chosen
objective. So it validates a *generic* optimizer, not the *constrained advice*.

**Decision (Option 2):** do NOT build a parallel walk-forward. Extend the existing
engine so the same machinery can validate the profile-constrained recommendation,
and reuse it from `recommend()`. Existing behavior, the existing report, and
existing tests stay unchanged.

## Goals / non-goals

- **Goal:** profile-aware OOS validation reusing one walk-forward engine.
- **Goal:** `recommend(validate=True)` attaches a `RecommendationValidation`.
- **Goal:** existing `run_rolling_windows` output and `create_validation_report`
  remain byte-identical when no profile/objective is supplied.
- **Non-goal (deferred):** PSR/DSR significance on the OOS series (verdict is
  degradation-only for now); surfacing the new "Recommended" column inside
  `create_validation_report` (opt-in follow-up); stochastic / regime / denoising
  estimation upgrades.

## Section 1 — Extract the shared stepping core (DRY, behavior-preserving)

Pull the per-window train/test loop + OOS-Sharpe computation out of
`run_rolling_windows` into a private helper in `validation_report.py`:

```python
_rolling_oos_sharpe(price_data, asset_cols, strategies, *, window_years=1,
                    step_months=3, risk_free_rate=0.02, benchmark_col=None,
                    denoise_cov=False, n_components=3, return_schedule=False)
```

- `strategies`: an **ordered** `dict[name -> fn(train_df) -> weights_dict]`. The
  callable receives the train-slice price DataFrame (asset columns only) and
  returns a `{ticker: weight}` dict.
- Per window: slice train/test by the existing date arithmetic (`window_years`,
  `step_months`), call each strategy's `fn(train_slice)`, apply the weights to the
  test slice via `get_portfolio_price`, build a `ReturnsBundle`, and record each
  strategy's `Realized Sharpe` (via `compute_metrics`).
- `benchmark_col`: when provided (the ctx path), used for the benchmark growth in
  the bundle; when `None` (the recommend path), the bundle uses the portfolio's
  own growth as both series (benchmark-relative fields are unused — we only read
  `Realized Sharpe`).
- Returns the same `rolling_df` (index = "Test Period", columns =
  `"{name} Sharpe"`), optionally with the per-strategy weight schedule, exactly as
  today.

`run_rolling_windows` becomes a thin wrapper: it builds its existing four strategy
callables (`Equal Wt`, `Min Vol`, `Max Sharpe`, `User Portfolio`) with the current
**unconstrained** bounds and calls `_rolling_oos_sharpe`. **Output is identical**,
so `create_validation_report` and `test/test_validation_unlock.py` are the
regression gate and must stay green verbatim.

## Section 2 — Extend `run_rolling_windows` (additive)

```python
def run_rolling_windows(ctx, window_years=1, step_months=3, denoise_cov=False,
                        n_components=3, return_schedule=False,
                        objective=None, profile=None):
```

- When `objective` or `profile` is provided, append one more strategy named
  **"Recommended"** whose per-window weights come from a profile-constrained
  optimize: `bounds, constraints = apply_constraints(profile, ctx.tickers,
  sector_map=getattr(ctx, "sector_map", None))` (when `profile` is `None`, fall
  back to unconstrained bounds + `build_constraints`), objective =
  `objective or objective_neg_sharpe`, then `find_optimal_portfolio`.
- When both are `None`, the strategy set is exactly today's four — no new column.
- The report renders `rolling_df` generically (`to_html`), so an opted-in profile
  transparently adds a "Recommended Sharpe" column without layout changes.

## Section 3 — `recommend(validate=True)` + `RecommendationValidation`

`recommend()` operates on raw `prices` (asset columns only, no ctx/benchmark), so
it calls `_rolling_oos_sharpe` **directly** with `benchmark_col=None` and two
strategies:
- `"Recommended"` — constrained optimize using the recommendation's `objective` +
  `profile` (the actual advice each window).
- `"Current"` — the user's `current_weights` held fixed (the OOS baseline), only
  when `current_weights` is provided.

From the resulting `rolling_df`:
- `oos_sharpe` = mean of the "Recommended Sharpe" column across windows.
- `baseline_oos_sharpe` = mean of the "Current Sharpe" column (or `None`).
- `in_sample_sharpe` = realized Sharpe of the **final** recommended weights over
  the whole `prices` window (apples-to-apples; both realized).
- `degradation` = the "Overfitting Score" from the existing
  `calculate_overfitting_score({"Recommended": is_metrics},
  {"Recommended": oos_metrics})` helper — i.e. `max(0, (IS−OOS)/IS)`.
- **verdict:**
  - `inconclusive` if `n_windows < 1` (returned cleanly, never raises);
  - `holds up` if `oos_sharpe > 0` **and** `degradation ≤ max_degradation`
    (default `0.5`);
  - `fragile (overfit)` otherwise.

```python
@dataclass
class RecommendationValidation:
    in_sample_sharpe: float
    oos_sharpe: float
    degradation: float
    n_windows: int
    verdict: str                       # 'holds up' | 'fragile (overfit)' | 'inconclusive'
    rationale: str
    baseline_oos_sharpe: Optional[float]
    per_window: list                   # [{'period','recommended_sharpe','current_sharpe'}, ...]
    evidence: dict

    def to_dict(self): ...
    def to_text(self): ...
```

Wiring:
- `recommend(prices, *, validate=False, window_years=1, step_months=3,
  max_degradation=0.5, …)` — when `validate=True`, run the walk-forward with the
  **same** `objective`/`profile`/`current_weights`, build the
  `RecommendationValidation`, attach it.
- **Import direction:** `validation_report.py` does **not** import
  `recommendation.py` (it imports `report_context`, `metrics`, `analytics`,
  `opt_core`, `opt_plotting`, `html_builder`). So `recommend()` can call
  `_rolling_oos_sharpe` via a **lazy import inside the function** — matching the
  existing `to_html` lazy-import pattern and keeping the dependency one-directional
  with zero cycle risk. The per-window constrained weight function it passes uses
  `apply_constraints` (already imported by `recommendation` for the Profile work).
- `Recommendation` gains `validation: Optional[RecommendationValidation] = None`,
  rendered in `to_text`, `to_dict`, and `to_html`.
- **Backward compatible:** `validate=False` default; field defaults `None`;
  `validate=False` reproduces today's behavior exactly.

## Section 4 — Files & testing

| File | Change |
|---|---|
| `validation_report.py` | extract `_rolling_oos_sharpe`; `run_rolling_windows` thin wrapper + optional `objective`/`profile`; reuse `calculate_overfitting_score` |
| `recommendation.py` | `recommend(validate=…)`; `Recommendation.validation` field + rendering; `RecommendationValidation` dataclass (or a small `recommendation_validation.py` if cycle pressure) |
| `__init__.py` | export `RecommendationValidation` |
| `test/test_recommendation_validation.py` | **NEW** |
| `test/test_validation_unlock.py` | unchanged — regression gate, must stay green |

### Tests
- **Regression:** legacy `run_rolling_windows(ctx)` (no profile) returns the same
  columns/shape as before; `test_validation_unlock.py` passes verbatim.
- **Faithfulness:** with a `profile` whose `max_position_weight` caps positions,
  every window's "Recommended" weights respect the cap.
- **Core on raw prices:** `_rolling_oos_sharpe(prices, cols, {...},
  benchmark_col=None)` produces a non-empty `rolling_df` with the expected columns
  on synthetic data.
- **Verdict:** synthetic robust series → `holds up`; synthetic overfit series
  (strong IS, weak OOS) → `fragile (overfit)`; `< 1` window → `inconclusive` with
  no exception.
- **Wiring:** `recommend(prices, profile=…, current_weights=…, validate=True)`
  attaches a `RecommendationValidation` with a `baseline_oos_sharpe`;
  `validate=False` leaves `.validation` `None` (backward-compat).
- **Rendering:** `to_text`/`to_dict` include the validation block when present.

## Source material
- Reuses the project's existing walk-forward (`run_rolling_windows`,
  `calculate_overfitting_score`) and the `apply_constraints` primitive from the
  Profile/IPS keystone spec (`2026-06-02-planning-ips-keystone-design.md`).
- CFA L2 Vol 5 — "Backtesting and Simulation" (walk-forward, OOS discipline).
