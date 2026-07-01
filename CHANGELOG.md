# Changelog

All notable changes to `quant_reporter` are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Backtest-honesty / overfitting diagnostics (`overfitting.py`).** A new module that
  flags when a backtest is likely a fluke: `probability_of_backtest_overfitting` (PBO via
  Combinatorially Symmetric Cross-Validation), `min_track_record_length` (MinTRL),
  `min_backtest_length` (MinBTL), and a one-call `assess_overfitting` verdict
  (`robust` / `caution` / `likely_overfit` / `inconclusive`) that bundles PBO with the
  deflated Sharpe ratio and MinTRL. `overfitting_section` renders the verdict as an
  embeddable report section. Consume-only, offline, no new dependencies. Refs: Bailey &
  López de Prado, *Deflated Sharpe Ratio* / *Probability of Backtest Overfitting*.
- **No-forecast allocation switch on the recommendation path** (GH #7). `recommend_weights`,
  `recommend`, and `walk_forward_recommendation` accept an opt-in `method=` — `"optimize"` (default,
  objective-based, uses expected returns) or the no-forecast allocators `"min_variance"`,
  `"risk_parity"`, and `"max_diversification"` that allocate from the covariance alone (the honest
  "we don't forecast returns" stance for the dominant source of optimizer error). The default
  preserves prior behavior; the chosen method and a `uses_return_forecast` flag are recorded in the
  recommendation `evidence`, and the rationale states the no-forecast assumption plainly.
  `risk_parity`/`max_diversification` use their own risk-based allocation and do not honor
  `bounds`/`constraints`/`profile` (only `optimize`/`min_variance` do).
- **Covariance-estimator switch on the recommendation path** (GH #6). `recommend_weights`,
  `recommend`, and `walk_forward_recommendation` accept an opt-in `cov_method=` — `"sample"`
  (default), `"ledoit_wolf"` (Ledoit-Wolf constant-correlation shrinkage), or `"denoise"`
  (eigenvalue-clipping), with `n_components` tuning the denoise cutoff — wiring the already-shipped
  robust estimators into the recommendation. The default preserves prior behavior byte-for-byte
  (SemVer-safe); the estimator and its diagnostics (e.g. Ledoit-Wolf shrinkage) are recorded in the
  recommendation `evidence`. Expected returns are unchanged (the raw annualized mean); the return
  estimate is addressed separately by GH #7/#8.
- **Explicit risk-free-rate failure signal** (GH #22). `YFinanceProvider` now exposes
  `fetch_risk_free_rate()`, which raises the new `RiskFreeRateUnavailable` when the live
  T-bill lookup fails, and the exception is exported from the package top level. Custom
  providers may raise it to opt into the same explicit fallback flagging.

### Changed
- **Unified Sortino definition.** The realized-metrics block (`analytics.compute_metrics`,
  which powers the portfolio/combined reports) now uses the canonical `metrics.sortino`
  (semi-deviation downside, MAR = risk-free rate) instead of a separate legacy helper that
  used a 0% threshold. Every surface now reports one Sortino definition; reported Sortino
  values may shift slightly as a result.

### Fixed
- **Risk-free-rate fallback detection is now explicit instead of heuristic** (GH #22).
  The "Data Quality Notes" banner previously inferred the 2% fallback by comparing the
  returned rate to the default, so a genuine live rate of ~2.00% was wrongly flagged as a
  fallback. The report layer now keys off an actual failed fetch (`fetch_risk_free_rate`
  raising `RiskFreeRateUnavailable`), so the flag is set iff the live lookup really failed.
  `get_risk_free_rate()` is unchanged (still returns the 2% default on failure), so existing
  callers are unaffected.
- HTML reports now HTML-escape section/item titles and descriptions, so a `&`/`<` in a
  ticker display name or label no longer breaks the markup (`html_builder.py`).

### Removed
- Dropped the unused legacy `calculate_sortino_ratio` helper (was internal, not part of the
  public API) now that the analytics core uses the canonical `sortino`.

## [2.2.0] - 2026-06-03

### Added
- **Decision-support planning layer (`planning.py`)** — a CFA-grounded investor
  profile as a reusable primitive: `Profile` (risk tolerance presets + return
  objective + TTLU constraints), `build_profile`, `combine_risk_tolerance`,
  `apply_constraints` (Profile → optimizer bounds/constraints), and
  `check_suitability` → `SuitabilityReport`. `recommend()`/`recommend_weights()`
  accept an optional `profile=` that constrains the optimizer **and** sets the
  alert thresholds — closing a seam where the limits previously drove only alerts,
  not the recommended weights.
- **Walk-forward validation of recommendations** — `recommend(validate=True)`
  attaches a `RecommendationValidation` (out-of-sample vs in-sample Sharpe,
  degradation, a holds-up/fragile/inconclusive verdict, and the current portfolio
  as an OOS baseline), rendered in `to_text`/`to_dict`/`to_html`. Implemented by
  extracting a shared `_rolling_oos_sharpe` core from `run_rolling_windows`
  (behavior-preserving) and a `walk_forward_recommendation` helper; the existing
  validation report is unchanged.

### Changed
- Maintainer contact email updated to the project's personal address
  (`mananpbansal@gmail.com`) in package metadata (`pyproject.toml`) and `SECURITY.md`.

## [2.1.0] - 2026-06-01

The 2.x line grows from a reporting layer into a full quantitative toolkit: a
pluggable data layer, a cost-aware backtest engine, position-sizing/risk overlays,
tactical signals, factor tilts, a per-asset info layer, a prebuilt-strategy library
with a unified runner, and an opt-in recommendation layer — plus packaging and CI
hardening that took the project from a local 1.1.1 to a published 2.1.0 on PyPI.

### Added

#### Data layer
- **`DataProvider` protocol** (`providers.py`): swap the data source (Bloomberg, CSV,
  database, fixtures) for the default `YFinanceProvider` via `get_default_provider` /
  `set_default_provider`, or pass `data_provider=` per report. Enables fully offline
  and reproducible runs.
- `build_context_from_prices()` — build a `ReportContext` from an already-fetched price
  DataFrame (no network); enables offline use and testing.

#### Analytics core (single source of truth)
- Analytics core (`analytics.py`): `portfolio_returns`/`ReturnsBundle`, `compute_metrics`
  (numeric) + `format_metrics`, `PortfolioAnalytics`, `compute_drawdown`/`DrawdownResult`,
  and the memoized `ctx.analytics` accessor — the single source of truth for portfolio
  returns, growth, drawdown, and realized metrics.
- `rebalance_freq` is now honored end-to-end (it was previously accepted but ignored):
  portfolio Growth-of-$1 routes through the rebalancing engine; `None` = buy-and-hold
  (default, unchanged).

#### Backtest engine
- SP1a primitives (`signals.py`, `robust_estimators.py`, `backtest.py`):
  `compute_trailing_volatility`, `volatility_target_positions`, `ledoit_wolf_covariance`,
  `portfolio_turnover`, `drawdown_stats` — pure, look-ahead-safe (no new dependencies).
- SP1b cost-aware engine: `simulate_strategy` (dict or dated-schedule weights; frictionless
  buy-and-hold matches `simulate_rebalanced_portfolio`), `transaction_cost_model` (commission
  + half-spread; `impact_model` hook reserved — market impact is future work),
  `generate_rebalance_dates`, `run_rolling_windows(return_schedule=True)` weight-schedule
  unlock, and `performance_stats.py` (`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`,
  `compare_strategies_oos`). Flagship: `examples/example_walk_forward_backtest.py`.

#### Position sizing & risk overlays (`sizing.py`, `opt_core.py`)
- Sizing: `forecast_portfolio_vol`, `target_volatility_scalar`, `inverse_volatility_weights`,
  `realized_tracking_error`, `kelly_fraction`, `cppi_weights`.
- Advanced risk decomposition: `risk_contributions`, `optimize_risk_budget`, `portfolio_cvar`.

#### Tactical signals (`signals.py`)
- `time_series_momentum_signal`, `moving_average_crossover_signal`,
  `cross_sectional_momentum_score`, `zscore_reversion_signal`.

#### Factor tilts (`factor_tilts.py`)
- `characteristic_tilt_weights`, `factor_neutralize_returns`, `resample_portfolio`.

#### Per-asset info layer (`asset_info.py`)
- `compute_asset_analytics`, `compute_asset_factor_exposures`, `get_asset_fundamentals`,
  `narrate_asset`, `build_asset_info_table` — drill from portfolio to individual holding.

#### Strategy library & runner
- Shared metrics library (`metrics.py`): `cagr`, `annual_volatility`, `sharpe`, `sortino`,
  `calmar`, `omega`, `max_drawdown`, `avg_drawdown`, `ulcer_index`, `value_at_risk`,
  `conditional_var`, `downside_deviation`, `tracking_error`, `information_ratio`, `hit_rate`,
  `win_loss_ratio`, `tail_ratio`, `skewness`, `kurtosis`, `summary_metrics`.
- Objectives / loss surface (`objectives.py`): `neg_sharpe`, `variance`, `cvar_objective`,
  `tracking_error_objective`, `mean_squared_error`, `mean_absolute_error`.
- Prebuilt strategies (`strategies.py`): `equal_weight`, `inverse_vol`, `min_variance`,
  `risk_parity`, `max_sharpe`, `trend_following`, `cross_sectional_momentum`,
  `vol_target_overlay`, and a `REGISTRY` for lookup by name.
- Unified runner (`strategy.py`): `Strategy`, `backtest`, `backtest_many`, `BacktestResult`,
  plus `create_backtest_report`.

#### Recommendation layer — opt-in opinions (`recommendation.py`)
- `recommend`, `recommend_weights`, `rebalance_trades`, `risk_alerts`, `compare_verdict`,
  with `Recommendation`, `RecommendedWeights`, `RebalancePlan`, `Trade`, `RiskAlert`,
  `StrategyVerdict` result types, and `create_recommendation_report`.

#### Packaging, CI & docs
- Published to **PyPI** as `quant-reporter` 2.1.0; version unified across the package.
- `py.typed` marker (PEP 561), MIT license metadata, pinned dependencies, and a serious
  README with disclaimers and badges.
- GitHub Actions CI matrix across Python **3.10–3.12** (ruff + pytest).
- **Example gallery** (`examples/gallery/generate_gallery.py`) with portfolios and sector
  maps, published to GitHub Pages: https://manan-tech.github.io/quant_reporter/.

### Changed
- **Breaking:** removed the string-returning `calculate_metrics`; use `compute_metrics` (numeric) + `format_metrics` for display.
- **Reports are now pure assemblers over `ctx.analytics`** — every report reads the same compute-once
  values. This fixes the dual-basis inconsistency: realized metrics (simple-return path) and the
  optimizer's expected/model metrics (log moments) are now clearly distinguished ("Realized …" vs
  "Expected …"), and drawdown/Sharpe/VaR are no longer recomputed divergently across sections.
- Monte Carlo: the duplicated engine was merged into one **seeded, reproducible** engine; simulated
  tail risk is labeled "Horizon VaR/CVaR (simulated)" vs daily historical VaR/CVaR.
- Factor attribution: portfolio is regressed on the canonical full-period returns via a single
  excess-return OLS engine (static + rolling unified, risk-free rate threaded; rolling uses the
  3-factor core). Brinson is now **honestly labeled** — "vs Equal-Weight Baseline" unless real
  benchmark sector weights are supplied (via `ctx.benchmark_weights`).
- Combined report is **fail-loud**: a failing module renders a visible error section (deterministic
  output) with an optional `strict=True` to re-raise; Monte Carlo parameters are forwarded; the
  duplicated correlation heatmap is de-duplicated.
- Walk-forward validation now uses the same covariance treatment (`denoise_cov`/`n_components`) as the
  in-sample optimization, and computes metrics numerically (no string round-trip).
- Risk-free-rate fetch failure now falls back to **0.02** (was 0.06) via `DEFAULT_RISK_FREE_RATE`,
  matching `build_context`'s default — one documented fallback.
- `plot_rolling_sharpe` de-collision: the single-portfolio variant keeps its name; the
  multi-strategy variant is now exported as `plot_rolling_sharpe_comparison` (it no longer
  shadows the single-portfolio export).

### Fixed
- **Benchmark that is also a holding** (e.g. a 60/40 of SPY+AGG benchmarked against SPY —
  common) is now handled in two places: `report_context._assemble_context` no longer
  duplicates the benchmark column (which inflated the asset count and caused `(2,)/(3,)`
  shape errors), and `portfolio_report`'s constituent-growth concat no longer re-adds the
  benchmark (which raised a Plotly `DuplicateError`). The benchmark is appended only when it
  isn't already a holding.
- **`create_combined_report` now forwards `data_provider`** — it has an explicit signature
  (not `**kwargs`) and previously dropped the provider, so the flagship report couldn't run
  offline.
- Self-regression residual test made **BLAS-independent**: it asserts residual *magnitude*
  relative to the input instead of the correlation of machine-epsilon residuals (numerically
  meaningless; passed on macOS Accelerate but failed on Linux OpenBLAS in CI).

## [2.0.0]

A rearchitecture of the reporting layer around a shared `ReportContext`, rebuilt on top
of the stable 1.1.1 release. **Breaking:** every report generator now takes the portfolio,
benchmark, and training window and fetches data itself (see "Migrating from 1.x" in the
README).

### Added
- `ReportContext` + `build_context()` — fetch price data once, derive train/test splits
  and optimization inputs, and share them across reports.
- Dedicated ctx-based generators: `create_portfolio_report`, `create_optimization_report`,
  `create_validation_report`, `create_factor_report`, `create_monte_carlo_report`, and a
  rewritten `create_combined_report` orchestrating all five with graceful per-module
  degradation.
- Factor report: Fama-French 5/3-factor regression, rolling exposures, style drift, macro
  regimes, and Brinson-Fachler attribution.
- Monte Carlo report: time-to-target and day-1 stress scenarios alongside GBM paths.
- Test suite for the new API and the bug fixes; a `[test]` optional-dependency extra.

### Changed
- `compute_brinson_attribution` now takes a single `asset_returns` matrix plus
  portfolio/benchmark weight dicts and a `sector_map` (was: separate return series).
- `create_full_report` is retained as an alias for `create_portfolio_report`.
- Require Python ≥ 3.9 and `pandas >= 2.2` (for the `'ME'`/`'YE'` resample aliases).
- README rewritten for the 2.0 API; example scripts consolidated onto the new API.

### Fixed
- **HRP optimizer** computed `sqrt((1 - rho)/2)` where floating-point error could make
  `rho > 1`, yielding NaN distances and corrupting the clustering. Correlations are now
  clamped to `[-1, 1]`.
- **Factor models** called `pd.infer_freq` unconditionally, raising `TypeError` on a
  non-`DatetimeIndex`. It is now guarded and falls back to daily annualization.
- **`get_risk_free_rate`** silently returned the 0.06 default on modern yfinance because
  `tbill['Close']` is a multi-indexed DataFrame; it now reads the live rate correctly.
- **Validation overfitting score** looked up the wrong metric keys, so the Overfitting
  Score and Strategy Degradation were always 0. They now compute correctly.
- **Factor report** double-annualized alpha in the regression summary (the value from
  `run_factor_regression` is already annualized).
- Duplicate `create_monte_carlo_report` definitions no longer collide: the public symbol
  is the ctx-based generator.
- Repository hygiene: removed committed `__pycache__`/`.DS_Store` artifacts and hardened
  `.gitignore`.

## [1.1.1]

Last release of the 1.x line: portfolio metrics, MPT optimizers (min-vol, max-Sharpe,
sector-constrained), advanced optimizers (Risk Parity, HRP, Min-Correlation,
Max-Diversification), Black-Litterman, Monte Carlo simulation, walk-forward validation,
and Plotly HTML reports.
