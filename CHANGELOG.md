# Changelog

All notable changes to `quant_reporter` are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [2.1.0] - Unreleased

### Added
- Analytics core (`analytics.py`): `portfolio_returns`/`ReturnsBundle`, `compute_metrics` (numeric)
  + `format_metrics`, `compute_drawdown`/`DrawdownResult`, and the memoized `ctx.analytics` accessor —
  the single source of truth for portfolio returns, growth, drawdown, and realized metrics.
- `build_context_from_prices()` — build a `ReportContext` from an already-fetched price DataFrame
  (no network); enables offline use and testing.
- `rebalance_freq` is now honored end-to-end (it was previously accepted but ignored): portfolio
  Growth-of-$1 routes through the rebalancing engine; `None` = buy-and-hold (default, unchanged).
- SP1a foundations (`signals.py`, `robust_estimators.py`, `backtest.py`): `compute_trailing_volatility`,
  `volatility_target_positions`, `ledoit_wolf_covariance`, `portfolio_turnover`, `drawdown_stats` —
  pure, look-ahead-safe Phase-1 primitives (no new dependencies).
- SP1b cost-aware backtest engine: `simulate_strategy` (dict or dated-schedule weights; frictionless
  buy-and-hold matches `simulate_rebalanced_portfolio`), `transaction_cost_model` (commission + half-spread;
  `impact_model` hook reserved — market impact is future work), `generate_rebalance_dates`,
  `run_rolling_windows(return_schedule=True)` weight-schedule unlock, and `performance_stats.py`
  (`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `compare_strategies_oos`). Flagship:
  `examples/example_walk_forward_backtest.py`.

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
