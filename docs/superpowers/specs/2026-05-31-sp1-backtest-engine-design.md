# SP1 — Cost-Aware Backtest Engine & Phase-1 Primitives (design spec)

**Date:** 2026-05-31 · **Branch:** `v2.1` · **Parent:** `2026-05-31-v2.1-roadmap.md`
**Status:** Approved design — ready for implementation plan.
**Maps to:** item 4 (quant strategy expansion), the backtest/execution + foundation slice.
**Depends on:** SP0 (analytics core `portfolio_returns`/`compute_drawdown`/`ctx.analytics`, offline `build_context_from_prices`).

---

## 1. Goal

Turn the library from "optimize + one-shot simulate" into a **walk-forward, cost-aware, honestly-evaluated backtest engine**, plus the Phase-1 primitives everything else composes through. All primitives are pure, transparent, offline-testable functions — **no new dependencies**, opinions stay out (the `recommend=True` layer is SP4).

## 2. Non-goals

Risk overlays beyond vol-targeting (SP2), tactical signals beyond the vol estimator (SP2), factor tilts (SP2), the recommendation layer (SP4), tax lots (later), and **market-impact cost modeling** (reserved as a future TODO via the `impact_model` hook — see §5).

## 3. Module layout (flat, re-exported from `__init__`)

| Module | New/Extend | Holds |
|--------|-----------|-------|
| **`signals.py`** *(new)* | new | `compute_trailing_volatility`, `volatility_target_positions` (the vol estimator the whole tactical/overlay family will size through) |
| **`robust_estimators.py`** *(new)* | new | `ledoit_wolf_covariance` (hardens every optimizer's cov input) |
| **`backtest.py`** *(new)* | new | `portfolio_turnover`, `transaction_cost_model`, `generate_rebalance_dates`, `drawdown_stats`, `simulate_strategy` (the hub) |
| **`performance_stats.py`** *(new)* | new | `probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, `compare_strategies_oos` |
| `validation_report.py` | extend | `run_rolling_windows` also returns a `target_weight_schedule` (the unlock) |

## 4. Signatures

```python
# signals.py
def compute_trailing_volatility(returns, lookback=63, method='ewma', annualize=True) -> pd.DataFrame
def volatility_target_positions(signal, returns, target_vol=0.10, vol_lookback=63,
                                method='ewma', max_leverage=2.0, scaling='per_asset', cov=None) -> pd.DataFrame

# robust_estimators.py  (sklearn.covariance is already a dependency)
def ledoit_wolf_covariance(returns, target='constant_correlation', periods_per_year=252, delta=None) -> dict
        # {'cov_matrix' (annualized PD DataFrame, same contract as get_optimization_inputs),
        #  'shrinkage' (float in [0,1]), 'target', 'sample_cov', 'target_matrix'}

# backtest.py
def portfolio_turnover(weights_before, weights_after, convention='one_way') -> dict
        # {'turnover','buys','sells','trades' (per-asset signed deltas)}
def transaction_cost_model(trades, commission_bps=1.0, spread_bps=5.0,
                           impact_model=None, portfolio_value=1.0) -> dict
        # {'cost_frac','cost_cash','cost_breakdown'}. impact_model=None for SP1 (bps+half-spread only).
def generate_rebalance_dates(index, mode='calendar', freq='M', tau=0.05,
                             per_asset_band=None, target_weights=None) -> pd.DatetimeIndex
        # calendar mode: drop-in for rebalancing.py's inline freq logic. threshold/band mode:
        # decided inside the simulate loop on data up to d-1.
def drawdown_stats(wealth, top_n=5, periods_per_year=252) -> dict
        # {'max_drawdown','underwater_curve','worst_drawdowns','ulcer_index','pain_index'}; calls metrics.compute_drawdown
def simulate_strategy(price_data, target_weights, cost_model=None, rebalance='M',
                      initial_value=1.0, cash_drag=0.0) -> dict
        # target_weights: dict (one-shot, like simulate_rebalanced_portfolio) OR DataFrame (a per-date
        #   schedule indexed by rebalance dates).
        # rebalance: None (buy-and-hold) | 'M'/'Q'/'Y' | int — same vocabulary as simulate_rebalanced_portfolio
        #   (ignored when target_weights is a dated schedule, which carries its own rebalance dates).
        # cost_model: a callable(trades_dict) -> {'cost_frac', ...}, OR None (frictionless). The shipped
        #   default is functools.partial(transaction_cost_model, commission_bps=..., spread_bps=...); users
        #   may pass any callable of the same shape.
        # returns {'wealth' (Growth series), 'weights' (drifted history), 'blotter' (trades per rebalance),
        #          'turnover' (series), 'cost_drag' (float), 'summary' (dict of headline stats)}

# performance_stats.py
def probabilistic_sharpe_ratio(returns, sr_threshold=0.0, periods_per_year=252) -> float
def deflated_sharpe_ratio(returns, n_trials, periods_per_year=252) -> float
def compare_strategies_oos(returns_dict, benchmark_returns=None, n_trials=None,
                           sr_threshold=0.0, periods_per_year=252) -> dict
        # {'summary' (per-strategy SR/PSR/DSR), 'sharpe_diff_pvalues', 'best_by_dsr'}
```

## 5. Cost model (decision)

SP1 ships **commission (bps) + half-spread (bps)** applied to one-way turnover — covers the large majority of use and keeps `simulate_strategy` lean and well-tested. The `impact_model` parameter is **reserved, defaulting to `None`**; a square-root / participation-rate **market-impact** model (needs ADV + volatility inputs) is a **documented future TODO**, not implemented in SP1. Because `simulate_strategy` accepts any `cost_model` callable, users can also supply their own today.

## 6. The unlock — `run_rolling_windows` returns weights

`validation_report.run_rolling_windows` already builds per-window Equal/MinVol/MaxSharpe/User weight dicts and **discards** them. Change it to also return a `target_weight_schedule` (a DataFrame of weights indexed by rebalance/window date, per strategy) — an **additive** return (existing callers unaffected; TB4's migration preserved). This schedule is the input `simulate_strategy` consumes to produce a real out-of-sample, cost-aware backtest, whose per-strategy OOS returns feed `compare_strategies_oos`.

## 7. Data flow

```
optimizer / research weights
   │  (dict one-shot)            ┌── run_rolling_windows → target_weight_schedule (DataFrame)
   ▼                             ▼
 simulate_strategy(price, weights|schedule, cost_model, rebalance)
   ├─ generate_rebalance_dates          (when to trade)
   ├─ portfolio_turnover                (what trades each rebalance)
   ├─ transaction_cost_model(trades)    (cost drag)
   └─→ wealth, blotter, turnover, cost_drag, summary
            │
            ├─ drawdown_stats(wealth)                         (risk of the realized path)
            └─ compare_strategies_oos({name: oos_returns})    (PSR/DSR honest selection)
```

## 8. Invariants & testing (offline, deterministic)

- **Back-compat anchor:** `simulate_strategy(price, w, cost_model=None, rebalance=None)` (frictionless buy-and-hold) equals `simulate_rebalanced_portfolio(price, w, None)` / `portfolio_returns(...).growth['Portfolio']` within float tolerance.
- **Look-ahead safety (`hypothesis` property tests):** `compute_trailing_volatility` at row *d* uses only data ≤ *d*; `volatility_target_positions` lags the vol estimate ≥1 period before scaling; `simulate_strategy` decides a schedule/threshold rebalance at *d* on data ≤ *d−1*, trades at *d*. Shuffling future bars must not change past positions/wealth.
- **Golden numbers:** EWMA vs simple trailing vol on the synthetic fixture; turnover & cost arithmetic on hand-computed cases; `ledoit_wolf_covariance` returns a symmetric PD matrix with `shrinkage ∈ [0,1]` and the same annualization (`×252`) contract as `get_optimization_inputs`.
- **Cost monotonicity:** higher `commission_bps`/`spread_bps` ⇒ higher `cost_drag` ⇒ lower terminal wealth; zero bps ⇒ `cost_drag == 0`.
- **PSR/DSR sanity:** PSR ∈ [0,1]; DSR penalizes as `n_trials` rises; `compare_strategies_oos` ranks a clearly-better strategy first.
- All tests use the SP0 `make_synthetic_prices` fixture + `build_context_from_prices`; no network.

## 9. Back-compat & deps

- **No new dependencies** (numpy/pandas/scipy/sklearn/statsmodels already pinned). CVaR-style work is SP2; nothing here needs cvxpy.
- `simulate_rebalanced_portfolio`'s `(Series, DataFrame)` signature is **unchanged** — `simulate_strategy` is the additive superset.
- All new symbols re-exported from `__init__` under new comment-banner sections.
- Weights are accepted as `dict` (ticker→weight) and `np.ndarray`/`pd.Series` where natural; `simulate_strategy` bridges dict one-shot and DataFrame schedule so optimizer output (array) and report weights (dict) both flow in without an adapter.

## 10. Phasing

- **SP1a — Foundations:** `compute_trailing_volatility`, `volatility_target_positions`, `ledoit_wolf_covariance`, `portfolio_turnover`, `drawdown_stats`. Each independently useful, pure, offline-TDD. (Files: `signals.py`, `robust_estimators.py`, `backtest.py` partial.)
- **SP1b — Engine:** `generate_rebalance_dates`, `transaction_cost_model`, `simulate_strategy`, the `run_rolling_windows` unlock, `compare_strategies_oos` + PSR/DSR, capped by `examples/example_walk_forward_backtest.py` (rolling windows → schedule → cost-aware `simulate_strategy` → `compare_strategies_oos` with deflated Sharpe vs Equal/MinVol/MaxSharpe).

## 11. Definition of done

1. Five Phase-1 primitives + the engine functions exist in the new flat modules, re-exported from `__init__`.
2. `simulate_strategy` accepts dict **and** DataFrame-schedule weights; frictionless buy-and-hold matches `simulate_rebalanced_portfolio` (test).
3. `run_rolling_windows` returns a `target_weight_schedule` (additive; existing validation report unaffected).
4. `compare_strategies_oos`/PSR/DSR computed; the flagship walk-forward example runs end-to-end (offline-capable on the fixture; with real data in `examples/`).
5. Look-ahead property tests + golden + cost-monotonicity tests pass with **no network**; existing suite stays green.
6. CHANGELOG `[2.1.0]` Added: the backtest engine + Phase-1 primitives; market-impact noted as a future extension.
