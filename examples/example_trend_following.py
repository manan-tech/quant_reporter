"""example_trend_following.py — SP2 trend-following strategy demo.

Demonstrates a composite CTA-style sleeve:
  1. Time-series momentum signal
  2. MA crossover signal
  3. Ensemble (average of both)
  4. Vol-target the ensemble via volatility_target_positions
  5. Backtest with simulate_strategy + compare_strategies_oos

All offline — no network calls.

Run:
    python examples/example_trend_following.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import numpy as np
import pandas as pd

from conftest import make_synthetic_prices
from quant_reporter import (
    # SP2 Phase 5: tactical signals
    time_series_momentum_signal,
    moving_average_crossover_signal,
    cross_sectional_momentum_score,
    # SP1: vol-targeting (Phase 1 building block)
    volatility_target_positions,
    # SP1: backtest engine
    simulate_strategy,
    transaction_cost_model,
    compare_strategies_oos,
    # SP3: per-asset info
    build_asset_info_table,
)

# ── 1. Data ───────────────────────────────────────────────────────────────────
print("=" * 60)
print("Trend-Following Strategy Demo (Offline)")
print("=" * 60)

prices = make_synthetic_prices(seed=99, n_days=1000, assets=("A", "B", "C", "D"), benchmark="BMK")
assets = ["A", "B", "C", "D"]
returns = prices[assets].pct_change().dropna()
print(f"\nData: {len(prices)} trading days, {len(assets)} assets")

# ── 2. Signals ────────────────────────────────────────────────────────────────
print("\n[Building Signals]")

# Time-series momentum: 12-month lookback, skip 1 month
tsm = time_series_momentum_signal(prices[assets], lookback=252, skip_recent=21)
print(f"  TSM:         {tsm.notna().sum().min()} valid rows (first {tsm.first_valid_index()})")

# MA crossover: 50/200
mac = moving_average_crossover_signal(prices[assets], fast=50, slow=200)
print(f"  MAC(50/200): {mac.notna().sum().min()} valid rows")

# Cross-sectional momentum score: for cross-asset sizing
csm = cross_sectional_momentum_score(prices[assets], lookback=126, skip_recent=5)
print(f"  CSM score:   {csm.notna().sum().min()} valid rows")

# ── 3. Ensemble signal ────────────────────────────────────────────────────────
# Average TSM and MAC; align on the same valid period
ensemble_raw = (tsm + mac) / 2.0

print("\n[Last row of ensemble signal]")
print(ensemble_raw.dropna(how="any").tail(1).to_string())

# ── 4. Vol-target the ensemble ────────────────────────────────────────────────
# Scale each position so each asset targets 5% annualized vol contribution
print("\n[Applying vol-targeting (target 5% per asset)]")
ensemble_scaled = volatility_target_positions(
    ensemble_raw.dropna(how="any"),
    returns,
    target_vol=0.05,
    vol_lookback=63,
    method="ewma",
    max_leverage=2.0,
    scaling="per_asset",
)
print(f"  Non-NaN rows: {ensemble_scaled.dropna(how='any').shape[0]}")
print(f"  Mean gross exposure: {ensemble_scaled.abs().sum(axis=1).dropna().mean():.3f}")

# ── 5. Convert to daily weight schedule ───────────────────────────────────────
# Normalize rows to sum to 1 (long-only by taking positive signal, or stay long-biased)
# For a long-only CTA, floor at 0:
long_only = ensemble_scaled.clip(lower=0)
row_sum = long_only.sum(axis=1)
# Where sum > 0, normalize; otherwise hold equal weight
normalized = long_only.div(row_sum.where(row_sum > 0, other=1.0), axis=0)
eq_fallback = pd.DataFrame(1 / len(assets), index=normalized.index, columns=assets)
schedule = normalized.where(row_sum > 0, other=eq_fallback)

print(f"\n[Weight Schedule]")
print(f"  Rows: {len(schedule)}")
print(f"  First valid: {schedule.dropna(how='any').index[0]}")
print(f"  Last row:\n{schedule.tail(1).to_string()}")

# ── 6. Backtest via simulate_strategy ─────────────────────────────────────────
print("\n[Backtesting strategies]")
def cost_model(trades):
    return transaction_cost_model(trades, commission_bps=2.0, spread_bps=5.0)

# Equal-weight baseline
eq_sim = simulate_strategy(prices[assets], {a: 0.25 for a in assets},
                           cost_model=cost_model, rebalance="M")

# Trend-following
trend_sim = simulate_strategy(prices[assets], schedule,
                              cost_model=cost_model, rebalance="M")

print(f"\n  Equal-weight:  final wealth = {eq_sim['wealth'].iloc[-1]:.4f}")
print(f"    Ann return:  {eq_sim['summary'].get('annualized_return', float('nan')):.1%}")
print(f"    Max DD:      {eq_sim['summary'].get('max_drawdown', float('nan')):.1%}")

print(f"\n  Trend-follow:  final wealth = {trend_sim['wealth'].iloc[-1]:.4f}")
print(f"    Ann return:  {trend_sim['summary'].get('annualized_return', float('nan')):.1%}")
print(f"    Max DD:      {trend_sim['summary'].get('max_drawdown', float('nan')):.1%}")
print(f"    Cost drag:   {trend_sim['cost_drag']:.4f}")

# ── 7. OOS comparison ─────────────────────────────────────────────────────────
print("\n[OOS Strategy Comparison (PSR/DSR)]")
oos_returns = {
    "EqualWeight": eq_sim["wealth"].pct_change().dropna(),
    "TrendFollowing": trend_sim["wealth"].pct_change().dropna(),
}
comparison = compare_strategies_oos(oos_returns)
summary_df = pd.DataFrame(comparison["summary"]).T
print(summary_df.to_string(float_format="{:.4f}".format))
print(f"\n  Best by DSR: {comparison['best_by_dsr']}")

# ── 8. Per-asset info table (SP3) ─────────────────────────────────────────────
print("\n[Per-Asset Analytics (SP3)]")
weights = dict(schedule.mean())  # time-average weight
table = build_asset_info_table(
    prices[assets + ["BMK"]], weights,
    benchmark_col="BMK", risk_free_rate=0.02
)
cols_to_show = ["weight", "annualized_return", "annualized_vol", "sharpe", "max_drawdown", "beta"]
print(table[cols_to_show].to_string(float_format="{:.3f}".format))

print("\n[Narrations]")
for ticker, narration in table["narration"].items():
    print(f"  {ticker}: {narration}")

print("\n✓ Trend-following demo complete.")
