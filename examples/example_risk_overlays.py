"""example_risk_overlays.py — SP2 risk overlay demo.

Demonstrates: inverse vol weights, vol-targeting, CPPI, risk parity,
and CVaR on a synthetic portfolio.  All offline — no network calls.

Run:
    python examples/example_risk_overlays.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import numpy as np
import pandas as pd

from conftest import make_synthetic_prices
from quant_reporter import (
    # SP1 engine
    simulate_strategy,
    compare_strategies_oos,
    # SP2 sizing
    inverse_volatility_weights,
    target_volatility_scalar,
    forecast_portfolio_vol,
    realized_tracking_error,
    kelly_fraction,
    cppi_weights,
    # SP2 Phase 4
    risk_contributions,
    optimize_risk_budget,
    portfolio_cvar,
    # SP1 (already present)
    ledoit_wolf_covariance,
)

# ── 1. Setup ──────────────────────────────────────────────────────────────────
print("=" * 60)
print("Risk Overlay Strategies Demo")
print("=" * 60)

prices = make_synthetic_prices(seed=42, n_days=756)  # ~3 years
assets = ["AAA", "BBB", "CCC"]
returns = prices[assets].pct_change().dropna()

# ── 2. Equal-weight baseline ───────────────────────────────────────────────────
equal_w = {a: 1 / 3 for a in assets}

# ── 3. Inverse-volatility weights ─────────────────────────────────────────────
inv_vol_w = inverse_volatility_weights(returns, lookback=63, method="ewma")
print("\n[Inverse-Vol Weights]")
for k, v in inv_vol_w.items():
    print(f"  {k}: {v:.3f}")

# ── 4. Risk-parity (equal risk contribution) ──────────────────────────────────
lw = ledoit_wolf_covariance(returns)
cov = lw["cov_matrix"]
rp_result = optimize_risk_budget(cov)
rp_w = rp_result["weights"]
print("\n[Equal Risk-Parity Weights]")
for k, v in rp_w.items():
    print(f"  {k}: {v:.3f}")
print("[Risk Contributions]")
rc = risk_contributions(rp_w, cov)
for k, v in rc.items():
    print(f"  {k}: {v:.3f}")

# ── 5. Portfolio-level vol stats ───────────────────────────────────────────────
print("\n[Forecast Portfolio Vol (annualized)]")
print(f"  Equal-weight:       {forecast_portfolio_vol(equal_w, cov):.1%}")
print(f"  Inv-vol weights:    {forecast_portfolio_vol(inv_vol_w, cov):.1%}")
print(f"  Risk-parity:        {forecast_portfolio_vol(rp_w, cov):.1%}")

# ── 6. Vol-targeting scalar ────────────────────────────────────────────────────
port_ret_ew = (returns * pd.Series(equal_w)).sum(axis=1)
scalar = target_volatility_scalar(port_ret_ew, target_vol=0.10)
print(f"\n[Vol-Targeting Scalar → 10%]  scalar = {scalar:.3f}")
scaled_w = {k: v * scalar for k, v in equal_w.items()}
print(f"  Scaled portfolio vol: {forecast_portfolio_vol(scaled_w, cov):.1%}")

# ── 7. Tracking error ─────────────────────────────────────────────────────────
port_ret_rp = (returns * pd.Series(rp_w)).sum(axis=1)
te = realized_tracking_error(port_ret_ew, port_ret_rp)
print(f"\n[Tracking Error: Equal-Weight vs Risk-Parity]  TE = {te:.1%} p.a.")

# ── 8. Kelly fraction (asset-level) ───────────────────────────────────────────
print("\n[Full Kelly Fraction per Asset]")
for col in assets:
    r = returns[col]
    mu_ex = float(r.mean()) - 0.02 / 252
    var = float(r.var(ddof=1))
    kf = kelly_fraction(mu_ex, var)
    print(f"  {col}: {kf:.1f}x  (half-Kelly: {kf/2:.1f}x)")

# ── 9. CPPI example ────────────────────────────────────────────────────────────
print("\n[CPPI Allocation  (floor=0.85, m=3, current_value=1.0)]")
result = cppi_weights(floor_value=0.85, multiplier=3, portfolio_value=1.0)
print(f"  Risky: {result['risky']:.1%}  |  Safe: {result['safe']:.1%}")

result_stressed = cppi_weights(floor_value=0.85, multiplier=3, portfolio_value=0.88)
print(f"  After drawdown to 0.88:")
print(f"  Risky: {result_stressed['risky']:.1%}  |  Safe: {result_stressed['safe']:.1%}")

# ── 10. CVaR ──────────────────────────────────────────────────────────────────
print("\n[Historical CVaR @ 95%]")
for label, w_dict in [("Equal-weight", equal_w), ("Inv-vol", inv_vol_w), ("Risk-parity", rp_w)]:
    port_r = (returns * pd.Series(w_dict)).sum(axis=1)
    cvar = portfolio_cvar(port_r, confidence=0.95)
    print(f"  {label:<18} CVaR = {cvar:.1%} p.a.")

# ── 11. Backtest comparison via simulate_strategy ──────────────────────────────
print("\n[OOS Strategy Comparison via simulate_strategy]")
strategy_returns = {}
for label, w_dict in [("EqualWeight", equal_w), ("InvVol", inv_vol_w), ("RiskParity", rp_w)]:
    sim = simulate_strategy(prices[assets], w_dict, rebalance="M")
    wealth = sim["wealth"]
    # compute simple OOS returns from wealth
    r = wealth.pct_change().dropna()
    strategy_returns[label] = r

comparison = compare_strategies_oos(strategy_returns)
summary_df = pd.DataFrame(comparison["summary"]).T
print(summary_df.to_string(float_format="{:.4f}".format))
print(f"\n  Best by DSR: {comparison['best_by_dsr']}")

print("\n✓ Risk overlay demo complete.")
