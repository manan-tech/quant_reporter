# Design: `planning.py` — Profile / IPS keystone

**Date:** 2026-06-02
**Status:** Approved — ready for implementation planning
**Component:** `quant_reporter` decision-support layer

## Context

`quant_reporter` 2.1.0 ships an opt-in recommendation layer (`recommendation.py`:
`recommend`, `recommend_weights`, `rebalance_trades`, `risk_alerts`,
`compare_verdict`). We are evolving the library along the **decision-support
(advisor)** direction: opinionated *allocation* advice driven by an investor's
goals and constraints — **not** per-asset long/short trading signals.

The audience decision is **library primitive**: ship a `Profile` dataclass plus
pure functions; advisors loop over many profiles in their own code. No client
registry, no app-like orchestration.

This spec covers the **first** build: the Profile / IPS keystone. Follow-on specs
(sequenced, not in scope here): estimation upgrade (Bayesian shrinkage,
regime-aware moments, Kalman beta) and the risk-management framework (formal risk
limits, stress testing).

### Latent bug this fixes

`recommend()` today accepts `max_weight`, `sector_caps`, `vol_target`, etc., but
only forwards them to `risk_alerts` — it does **not** pass `bounds`/`constraints`
into `recommend_weights`. So the recommended weights can violate the stated limits
and the user only gets an *alert after the fact*. The Profile layer makes one
profile constrain the optimizer **and** set the alert thresholds, consistently.

## Approach (chosen: A)

New `planning.py` with a `Profile` dataclass + three pure functions. Thread an
optional `profile=` parameter into `recommend_weights()` and `recommend()`. When a
profile is present it derives the optimizer's `bounds`/`constraints` **and** the
alert thresholds; when absent, behavior is unchanged (fully backward-compatible).

Rejected alternatives:
- **B — thresholds only (no optimizer coupling):** leaves the seam; advice can
  knowingly violate the client's goals, only flagged afterward.
- **C — separate `advise()` orchestrator:** parallel entry point duplicating
  `recommend()`'s orchestration; two ways to do the same thing.

## Section 1 — The `Profile` dataclass

Grounded in the CFA Level I IPS framework: **return objective** + **risk
tolerance** (ability + willingness) + **constraints** (Time horizon, Taxes,
Liquidity, Legal, Unique circumstances — "TTLU").

```python
@dataclass
class Profile:
    # Risk tolerance (ability + willingness, combined)
    risk_tolerance: str = "moderate"        # 'conservative'|'moderate'|'aggressive'
    max_volatility: float | None = None     # annualized; derived from preset if None
    max_drawdown_tolerance: float = 0.20
    # Return objective
    return_target: float | None = None      # desired annualized return
    # Constraints (TTLU)
    horizon_years: float | None = None      # Time
    liquidity_floor: float = 0.0            # Liquidity: min cash sleeve (0-1)
    max_position_weight: float = 1.0        # concentration cap
    sector_caps: dict | None = None         # Legal/Unique: sector exposure caps
    sector_mins: dict | None = None
    excluded_assets: tuple = ()             # ESG / Unique exclusions
```

Risk-tolerance **presets** fill derived defaults when a field is left `None`:

| Preset | `max_volatility` | `max_position_weight` |
|---|---|---|
| conservative | 0.08 | 0.25 |
| moderate | 0.12 | 0.40 |
| aggressive | 0.20 | 0.60 |

Explicit fields always override the preset. Taxes are deliberately deferred —
tax-aware rebalancing is a later spec.

## Section 2 — Pure functions + suitability types

### `build_profile(...) -> Profile`
Convenience constructor: applies presets to fill `None` fields, validates ranges
and raises `ValueError` on nonsense (negative horizon; `liquidity_floor` ∉ [0,1);
`max_position_weight` ∉ (0,1]; `max_volatility` ≤ 0; `max_drawdown_tolerance` ∉
(0,1]; unknown `risk_tolerance`).

> **Correction (post-implementation):** an earlier draft of this spec also
> required `sector_caps` summing to ≥ 1 to raise otherwise. That rule was
> **dropped during implementation** (commit `b8ac536`) because it is incorrect:
> `sector_caps` are *per-sector upper bounds*, not a partition of the portfolio —
> assets in uncapped sectors remain fully investable, so e.g. `{Tech: 0.3,
> Fin: 0.3}` is valid. No sector-cap-sum validation is performed.

Optional CFA-grounded helper `combine_risk_tolerance(ability, willingness)` — the
lower of the two governs, with a conflict note in the returned rationale.

### `apply_constraints(profile, columns, *, sector_map=None) -> (bounds, constraints)`
The Profile→optimizer bridge. Handles **box/linear** limits only:
- per-asset `bounds = (0, max_position_weight)`;
- excluded assets forced to `(0, 0)`;
- sector caps/mins via the existing `opt_core.build_constraints`;
- `liquidity_floor`: when `> 0`, the budget equality is `sum(w) == 1 -
  liquidity_floor` (remainder is implicit cash). `build_constraints` already emits a
  `sum(w) == 1` budget constraint, so `apply_constraints` must **replace** that
  budget constraint rather than append a second, conflicting one (when
  `liquidity_floor == 0`, the default `sum(w) == 1` is left as-is).

Returns objects directly consumable by `find_optimal_portfolio` /
`recommend_weights`.

**Deliberate separation:** `max_volatility`, `max_drawdown_tolerance`, and
`return_target` are nonlinear and are evaluated in `check_suitability`, not crammed
into the linear optimizer. (`sizing.vol_target_overlay` remains available if we
later want to scale risky weights to the vol cap.)

### `check_suitability(recommendation, profile, *, prices=None, sector_map=None) -> SuitabilityReport`
Audits a recommendation against the profile:
- concentration: any weight > `max_position_weight`;
- exclusions: any excluded asset with weight > 0;
- sector caps: sector exposures within caps (needs `sector_map`);
- liquidity: cash fraction ≥ `liquidity_floor` (i.e. `sum(weights) ≤ 1 - floor`);
- volatility: expected vol ≤ `max_volatility` (reads `evidence.expected_vol`);
- return target: expected_return ≥ `return_target` (informational shortfall flag);
- drawdown: if `prices` given, historical max drawdown ≤ `max_drawdown_tolerance`.

```python
@dataclass
class SuitabilityCheck:
    name: str
    passed: bool
    detail: str
    observed: float
    limit: float

@dataclass
class SuitabilityReport:
    suitable: bool          # all HARD checks pass
    checks: list            # list[SuitabilityCheck]
    rationale: str
    # + to_dict() / to_text()
```

The return-target check is informational (does not flip `suitable`); concentration,
exclusions, sector caps, liquidity, volatility, and drawdown are hard checks.

## Section 3 — Wiring into the existing layer (the seam fix)

- `recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None,
  profile=None, sector_map=None, risk_free_rate=0.02)` — when `profile` is set and
  `bounds`/`constraints` are `None`, derive them via `apply_constraints`. Explicit
  `bounds`/`constraints` still win.
- `recommend(prices, *, profile=None, ...)` — when `profile` is set: derive alert
  thresholds from it (`vol_target`, `max_drawdown_limit`, `max_weight`,
  `sector_caps`) unless explicitly overridden; pass `profile` into
  `recommend_weights` (**closes the seam**); attach a `SuitabilityReport`.
- `Recommendation` gains `suitability: SuitabilityReport | None = None` plus
  rendering in `to_dict`, `to_text`, and `to_html`.
- **Backward-compatible:** every new param defaults to `None`/unchanged;
  `profile=None` reproduces today's behavior exactly.

## Section 4 — Files & testing

| File | Change |
|---|---|
| `src/quant_reporter/planning.py` | **NEW** — Profile, build_profile, apply_constraints, check_suitability, SuitabilityCheck, SuitabilityReport, combine_risk_tolerance |
| `src/quant_reporter/recommendation.py` | thread `profile=`; add `suitability` field + rendering |
| `src/quant_reporter/__init__.py` | export the new public surface |
| `test/test_planning.py` | **NEW** |

### Tests
- `build_profile` validation raises `ValueError` on each bad input class.
- Preset application: `None` fields filled from the risk-tolerance preset; explicit
  fields override.
- `apply_constraints`: weight cap respected in bounds; excluded assets → `(0,0)`;
  sector caps/mins passed through; liquidity floor → `sum == 1 - floor` constraint.
- `check_suitability`: each hard rule fires on a crafted violation; return-target
  shortfall flags but does not flip `suitable`.
- Integration: `recommend(prices, profile=conservative)` output **actually respects**
  the weight cap (proves the seam fix) and carries a passing `SuitabilityReport`.
- Regression: `recommend()` / `recommend_weights()` with `profile=None` reproduce
  the current behavior exactly.

## Out of scope (follow-on specs)
- Estimation upgrade: Bayesian/Ledoit-Wolf shrinkage, regime-aware (Markov) moments,
  Kalman dynamic beta.
- Risk-management framework: configurable risk limits, scenario & stress testing,
  structured alert engine.
- Tax-aware rebalancing.
- HTML report styling for the suitability section beyond basic rendering.

## Source material
- **CFA L1 Vol 9 (Portfolio Management):** IPS, portfolio planning & construction,
  risk tolerance (ability vs. willingness), constraints framework.
- **CFA L2 Vol 5:** market-risk measurement (informs the deferred risk framework).
- **Ahlawat, Statistical Quantitative Methods in Finance:** Bayesian/regime methods
  (informs the deferred estimation upgrade).
