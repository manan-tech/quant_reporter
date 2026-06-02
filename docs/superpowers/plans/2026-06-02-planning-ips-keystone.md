# Planning / IPS Keystone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CFA-grounded `Profile` (IPS) primitive and three pure functions that make `quant_reporter`'s existing recommendation layer goals-aware, and close the latent seam where `recommend()` limits drove only alerts, not the optimizer.

**Architecture:** A new `planning.py` holds a `Profile` dataclass (presets resolved + validated in `__post_init__`), `build_profile`/`combine_risk_tolerance`, `apply_constraints` (Profile → optimizer `bounds`/`constraints`), and `check_suitability` (audit a recommendation → `SuitabilityReport`). An optional `profile=` param is threaded into `recommend_weights()` and `recommend()`; `profile=None` reproduces today's behavior exactly.

**Tech Stack:** Python 3.10+, numpy, pandas, scipy (via existing `opt_core`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-02-planning-ips-keystone-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/quant_reporter/planning.py` | **NEW** — Profile, presets, build_profile, combine_risk_tolerance, apply_constraints, check_suitability, SuitabilityCheck, SuitabilityReport |
| `src/quant_reporter/recommendation.py` | **MODIFY** — thread `profile=` into `recommend_weights`/`recommend`; add `Recommendation.suitability` + rendering |
| `src/quant_reporter/__init__.py` | **MODIFY** — export the new public surface |
| `test/test_planning.py` | **NEW** — unit + integration + regression tests |

**Key reference facts (verified in the codebase):**
- `opt_core.build_constraints(num_assets, raw_tickers, sector_map=None, sector_caps=None, sector_mins=None)` returns a **tuple** whose **index 0** is the budget constraint `{'type':'eq','fun': lambda x: np.sum(x)-1}`.
- `opt_core.get_optimization_inputs(prices)` returns `(mean, cov, log_returns)`; optimizer columns are `list(cov.columns)`.
- `metrics.max_drawdown(returns)` takes a returns Series and returns a **negative** float (NaN if empty).
- `recommendation.RecommendedWeights` has fields `weights, objective, rationale, evidence`; `evidence` carries `expected_vol` and `expected_return`.
- `recommendation.Recommendation` currently has fields `target_weights, trades, alerts, verdict` with `to_dict/to_text/to_html`.

---

## Task 1: `Profile` dataclass + presets + `build_profile` + `combine_risk_tolerance`

**Files:**
- Create: `src/quant_reporter/planning.py`
- Test: `test/test_planning.py`

- [ ] **Step 1: Write the failing tests**

Create `test/test_planning.py`:

```python
import numpy as np
import pandas as pd
import pytest

from quant_reporter.planning import (
    Profile, build_profile, combine_risk_tolerance,
)


def _prices(cols=("AAA", "BBB", "CCC", "DDD"), n=400, seed=0):
    """Deterministic synthetic price panel for tests."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, size=(n, len(cols)))
    px = 100 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(px, columns=list(cols), index=idx)


def test_profile_presets_fill_defaults():
    p = Profile(risk_tolerance="conservative")
    assert p.max_volatility == 0.08
    assert p.max_position_weight == 0.25


def test_profile_explicit_overrides_preset():
    p = Profile(risk_tolerance="conservative", max_volatility=0.05, max_position_weight=0.5)
    assert p.max_volatility == 0.05
    assert p.max_position_weight == 0.5


def test_profile_invalid_risk_tolerance_raises():
    with pytest.raises(ValueError):
        Profile(risk_tolerance="yolo")


def test_profile_invalid_liquidity_floor_raises():
    with pytest.raises(ValueError):
        Profile(liquidity_floor=1.5)


def test_profile_sector_caps_summing_below_one_raises():
    with pytest.raises(ValueError):
        Profile(sector_caps={"Tech": 0.3, "Fin": 0.3})


def test_combine_risk_tolerance_lower_governs():
    assert combine_risk_tolerance("aggressive", "conservative") == "conservative"
    assert combine_risk_tolerance("moderate", "aggressive") == "moderate"


def test_build_profile_from_ability_willingness():
    p = build_profile(ability="aggressive", willingness="moderate")
    assert p.risk_tolerance == "moderate"
    assert p.max_position_weight == 0.40  # moderate preset
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'quant_reporter.planning'`

- [ ] **Step 3: Write minimal implementation**

Create `src/quant_reporter/planning.py`:

```python
# src/quant_reporter/planning.py
"""Planning / IPS layer — investor goals & constraints as a reusable primitive.

Grounded in the CFA Level I IPS framework: a `Profile` captures the return
objective, risk tolerance (ability + willingness), and constraints (Time,
Taxes, Liquidity, Legal, Unique). Three pure functions bridge a profile to the
existing recommendation layer:

  build_profile(...)      -> Profile           construct (+ validate, + presets)
  apply_constraints(...)  -> (bounds, cons)    Profile -> optimizer inputs
  check_suitability(...)  -> SuitabilityReport audit a recommendation vs a profile

Library primitive: single-investor by default; advisors loop over profiles
themselves. Taxes are intentionally out of scope (later spec).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .opt_core import build_constraints
from .metrics import max_drawdown


_PRESETS = {
    "conservative": {"max_volatility": 0.08, "max_position_weight": 0.25},
    "moderate":     {"max_volatility": 0.12, "max_position_weight": 0.40},
    "aggressive":   {"max_volatility": 0.20, "max_position_weight": 0.60},
}
_ORDER = ["conservative", "moderate", "aggressive"]


@dataclass
class Profile:
    """Investor profile (IPS). Fields left None are filled from the
    risk-tolerance preset; values are validated on construction."""
    risk_tolerance: str = "moderate"
    max_volatility: Optional[float] = None        # annualized; preset-filled
    max_drawdown_tolerance: float = 0.20
    return_target: Optional[float] = None         # desired annualized return
    horizon_years: Optional[float] = None
    liquidity_floor: float = 0.0                  # min cash sleeve, [0, 1)
    max_position_weight: Optional[float] = None   # preset-filled
    sector_caps: Optional[dict] = None
    sector_mins: Optional[dict] = None
    excluded_assets: tuple = ()

    def __post_init__(self):
        if self.risk_tolerance not in _PRESETS:
            raise ValueError(
                f"risk_tolerance must be one of {list(_PRESETS)}; got {self.risk_tolerance!r}"
            )
        preset = _PRESETS[self.risk_tolerance]
        if self.max_volatility is None:
            self.max_volatility = preset["max_volatility"]
        if self.max_position_weight is None:
            self.max_position_weight = preset["max_position_weight"]
        if not (0.0 < self.max_position_weight <= 1.0):
            raise ValueError("max_position_weight must be in (0, 1]")
        if not (0.0 <= self.liquidity_floor < 1.0):
            raise ValueError("liquidity_floor must be in [0, 1)")
        if self.max_volatility <= 0:
            raise ValueError("max_volatility must be > 0")
        if not (0.0 < self.max_drawdown_tolerance <= 1.0):
            raise ValueError("max_drawdown_tolerance must be in (0, 1]")
        if self.horizon_years is not None and self.horizon_years <= 0:
            raise ValueError("horizon_years must be > 0")
        if self.sector_caps and sum(self.sector_caps.values()) < 1.0 - 1e-9:
            raise ValueError(
                "sector_caps sum to < 1; portfolio cannot be fully invested"
            )


def combine_risk_tolerance(ability, willingness):
    """Combine ability- and willingness-to-take-risk into a governing tolerance.

    CFA approach: the LOWER of the two governs (the binding constraint)."""
    for label, val in (("ability", ability), ("willingness", willingness)):
        if val not in _ORDER:
            raise ValueError(f"{label} must be one of {_ORDER}; got {val!r}")
    return _ORDER[min(_ORDER.index(ability), _ORDER.index(willingness))]


def build_profile(*, risk_tolerance=None, ability=None, willingness=None, **kwargs):
    """Friendly constructor. Either pass `risk_tolerance` directly, or pass
    `ability` and `willingness` to derive it via `combine_risk_tolerance`."""
    if risk_tolerance is None and ability is not None and willingness is not None:
        risk_tolerance = combine_risk_tolerance(ability, willingness)
    if risk_tolerance is None:
        risk_tolerance = "moderate"
    return Profile(risk_tolerance=risk_tolerance, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/planning.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(planning): Profile/IPS dataclass + build_profile + risk-tolerance combine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `apply_constraints` — Profile → optimizer bounds/constraints

**Files:**
- Modify: `src/quant_reporter/planning.py`
- Test: `test/test_planning.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_planning.py`:

```python
from quant_reporter.planning import apply_constraints


def test_apply_constraints_weight_cap():
    p = Profile(risk_tolerance="aggressive")  # cap 0.60
    bounds, cons = apply_constraints(p, ["AAA", "BBB", "CCC"])
    assert bounds == ((0.0, 0.60), (0.0, 0.60), (0.0, 0.60))


def test_apply_constraints_excludes_assets():
    p = Profile(risk_tolerance="aggressive", excluded_assets=("BBB",))
    bounds, cons = apply_constraints(p, ["AAA", "BBB", "CCC"])
    assert bounds[1] == (0.0, 0.0)
    assert bounds[0] == (0.0, 0.60)


def test_apply_constraints_liquidity_budget_targets_invested_fraction():
    p = Profile(risk_tolerance="aggressive", liquidity_floor=0.10)
    cols = ["AAA", "BBB", "CCC"]
    bounds, cons = apply_constraints(p, cols)
    budget = cons[0]                      # replaced budget constraint
    invested = np.full(len(cols), 0.90 / len(cols))   # sums to 0.90
    assert abs(budget["fun"](invested)) < 1e-9


def test_apply_constraints_default_budget_is_one():
    p = Profile(risk_tolerance="aggressive")
    bounds, cons = apply_constraints(p, ["AAA", "BBB", "CCC"])
    fully_invested = np.full(3, 1.0 / 3)
    assert abs(cons[0]["fun"](fully_invested)) < 1e-9


def test_apply_constraints_infeasible_cap_raises():
    # conservative cap 0.25 across 3 assets -> max 0.75 < 1.0 budget
    p = Profile(risk_tolerance="conservative")
    with pytest.raises(ValueError):
        apply_constraints(p, ["AAA", "BBB", "CCC"])


def test_apply_constraints_sector_caps_add_constraints():
    p = Profile(risk_tolerance="aggressive", sector_caps={"Tech": 0.5, "Fin": 0.6})
    sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Fin"}
    bounds, cons = apply_constraints(p, ["AAA", "BBB", "CCC"], sector_map=sector_map)
    assert len(cons) > 1   # budget + at least one sector cap inequality
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k apply_constraints -q`
Expected: FAIL — `ImportError: cannot import name 'apply_constraints'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/quant_reporter/planning.py`:

```python
def apply_constraints(profile, columns, *, sector_map=None):
    """Translate a Profile into optimizer (bounds, constraints).

    Box/linear limits only: per-asset upper bound = max_position_weight,
    excluded assets pinned to (0, 0), sector caps/mins via build_constraints,
    and a liquidity cash sleeve via the budget equality. Nonlinear limits
    (volatility, drawdown, return target) are evaluated in check_suitability.

    Returns (bounds, constraints) ready for find_optimal_portfolio /
    recommend_weights.
    """
    cols = list(columns)
    n = len(cols)
    cap = profile.max_position_weight
    excluded = set(profile.excluded_assets)
    invest_budget = 1.0 - profile.liquidity_floor

    if n * cap < invest_budget - 1e-9:
        raise ValueError(
            f"max_position_weight={cap} across {n} assets cannot reach the "
            f"{invest_budget:.0%} invested budget; raise the cap or add assets"
        )

    bounds = tuple((0.0, 0.0) if c in excluded else (0.0, cap) for c in cols)

    constraints = list(build_constraints(
        n, cols, sector_map=sector_map,
        sector_caps=profile.sector_caps, sector_mins=profile.sector_mins,
    ))
    # build_constraints emits sum(w)==1 at index 0; replace it with the
    # cash-sleeve budget when a liquidity floor is requested.
    if profile.liquidity_floor > 0:
        constraints[0] = {
            "type": "eq",
            "fun": lambda x, t=invest_budget: float(np.sum(x) - t),
        }
    return bounds, tuple(constraints)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k apply_constraints -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/planning.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(planning): apply_constraints bridges Profile to optimizer bounds/constraints

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Suitability types + `check_suitability`

**Files:**
- Modify: `src/quant_reporter/planning.py`
- Test: `test/test_planning.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_planning.py`:

```python
from quant_reporter.planning import check_suitability, SuitabilityReport
from quant_reporter.recommendation import RecommendedWeights


def _rw(weights, expected_vol=0.10, expected_return=0.12):
    return RecommendedWeights(
        weights=weights, objective="neg_sharpe", rationale="x",
        evidence={"expected_vol": expected_vol, "expected_return": expected_return},
    )


def test_suitability_concentration_fails():
    rw = _rw({"AAA": 0.8, "BBB": 0.2})
    rep = check_suitability(rw, Profile(risk_tolerance="moderate"))  # cap 0.40
    conc = next(c for c in rep.checks if c.name == "concentration")
    assert conc.passed is False
    assert rep.suitable is False


def test_suitability_exclusions_fail():
    rw = _rw({"AAA": 0.5, "BBB": 0.5})
    rep = check_suitability(rw, Profile(risk_tolerance="aggressive", excluded_assets=("AAA",)))
    excl = next(c for c in rep.checks if c.name == "exclusions")
    assert excl.passed is False
    assert rep.suitable is False


def test_suitability_liquidity_pass_and_fail():
    p = Profile(risk_tolerance="aggressive", liquidity_floor=0.10)  # invested <= 0.90
    rep_fail = check_suitability(_rw({"AAA": 0.5, "BBB": 0.5}), p)  # invested 1.0
    assert next(c for c in rep_fail.checks if c.name == "liquidity").passed is False
    rep_ok = check_suitability(_rw({"AAA": 0.45, "BBB": 0.45}), p)  # invested 0.90
    assert next(c for c in rep_ok.checks if c.name == "liquidity").passed is True


def test_suitability_volatility_from_evidence():
    rw = _rw({"AAA": 0.5, "BBB": 0.5}, expected_vol=0.30)
    rep = check_suitability(rw, Profile(risk_tolerance="moderate"))  # vol cap 0.12
    vol = next(c for c in rep.checks if c.name == "volatility")
    assert vol.passed is False
    assert rep.suitable is False


def test_suitability_return_target_is_informational():
    rw = _rw({"AAA": 0.3, "BBB": 0.3, "CCC": 0.4}, expected_vol=0.05, expected_return=0.03)
    p = Profile(risk_tolerance="aggressive", return_target=0.10)
    rep = check_suitability(rw, p)
    rt = next(c for c in rep.checks if c.name == "return_target")
    assert rt.passed is False        # 3% < 10% target
    assert rep.suitable is True      # informational only -> still suitable


def test_suitability_all_pass():
    rw = _rw({"AAA": 0.3, "BBB": 0.3, "CCC": 0.4}, expected_vol=0.05)
    rep = check_suitability(rw, Profile(risk_tolerance="aggressive"))
    assert isinstance(rep, SuitabilityReport)
    assert rep.suitable is True


def test_suitability_drawdown_with_prices():
    rw = _rw({"AAA": 0.5, "BBB": 0.5}, expected_vol=0.05)
    p = Profile(risk_tolerance="aggressive", max_drawdown_tolerance=0.001)  # absurdly tight
    rep = check_suitability(rw, p, prices=_prices(cols=("AAA", "BBB")))
    dd = next(c for c in rep.checks if c.name == "max_drawdown")
    assert dd.passed is False
    assert rep.suitable is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k suitability -q`
Expected: FAIL — `ImportError: cannot import name 'check_suitability'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/quant_reporter/planning.py`:

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

    def to_dict(self):
        return {
            "suitable": self.suitable,
            "rationale": self.rationale,
            "checks": [vars(c) for c in self.checks],
        }

    def to_text(self):
        lines = [f"Suitability: {'OK' if self.suitable else 'NOT SUITABLE'}"]
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        return "\n".join(lines)


def check_suitability(recommendation, profile, *, prices=None, sector_map=None):
    """Audit a Recommendation (or RecommendedWeights) against a Profile.

    Hard checks (flip `suitable`): concentration, exclusions, sector caps,
    liquidity, volatility, max_drawdown. The return-target check is
    informational only.
    """
    rw = getattr(recommendation, "target_weights", recommendation)
    weights = dict(getattr(rw, "weights", {}) or {})
    evidence = getattr(rw, "evidence", {}) or {}
    checks, hard = [], []

    # 1. Concentration
    if weights:
        tk = max(weights, key=lambda k: weights[k])
        observed = float(weights[tk])
        detail = f"max position {observed:.2%} ({tk}) vs cap {profile.max_position_weight:.2%}"
    else:
        observed, detail = 0.0, "no positions"
    passed = observed <= profile.max_position_weight + 1e-9
    checks.append(SuitabilityCheck("concentration", passed, detail, observed,
                                   profile.max_position_weight))
    hard.append(passed)

    # 2. Exclusions
    excl = set(profile.excluded_assets)
    excl_wt = float(sum(w for t, w in weights.items() if t in excl))
    passed = excl_wt <= 1e-9
    checks.append(SuitabilityCheck(
        "exclusions", passed,
        f"weight in excluded assets {excl_wt:.2%}" if excl else "no exclusions set",
        excl_wt, 0.0))
    hard.append(passed)

    # 3. Sector caps
    if sector_map and profile.sector_caps:
        exposures = {}
        for t, w in weights.items():
            sec = sector_map.get(t)
            if sec is not None:
                exposures[sec] = exposures.get(sec, 0.0) + w
        breaches = {s: exposures.get(s, 0.0) for s, cap in profile.sector_caps.items()
                    if exposures.get(s, 0.0) > cap + 1e-9}
        passed = not breaches
        observed = float(max(breaches.values(), default=0.0))
        detail = ("all sectors within caps" if passed else
                  "over cap: " + ", ".join(f"{s} {v:.2%}" for s, v in breaches.items()))
        checks.append(SuitabilityCheck("sector_caps", passed, detail, observed,
                                       float(max(profile.sector_caps.values()))))
        hard.append(passed)

    # 4. Liquidity (invested fraction must leave the cash sleeve)
    invested = float(sum(weights.values()))
    limit = 1.0 - profile.liquidity_floor
    passed = invested <= limit + 1e-6
    checks.append(SuitabilityCheck(
        "liquidity", passed,
        f"invested {invested:.2%} vs max {limit:.2%} (cash floor {profile.liquidity_floor:.2%})",
        invested, limit))
    hard.append(passed)

    # 5. Volatility (from evidence if available)
    ev_vol = evidence.get("expected_vol")
    if ev_vol is not None:
        observed = float(ev_vol)
        passed = observed <= profile.max_volatility + 1e-9
        checks.append(SuitabilityCheck(
            "volatility", passed,
            f"expected vol {observed:.2%} vs cap {profile.max_volatility:.2%}",
            observed, profile.max_volatility))
        hard.append(passed)

    # 6. Return target (INFORMATIONAL — does not flip `suitable`)
    if profile.return_target is not None:
        ev_ret = evidence.get("expected_return")
        if ev_ret is not None:
            observed = float(ev_ret)
            passed = observed >= profile.return_target - 1e-9
            checks.append(SuitabilityCheck(
                "return_target", passed,
                f"expected return {observed:.2%} vs target {profile.return_target:.2%} "
                f"(informational)",
                observed, profile.return_target))
            # intentionally NOT appended to `hard`

    # 7. Drawdown (only if prices provided)
    if prices is not None and weights:
        cols = [c for c in weights if c in prices.columns]
        if cols:
            w = np.array([weights[c] for c in cols], dtype=float)
            rets = prices[cols].pct_change().dropna()
            port = (rets * w).sum(axis=1)
            observed = abs(max_drawdown(port))
            passed = observed <= profile.max_drawdown_tolerance + 1e-9
            checks.append(SuitabilityCheck(
                "max_drawdown", passed,
                f"historical max drawdown {observed:.2%} vs tolerance "
                f"{profile.max_drawdown_tolerance:.2%}",
                float(observed), profile.max_drawdown_tolerance))
            hard.append(passed)

    suitable = all(hard)
    n_fail = sum(1 for p in hard if not p)
    rationale = ("All suitability checks pass." if suitable else
                 f"{n_fail} hard suitability check(s) failed; recommendation not "
                 f"suitable for this profile.")
    return SuitabilityReport(suitable=suitable, checks=checks, rationale=rationale)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k suitability -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/quant_reporter/planning.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(planning): check_suitability + SuitabilityReport audit of a recommendation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Thread `profile=` into `recommend_weights`

**Files:**
- Modify: `src/quant_reporter/recommendation.py:38-65`
- Test: `test/test_planning.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_planning.py`:

```python
from quant_reporter.recommendation import recommend_weights


def test_recommend_weights_profile_respects_cap():
    prices = _prices()
    rw = recommend_weights(prices, profile=Profile(max_position_weight=0.30))
    assert all(w <= 0.30 + 1e-6 for w in rw.weights.values())
    assert abs(sum(rw.weights.values()) - 1.0) < 1e-6


def test_recommend_weights_no_profile_runs_and_sums_to_one():
    prices = _prices()
    rw = recommend_weights(prices)               # profile=None: legacy path
    assert abs(sum(rw.weights.values()) - 1.0) < 1e-6


def test_recommend_weights_explicit_bounds_override_profile():
    prices = _prices()
    n = prices.shape[1]
    rw = recommend_weights(
        prices, profile=Profile(max_position_weight=0.30),
        bounds=tuple((0.0, 1.0) for _ in range(n)),
    )
    # explicit bounds win -> the 0.30 cap is NOT enforced via profile
    assert abs(sum(rw.weights.values()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k recommend_weights -q`
Expected: FAIL — `TypeError: recommend_weights() got an unexpected keyword argument 'profile'`

- [ ] **Step 3: Modify `recommend_weights`**

In `src/quant_reporter/recommendation.py`, replace the function definition and the lines that set `bounds`/`constraints` (currently lines 38-52):

```python
def recommend_weights(prices, *, objective=neg_sharpe, bounds=None, constraints=None,
                      profile=None, sector_map=None, risk_free_rate=0.02):
    """Point-in-time optimal target weights from a single objective.

    Optimizes `objective` over all columns of `prices` on the canonical
    (get_optimization_inputs) basis. Distinct from compare_verdict's
    backtest-driven pick. When `profile` is given and `bounds`/`constraints`
    are not supplied explicitly, they are derived from the profile via
    planning.apply_constraints (explicit bounds/constraints always win).
    """
    mean, cov, _ = get_optimization_inputs(prices)
    cols = list(cov.columns)
    n = len(cols)
    if profile is not None and bounds is None and constraints is None:
        from .planning import apply_constraints
        bounds, constraints = apply_constraints(profile, cols, sector_map=sector_map)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n))
    if constraints is None:
        constraints = build_constraints(n, cols)
```

(Leave the rest of the function body — the `with np.errstate(...)` block onward — unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k recommend_weights -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the existing recommendation tests (regression)**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/ -k recommend -q`
Expected: PASS (no regressions in existing recommendation tests)

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(recommendation): recommend_weights accepts profile= (derives bounds/constraints)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Thread `profile=` into `recommend` + `Recommendation.suitability` + rendering

**Files:**
- Modify: `src/quant_reporter/recommendation.py` (imports near top; `Recommendation` dataclass at 286-338; `recommend` at 341-361)
- Test: `test/test_planning.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/test_planning.py`:

```python
from quant_reporter.recommendation import recommend


def test_recommend_profile_constrains_optimizer_seam_fix():
    prices = _prices()
    rec = recommend(prices, profile=Profile(risk_tolerance="conservative"))  # cap 0.25
    # the seam fix: target weights themselves respect the cap, not just alerts
    assert all(w <= 0.25 + 1e-6 for w in rec.target_weights.weights.values())
    assert rec.suitability is not None
    conc = next(c for c in rec.suitability.checks if c.name == "concentration")
    assert conc.passed is True


def test_recommend_no_profile_suitability_none():
    prices = _prices()
    rec = recommend(prices)
    assert rec.suitability is None


def test_recommendation_to_dict_includes_suitability():
    prices = _prices()
    rec = recommend(prices, profile=Profile(risk_tolerance="aggressive"))
    d = rec.to_dict()
    assert "suitability" in d
    assert d["suitability"] is not None
    assert "checks" in d["suitability"]
    # legacy path: key present, value None
    assert recommend(prices).to_dict()["suitability"] is None


def test_recommendation_to_text_renders_suitability():
    prices = _prices()
    rec = recommend(prices, profile=Profile(risk_tolerance="aggressive"))
    assert "Suitability" in rec.to_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k "recommend_profile or suitability_none or to_dict_includes or to_text_renders" -q`
Expected: FAIL — `TypeError: recommend() got an unexpected keyword argument 'profile'`

- [ ] **Step 3a: Add the planning import near the top of `recommendation.py`**

After the existing `from .asset_info import compute_asset_factor_exposures` line (line 27), add:

```python
from .planning import check_suitability, SuitabilityReport
```

- [ ] **Step 3b: Add the `suitability` field to `Recommendation`**

In the `Recommendation` dataclass (line 287-291), change the fields to:

```python
@dataclass
class Recommendation:
    target_weights: RecommendedWeights
    trades: Optional[RebalancePlan]
    alerts: list              # list[RiskAlert]
    verdict: Optional[StrategyVerdict]
    suitability: Optional[SuitabilityReport] = None
```

- [ ] **Step 3c: Render `suitability` in `to_dict`**

In `Recommendation.to_dict`, add a `"suitability"` entry to the returned dict (after the `"verdict"` entry, before the closing brace):

```python
            "suitability": None if self.suitability is None else self.suitability.to_dict(),
```

- [ ] **Step 3d: Render `suitability` in `to_text`**

In `Recommendation.to_text`, immediately before the final `return "\n".join(lines)`, add:

```python
        if self.suitability is not None:
            lines += ["", self.suitability.to_text()]
```

- [ ] **Step 3e: Thread `profile=` through `recommend`**

Replace the `recommend` function (lines 341-361) with:

```python
def recommend(prices, *, current_weights=None, objective=neg_sharpe, results=None,
              cost_model=None, threshold=0.0, profile=None, vol_target=None,
              max_drawdown_limit=None, max_weight=None, max_risk_contribution=0.40,
              sector_map=None, sector_caps=None, factor_returns=None,
              factor_loading_limit=None, risk_free_rate=0.02):
    """Opt-in recommendation bundle. `prices` are asset prices only. Alerts run on
    `current_weights` when given, else on the recommended target.

    When `profile` is supplied: it constrains the optimizer (via recommend_weights),
    fills the alert thresholds (vol_target, max_drawdown_limit, max_weight,
    sector_caps) unless those are passed explicitly, and a SuitabilityReport is
    attached. `profile=None` reproduces the legacy behavior exactly.
    """
    if profile is not None:
        if vol_target is None:
            vol_target = profile.max_volatility
        if max_drawdown_limit is None:
            max_drawdown_limit = profile.max_drawdown_tolerance
        if max_weight is None:
            max_weight = profile.max_position_weight
        if sector_caps is None:
            sector_caps = profile.sector_caps
    # historical defaults when still unset (legacy behavior)
    if vol_target is None:
        vol_target = 0.10
    if max_drawdown_limit is None:
        max_drawdown_limit = 0.20
    if max_weight is None:
        max_weight = 0.40

    target = recommend_weights(prices, objective=objective, profile=profile,
                               sector_map=sector_map, risk_free_rate=risk_free_rate)
    trades = None
    if current_weights is not None:
        trades = rebalance_trades(current_weights, target.weights,
                                  cost_model=cost_model, threshold=threshold)
    alert_weights = current_weights if current_weights is not None else target.weights
    alerts = risk_alerts(alert_weights, prices, vol_target=vol_target,
                         max_drawdown_limit=max_drawdown_limit, max_weight=max_weight,
                         max_risk_contribution=max_risk_contribution,
                         sector_map=sector_map, sector_caps=sector_caps,
                         factor_returns=factor_returns,
                         factor_loading_limit=factor_loading_limit)
    verdict = compare_verdict(results) if results else None
    suitability = None
    if profile is not None:
        suitability = check_suitability(target, profile, prices=prices,
                                        sector_map=sector_map)
    return Recommendation(target_weights=target, trades=trades, alerts=alerts,
                          verdict=verdict, suitability=suitability)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -q`
Expected: PASS (all planning tests)

- [ ] **Step 5: Run existing recommendation tests (regression)**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/ -k recommend -q`
Expected: PASS (legacy recommend() behavior unchanged — `profile=None` path)

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/recommendation.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(recommendation): profile= constrains optimizer, fills thresholds, attaches suitability

Closes the seam where recommend() limits drove only alerts: target weights now
respect the profile's constraints. Backward compatible (profile=None unchanged).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Export public surface + full suite + lint

**Files:**
- Modify: `src/quant_reporter/__init__.py:226-229`
- Test: full suite

- [ ] **Step 1: Write the failing test**

Append to `test/test_planning.py`:

```python
def test_public_surface_exported():
    import quant_reporter as qr
    for name in ("Profile", "build_profile", "combine_risk_tolerance",
                 "apply_constraints", "check_suitability",
                 "SuitabilityReport", "SuitabilityCheck"):
        assert hasattr(qr, name), f"{name} not exported from quant_reporter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k public_surface -q`
Expected: FAIL — `AssertionError: Profile not exported from quant_reporter`

- [ ] **Step 3: Add exports to `__init__.py`**

In `src/quant_reporter/__init__.py`, immediately before the final `__version__ = "2.1.0"` line, add:

```python
# --- SP-Planning: IPS / Profile keystone (decision-support) ---
from .planning import (
    Profile,
    build_profile,
    combine_risk_tolerance,
    apply_constraints,
    check_suitability,
    SuitabilityCheck,
    SuitabilityReport,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/test_planning.py -k public_surface -q`
Expected: PASS

- [ ] **Step 5: Run the full test suite and lint**

Run: `cd /Users/mananbansal/Desktop/Majors/quant && python -m pytest test/ -q && ruff check src/`
Expected: All tests pass (existing 354 + new planning tests), ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/quant_reporter/__init__.py test/test_planning.py
git commit -m "$(cat <<'EOF'
feat(planning): export Profile/IPS public surface from quant_reporter

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Notes (for the implementer)

- **Circular imports:** `planning.py` imports only from `opt_core` and `metrics` (neither imports `recommendation` or `planning`), so `recommendation.py` can import `planning` at module top safely. `recommend_weights` still uses a local `from .planning import apply_constraints` to mirror the existing lazy-import style and keep the optimizer path import-light.
- **Backward compatibility is load-bearing:** the only behavioral change when `profile=None` is that `recommend`'s `vol_target`/`max_drawdown_limit`/`max_weight` defaults moved from literals to `None`-resolved-to-the-same-literals. Confirm the regression tests in Task 4/5 Step 5 stay green.
- **Feasibility:** `apply_constraints` raises if `n * max_position_weight < invested_budget` — tested in Task 2. This is why the conservative preset (cap 0.25) needs ≥ 4 assets to be fully invested; the `_prices()` fixture uses 4 columns.
