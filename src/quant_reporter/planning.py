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

    n_excluded = len(excluded & set(cols))
    n_investable = n - n_excluded
    if n_investable * cap < invest_budget - 1e-9:
        raise ValueError(
            f"max_position_weight={cap} across {n_investable} investable assets "
            f"(of {n} total; {n_excluded} excluded) cannot reach the "
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

    # 3. Sector caps — one SuitabilityCheck per breached sector so that
    # .limit always refers to the cap of *that* sector.  A single aggregate
    # check with limit=max(all caps) would make observed > limit false for
    # breaches of a tighter cap while max(all caps) is loose.
    if sector_map and profile.sector_caps:
        exposures = {}
        for t, w in weights.items():
            sec = sector_map.get(t)
            if sec is not None:
                exposures[sec] = exposures.get(sec, 0.0) + w
        any_breach = False
        for sec, cap in profile.sector_caps.items():
            obs = float(exposures.get(sec, 0.0))
            sec_cap = float(cap)
            sec_passed = obs <= sec_cap + 1e-9
            if not sec_passed:
                any_breach = True
                detail = f"sector {sec}: {obs:.2%} > cap {sec_cap:.2%}"
                checks.append(SuitabilityCheck(
                    f"sector_caps_{sec}", sec_passed, detail, obs, sec_cap))
                hard.append(sec_passed)

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
