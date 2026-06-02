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
