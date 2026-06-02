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
