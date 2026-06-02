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


def test_combine_risk_tolerance_lower_governs():
    assert combine_risk_tolerance("aggressive", "conservative") == "conservative"
    assert combine_risk_tolerance("moderate", "aggressive") == "moderate"


def test_build_profile_from_ability_willingness():
    p = build_profile(ability="aggressive", willingness="moderate")
    assert p.risk_tolerance == "moderate"
    assert p.max_position_weight == 0.40  # moderate preset


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


def test_apply_constraints_infeasible_after_exclusion_raises():
    # moderate cap 0.40 with 3 assets: n * cap = 3 * 0.40 = 1.20 >= 1.0 (passes raw guard)
    # but with AAA excluded: n_investable * cap = 2 * 0.40 = 0.80 < 1.0 (must raise)
    p = Profile(risk_tolerance="moderate", excluded_assets=("AAA",))
    with pytest.raises(ValueError):
        apply_constraints(p, ["AAA", "BBB", "CCC"])


def test_apply_constraints_infeasible_liquidity_plus_exclusion_raises():
    # aggressive cap 0.60 with 3 assets: n * cap = 3 * 0.60 = 1.80 >= 0.85 (passes raw guard)
    # with 1 excluded: n_investable * cap = 2 * 0.60 = 1.20 >= 0.85 (still passes — correct)
    # with liquidity_floor=0.50: invest_budget=0.50, 2 * 0.60 = 1.20 >= 0.50 (passes — correct)
    # now use moderate cap 0.40, liquidity_floor=0.20, 2 excluded of 3:
    # n_investable=1, 1 * 0.40 = 0.40 < 0.80 invest_budget — must raise
    p = Profile(risk_tolerance="moderate", liquidity_floor=0.20, excluded_assets=("AAA", "BBB"))
    with pytest.raises(ValueError):
        apply_constraints(p, ["AAA", "BBB", "CCC"])


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
