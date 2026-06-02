import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_prices
import quant_reporter as qr


def _asset_prices(n_days=756):
    """Asset-only price panel (drop the benchmark column) for the recommend path."""
    return make_synthetic_prices(n_days=n_days)[["AAA", "BBB", "CCC"]]


def test_rolling_core_runs_on_raw_prices():
    from quant_reporter.validation_report import _rolling_oos_sharpe
    prices = make_synthetic_prices(n_days=756)
    cols = ["AAA", "BBB", "CCC"]
    strategies = {"EqualWt": lambda tr, m, c: {t: 1.0 / len(cols) for t in cols}}
    df = _rolling_oos_sharpe(prices, cols, strategies, window_years=1,
                             step_months=3, benchmark_col=None)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "EqualWt Sharpe" in df.columns


from quant_reporter.validation_report import run_rolling_windows


def _ctx():
    prices = make_synthetic_prices(n_days=756)
    return qr.build_context_from_prices(
        prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK", "2021-01-01", "2022-06-30")


def test_run_rolling_windows_no_profile_unchanged_columns():
    df = run_rolling_windows(_ctx())
    assert "Recommended Sharpe" not in df.columns
    assert "Max Sharpe Sharpe" in df.columns


def test_run_rolling_windows_recommended_column_with_profile():
    prof = qr.build_profile(max_position_weight=0.5)
    df, schedule = run_rolling_windows(_ctx(), return_schedule=True, profile=prof)
    assert "Recommended Sharpe" in df.columns
    assert "Recommended" in schedule
    rec = schedule["Recommended"].dropna(how="all")
    if not rec.empty:
        # every window's recommended weights respect the profile cap
        assert (rec.fillna(0.0).values <= 0.5 + 1e-6).all()


def test_recommend_validate_attaches_validation():
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    rec = qr.recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                       profile=prof, validate=True)
    assert rec.validation is not None
    assert rec.validation.n_windows >= 1
    assert rec.validation.baseline_oos_sharpe is not None
    assert rec.validation.verdict in ("holds up", "fragile (overfit)", "inconclusive")


def test_recommend_validate_false_leaves_validation_none():
    prices = _asset_prices()
    rec = qr.recommend(prices, profile=qr.build_profile(max_position_weight=0.6))
    assert rec.validation is None


def test_walk_forward_inconclusive_on_short_data():
    from quant_reporter.recommendation import walk_forward_recommendation
    prices = _asset_prices().iloc[:40]
    v = walk_forward_recommendation(prices, profile=qr.build_profile(max_position_weight=0.6))
    assert v.verdict == "inconclusive"
    assert v.n_windows == 0


def test_verdict_threshold_controls_outcome():
    from quant_reporter.recommendation import walk_forward_recommendation
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    strict = walk_forward_recommendation(prices, profile=prof, max_degradation=-1.0)
    assert strict.verdict == "fragile (overfit)"   # degradation >= 0 can never be <= -1
    lenient = walk_forward_recommendation(prices, profile=prof, max_degradation=10.0)
    if lenient.oos_sharpe > 0:
        assert lenient.verdict == "holds up"


def test_validation_renders_in_text_and_dict():
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    rec = qr.recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                       profile=prof, validate=True)
    assert "Validation (walk-forward)" in rec.to_text()
    d = rec.to_dict()
    assert d["validation"] is not None
    assert "oos_sharpe" in d["validation"]
    # legacy path: key present, value None
    assert qr.recommend(prices, profile=prof).to_dict()["validation"] is None


def test_validation_public_surface_exported():
    for name in ("RecommendationValidation", "walk_forward_recommendation"):
        assert hasattr(qr, name), f"{name} not exported from quant_reporter"


def test_validation_renders_in_html(tmp_path):
    prices = _asset_prices()
    prof = qr.build_profile(max_position_weight=0.6)
    rec = qr.recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                       profile=prof, validate=True)
    out = tmp_path / "rec.html"
    rec.to_html(str(out))
    html = out.read_text()
    assert "Walk-Forward Validation" in html
    assert any(v in html for v in ("HOLDS UP", "FRAGILE", "INCONCLUSIVE"))
