"""
Tests for the validation report's overfitting scoring.

calculate_overfitting_score now expects numeric metric dicts (from compute_metrics)
keyed by "Realized Sharpe" and "Realized CAGR" — no string round-trip via _to_num.
"""
from quant_reporter.validation_report import calculate_overfitting_score


def test_overfitting_score_computes_nonzero():
    # Numeric inputs matching compute_metrics output keys
    is_metrics = {"Strat": {"Realized Sharpe": 2.0, "Realized CAGR": 0.20}}
    oos_metrics = {"Strat": {"Realized Sharpe": 1.0, "Realized CAGR": 0.10}}
    df = calculate_overfitting_score(is_metrics, oos_metrics)
    assert abs(df.loc["Strat", "Overfitting Score"] - 0.5) < 1e-9
    assert abs(df.loc["Strat", "Strategy Degradation"] - 0.5) < 1e-9


def test_overfitting_score_floored_when_oos_better():
    is_metrics = {"Strat": {"Realized Sharpe": 1.0, "Realized CAGR": 0.10}}
    oos_metrics = {"Strat": {"Realized Sharpe": 1.5, "Realized CAGR": 0.15}}
    df = calculate_overfitting_score(is_metrics, oos_metrics)
    # Improvements are floored at 0 (no negative "overfitting").
    assert df.loc["Strat", "Overfitting Score"] == 0
    assert df.loc["Strat", "Strategy Degradation"] == 0
