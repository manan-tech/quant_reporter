"""
Tests for the validation report's overfitting scoring.

Regression test: the score previously looked up "Sharpe Ratio"/"CAGR", but
calculate_metrics emits "... (Asset)" keys, so the lookups returned None and the
Overfitting Score / Strategy Degradation were silently always 0.
"""
from quant_reporter.validation_report import calculate_overfitting_score


def test_overfitting_score_computes_nonzero():
    is_metrics = {"Strat": {"Sharpe Ratio (Asset)": "2.00", "CAGR (Asset)": "20.00%"}}
    oos_metrics = {"Strat": {"Sharpe Ratio (Asset)": "1.00", "CAGR (Asset)": "10.00%"}}
    df = calculate_overfitting_score(is_metrics, oos_metrics)
    assert abs(df.loc["Strat", "Overfitting Score"] - 0.5) < 1e-9
    assert abs(df.loc["Strat", "Strategy Degradation"] - 0.5) < 1e-9


def test_overfitting_score_floored_when_oos_better():
    is_metrics = {"Strat": {"Sharpe Ratio (Asset)": "1.00", "CAGR (Asset)": "10.00%"}}
    oos_metrics = {"Strat": {"Sharpe Ratio (Asset)": "1.50", "CAGR (Asset)": "15.00%"}}
    df = calculate_overfitting_score(is_metrics, oos_metrics)
    # Improvements are floored at 0 (no negative "overfitting").
    assert df.loc["Strat", "Overfitting Score"] == 0
    assert df.loc["Strat", "Strategy Degradation"] == 0
