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
