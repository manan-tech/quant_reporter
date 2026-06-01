import numpy as np
import pandas as pd
import pytest

from conftest import make_synthetic_prices
import quant_reporter as qr
from quant_reporter.validation_report import run_rolling_windows


def _ctx():
    prices = make_synthetic_prices(n_days=756)
    return qr.build_context_from_prices(
        prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK", "2021-01-01", "2022-06-30")


def test_run_rolling_windows_backcompat_returns_dataframe():
    out = run_rolling_windows(_ctx())
    assert isinstance(out, pd.DataFrame)  # unchanged default contract


def test_run_rolling_windows_schedule_unlock():
    rolling_df, schedule = run_rolling_windows(_ctx(), return_schedule=True)
    assert isinstance(rolling_df, pd.DataFrame)
    assert isinstance(schedule, dict)
    assert {"Equal Wt", "Min Vol", "Max Sharpe", "User Portfolio"} <= set(schedule)
    eq = schedule["Equal Wt"]
    assert isinstance(eq, pd.DataFrame)
    assert list(eq.columns) == ["AAA", "BBB", "CCC"]
    # equal-weight rows really are equal weights
    if not eq.empty:
        assert np.allclose(eq.iloc[0].values, 1.0 / 3.0)
        # each window's weights sum to ~1
        assert np.allclose(schedule["Min Vol"].sum(axis=1).values, 1.0, atol=1e-6)
