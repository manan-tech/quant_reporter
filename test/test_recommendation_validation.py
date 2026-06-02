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
