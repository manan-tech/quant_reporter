"""
Tests for the 2.x factor/attribution engine:
- compute_brinson_attribution with the single-asset_returns API
- compute_factor_attribution / run_factor_regression robustness on a
  non-DatetimeIndex (regression test for the infer_freq guard).
"""
import numpy as np
import pandas as pd

from quant_reporter.attribution import compute_brinson_attribution
from quant_reporter.factor_models import (
    run_factor_regression,
    compute_factor_attribution,
)


def _sample_asset_returns():
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "AAPL": rng.normal(0.001, 0.01, 60),
            "MSFT": rng.normal(0.0008, 0.012, 60),
            "XOM": rng.normal(0.0005, 0.015, 60),
        },
        index=dates,
    )


_PW = {"AAPL": 0.5, "MSFT": 0.3, "XOM": 0.2}
_BW = {"AAPL": 0.4, "MSFT": 0.4, "XOM": 0.2}
_SECTORS = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}


def test_brinson_attribution_new_signature_columns():
    attr = compute_brinson_attribution(_PW, _BW, _sample_asset_returns(), _SECTORS)
    assert isinstance(attr, pd.DataFrame)
    for col in ("Allocation_Effect", "Selection_Effect", "Interaction_Effect"):
        assert col in attr.columns
    assert "Total" in attr.index


def test_brinson_total_effects_are_additive():
    attr = compute_brinson_attribution(_PW, _BW, _sample_asset_returns(), _SECTORS)
    total = attr.loc["Total"]
    additive = (
        total["Allocation_Effect"]
        + total["Selection_Effect"]
        + total["Interaction_Effect"]
    )
    assert abs(total["Total_Effect"] - additive) < 1e-9


def test_factor_attribution_handles_non_datetime_index():
    # Regression: previously raised
    #   TypeError: cannot infer freq from a non-convertible index of dtype int64
    portfolio_ret = pd.Series([0.01, 0.02, -0.01])  # default integer index
    factor_ret = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.01, -0.01],
            "SMB": [0.0, 0.01, 0.0],
            "HML": [0.0, 0.0, 0.0],
            "RF": [0.001, 0.001, 0.001],
        }
    )
    betas = pd.Series({"Mkt-RF": 1.0, "SMB": 0.5, "HML": 0.0})

    attr = compute_factor_attribution(portfolio_ret, factor_ret, betas, alpha=0.001)
    assert "RiskFree_Contribution" in attr.columns
    np.testing.assert_allclose(
        attr["Total_Return"].values,
        (attr["Explained"] + attr["Unexplained"]).values,
    )


def test_factor_regression_handles_non_datetime_index():
    portfolio_ret = pd.Series([0.01, 0.02, -0.01, 0.005, -0.002])
    factor_ret = pd.DataFrame(
        {
            "Mkt-RF": [0.01, 0.01, -0.01, 0.004, -0.003],
            "SMB": [0.0, 0.01, 0.0, 0.002, 0.0],
            "HML": [0.0, 0.0, 0.0, 0.001, 0.0],
            "RF": [0.001] * 5,
        }
    )
    res = run_factor_regression(portfolio_ret, factor_ret)
    assert "alpha" in res and "betas" in res
    assert np.isfinite(res["alpha"])
