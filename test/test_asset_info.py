# test/test_asset_info.py
import numpy as np
import pandas as pd
import pytest

from quant_reporter.asset_info import (
    compute_asset_analytics,
    compute_asset_factor_exposures,
    narrate_asset,
    build_asset_info_table,
)
from conftest import make_synthetic_prices


def _pw():
    prices = make_synthetic_prices()
    weights = {"AAA": 0.4, "BBB": 0.35, "CCC": 0.25}
    return prices, weights


# ---------------------------------------------------------------------------
# compute_asset_analytics
# ---------------------------------------------------------------------------

EXPECTED_COLS = {
    "total_return", "annualized_return", "annualized_vol",
    "sharpe", "sortino", "max_drawdown",
    "beta", "alpha", "weight", "correlation_to_portfolio",
}


def test_analytics_has_expected_columns():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert EXPECTED_COLS.issubset(set(out.columns))


def test_analytics_indexed_by_ticker():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert set(out.index) == {"AAA", "BBB", "CCC"}


def test_analytics_weights_match_input():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert out.loc["AAA", "weight"] == pytest.approx(0.4)
    assert out.loc["CCC", "weight"] == pytest.approx(0.25)


def test_analytics_vol_positive():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert (out["annualized_vol"] > 0).all()


def test_analytics_max_dd_non_positive():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert (out["max_drawdown"] <= 0).all()


def test_analytics_with_benchmark_gives_finite_beta():
    prices, weights = _pw()
    out = compute_asset_analytics(prices, weights, benchmark_col="BMK")
    assert out["beta"].notna().any()


def test_analytics_no_benchmark_gives_nan_beta():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    assert out["beta"].isna().all()


def test_analytics_correlation_to_portfolio_bounded():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    corr = out["correlation_to_portfolio"].dropna()
    assert ((corr >= -1.0) & (corr <= 1.0)).all()


def test_analytics_total_return_consistent():
    prices, weights = _pw()
    out = compute_asset_analytics(prices[["AAA", "BBB", "CCC"]], weights)
    # Manually check AAA total return
    r = prices["AAA"].pct_change().dropna()
    expected = float((1 + r).prod() - 1)
    assert out.loc["AAA", "total_return"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_asset_factor_exposures
# ---------------------------------------------------------------------------

def test_factor_exposures_shape():
    prices = make_synthetic_prices()
    returns = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    factors = pd.DataFrame({"F1": returns["AAA"] + 0.001, "F2": returns["BBB"] * 0.5},
                           index=returns.index)
    out = compute_asset_factor_exposures(returns, factors)
    assert out.shape == (3, 2)
    assert list(out.columns) == ["F1", "F2"]
    assert list(out.index) == ["AAA", "BBB", "CCC"]


def test_factor_exposures_finite():
    prices = make_synthetic_prices()
    returns = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    factors = pd.DataFrame({"F1": returns["AAA"]}, index=returns.index)
    out = compute_asset_factor_exposures(returns, factors)
    assert out.notna().any().any()


def test_factor_exposures_insufficient_data_gives_nan():
    prices = make_synthetic_prices(n_days=5)
    returns = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    factors = pd.DataFrame({"F1": returns["AAA"], "F2": returns["BBB"],
                            "F3": returns["CCC"], "F4": returns["CCC"] * 0.5,
                            "F5": returns["CCC"] * 0.3}, index=returns.index)
    out = compute_asset_factor_exposures(returns, factors)
    # With 4 obs and 5 factors → all NaN
    assert out.isna().all().all()


# ---------------------------------------------------------------------------
# narrate_asset
# ---------------------------------------------------------------------------

def test_narrate_returns_string():
    row = {"weight": 0.3, "annualized_return": 0.12, "annualized_vol": 0.18,
           "sharpe": 0.67, "max_drawdown": -0.15}
    text = narrate_asset(row)
    assert isinstance(text, str) and len(text) > 5


def test_narrate_llm_hook_used():
    row = {"weight": 0.3}
    result = narrate_asset(row, llm_hook=lambda r: "Custom narration.")
    assert result == "Custom narration."


def test_narrate_fallback_on_hook_error():
    row = {"weight": 0.3, "annualized_return": 0.10, "annualized_vol": 0.15,
           "sharpe": 0.67, "max_drawdown": -0.10}

    def bad_hook(r):
        raise RuntimeError("hook failed")

    text = narrate_asset(row, llm_hook=bad_hook)
    assert isinstance(text, str) and len(text) > 0


def test_narrate_nan_values():
    row = {"weight": float("nan"), "annualized_return": float("nan"),
           "annualized_vol": float("nan"), "sharpe": float("nan")}
    text = narrate_asset(row)
    assert isinstance(text, str)


def test_narrate_empty_hook_string_falls_back():
    row = {"weight": 0.3, "annualized_return": 0.08}
    text = narrate_asset(row, llm_hook=lambda r: "")
    assert isinstance(text, str)


# ---------------------------------------------------------------------------
# build_asset_info_table
# ---------------------------------------------------------------------------

def test_build_returns_dataframe():
    prices, weights = _pw()
    out = build_asset_info_table(prices[["AAA", "BBB", "CCC"]], weights)
    assert isinstance(out, pd.DataFrame)
    assert set(out.index) == {"AAA", "BBB", "CCC"}


def test_build_has_narration_column():
    prices, weights = _pw()
    out = build_asset_info_table(prices[["AAA", "BBB", "CCC"]], weights)
    assert "narration" in out.columns
    assert out["narration"].apply(lambda x: isinstance(x, str)).all()


def test_build_with_benchmark():
    prices, weights = _pw()
    out = build_asset_info_table(prices, weights, benchmark_col="BMK")
    assert "beta" in out.columns
    assert out["beta"].notna().any()


def test_build_with_factor_returns():
    prices, weights = _pw()
    returns = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    factor_returns = pd.DataFrame({"Mkt-RF": returns["AAA"],
                                   "SMB": returns["BBB"] * 0.5},
                                  index=returns.index)
    out = build_asset_info_table(prices[["AAA", "BBB", "CCC"]], weights,
                                 factor_returns=factor_returns)
    assert "Mkt-RF" in out.columns
    assert "SMB" in out.columns
