import numpy as np
import pandas as pd
import pytest

from quant_reporter.backtest import portfolio_turnover


def test_turnover_one_way_golden():
    out = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3})
    assert out["turnover"] == pytest.approx(0.2)   # 0.5 * (|0.2| + |0.2|)
    assert out["buys"] == pytest.approx(0.2)
    assert out["sells"] == pytest.approx(0.2)
    assert out["trades"]["A"] == pytest.approx(0.2)
    assert out["trades"]["B"] == pytest.approx(-0.2)


def test_turnover_two_way_doubles_one_way():
    one = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3}, convention="one_way")
    two = portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.7, "B": 0.3}, convention="two_way")
    assert two["turnover"] == pytest.approx(2 * one["turnover"])


def test_turnover_handles_new_and_dropped_assets():
    out = portfolio_turnover({"A": 1.0}, {"A": 0.5, "B": 0.5})
    assert out["trades"]["A"] == pytest.approx(-0.5)
    assert out["trades"]["B"] == pytest.approx(0.5)
    assert out["turnover"] == pytest.approx(0.5)


def test_turnover_identity_is_zero():
    out = portfolio_turnover({"A": 0.3, "B": 0.7}, {"A": 0.3, "B": 0.7})
    assert out["turnover"] == pytest.approx(0.0)


def test_turnover_accepts_series():
    a = pd.Series({"A": 0.5, "B": 0.5})
    b = pd.Series({"A": 0.7, "B": 0.3})
    assert portfolio_turnover(a, b)["turnover"] == pytest.approx(0.2)


def test_turnover_rejects_unknown_convention():
    with pytest.raises(ValueError):
        portfolio_turnover({"A": 1.0}, {"A": 1.0}, convention="round_trip")


from quant_reporter.backtest import drawdown_stats


def test_drawdown_stats_golden():
    wealth = pd.Series([1.0, 1.1, 0.99, 1.05, 1.21, 1.0],
                       index=pd.bdate_range("2022-01-03", periods=6))
    out = drawdown_stats(wealth, top_n=5)
    # deepest dd is the final 1.21 -> 1.00 leg: (1.00-1.21)/1.21
    assert out["max_drawdown"] == pytest.approx(-0.1735537, rel=1e-5)
    assert out["worst_drawdowns"][0]["depth"] == pytest.approx(-0.1735537, rel=1e-5)
    assert out["pain_index"] >= 0
    assert out["ulcer_index"] >= 0


def test_drawdown_stats_keys_and_underwater_curve():
    wealth = pd.Series(np.linspace(1.0, 2.0, 200) + np.sin(np.linspace(0, 12, 200)) * 0.05,
                       index=pd.bdate_range("2021-01-01", periods=200))
    out = drawdown_stats(wealth)
    assert set(out) >= {"max_drawdown", "underwater_curve", "worst_drawdowns",
                        "ulcer_index", "pain_index"}
    assert isinstance(out["underwater_curve"], pd.Series)
    assert (out["underwater_curve"] <= 1e-12).all()  # never positive


def test_drawdown_stats_monotone_wealth_has_zero_drawdown():
    wealth = pd.Series(np.cumprod(1 + np.full(100, 0.001)),
                       index=pd.bdate_range("2021-01-01", periods=100))
    out = drawdown_stats(wealth)
    assert out["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert out["worst_drawdowns"] == [] or out["worst_drawdowns"][0]["depth"] == pytest.approx(0.0, abs=1e-12)


def test_drawdown_stats_top_n_caps_episode_count():
    rng = np.random.default_rng(0)
    wealth = pd.Series(np.cumprod(1 + rng.normal(0, 0.02, 500)),
                       index=pd.bdate_range("2021-01-01", periods=500))
    out = drawdown_stats(wealth, top_n=3)
    assert len(out["worst_drawdowns"]) <= 3
