"""Offline tests for the Black-Litterman model and its provider contract.

No network: covariance/returns are constructed inline. These exercise the math
(market weights, implied equilibrium returns, posterior with absolute and
relative views) and the fail-loud behaviour when a custom provider cannot supply
market caps.
"""
import numpy as np
import pandas as pd
import pytest

from quant_reporter.black_litterman import (
    get_market_caps,
    calculate_market_weights,
    calculate_implied_equilibrium_returns,
    calculate_black_litterman_posterior,
)
from quant_reporter.providers import YFinanceProvider


TICKERS = ["AAPL", "MSFT", "XOM"]


def _cov():
    # A simple, positive-definite annualized covariance matrix.
    data = np.array([
        [0.040, 0.018, 0.006],
        [0.018, 0.050, 0.004],
        [0.006, 0.004, 0.030],
    ])
    return pd.DataFrame(data, index=TICKERS, columns=TICKERS)


def _caps():
    return pd.Series({"AAPL": 3.0e12, "MSFT": 2.5e12, "XOM": 0.5e12})


# --- provider contract -------------------------------------------------------

def test_get_market_caps_uses_provider_method():
    class CapProvider:
        def get_prices(self, tickers, start, end):
            raise AssertionError("should not be called")
        def get_risk_free_rate(self):
            return 0.03
        def get_market_caps(self, tickers):
            return {t: 100.0 * (i + 1) for i, t in enumerate(tickers)}

    caps = get_market_caps(TICKERS, provider=CapProvider())
    assert caps["AAPL"] == 100.0
    assert caps["XOM"] == 300.0


def test_get_market_caps_fails_loud_for_custom_provider_without_caps():
    class CSVProvider:  # custom, no get_market_caps
        def get_prices(self, tickers, start, end):
            return pd.DataFrame()
        def get_risk_free_rate(self):
            return 0.03

    with pytest.raises(ValueError, match="does not implement get_market_caps"):
        get_market_caps(TICKERS, provider=CSVProvider())


def test_get_market_caps_yfinance_provider_is_allowed_to_use_network(monkeypatch):
    # With the default YFinanceProvider, yfinance IS the sanctioned path — so it
    # should attempt it (we stub the network) rather than raising.
    import yfinance as yf

    class FakeTicker:
        def __init__(self, t):
            self._t = t
        @property
        def info(self):
            return {"marketCap": 1.0e12}

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    caps = get_market_caps(TICKERS, provider=YFinanceProvider())
    assert (caps > 0).all()
    assert set(caps.index) == set(TICKERS)


# --- math --------------------------------------------------------------------

def test_market_weights_sum_to_one():
    w = calculate_market_weights(_caps())
    assert pytest.approx(w.sum(), abs=1e-12) == 1.0
    assert w["AAPL"] > w["XOM"]  # bigger cap -> bigger weight


def test_implied_equilibrium_returns_shape_and_sign():
    cov = _cov()
    w = calculate_market_weights(_caps())
    pi = calculate_implied_equilibrium_returns(cov, w, risk_aversion=2.5)
    assert list(pi.index) == TICKERS
    assert (pi > 0).all()  # positive cov + positive weights -> positive premia


def test_posterior_no_views_returns_prior():
    cov = _cov()
    hist = pd.Series({"AAPL": 0.10, "MSFT": 0.09, "XOM": 0.05})
    pi, post_cov = calculate_black_litterman_posterior(
        hist, cov, market_caps=_caps(), tau=0.05,
    )
    # No views => posterior mean is the equilibrium prior; cov scales by (1+tau).
    expected_pi = calculate_implied_equilibrium_returns(
        cov, calculate_market_weights(_caps()), 2.5
    )
    pd.testing.assert_series_equal(pi, expected_pi, check_names=False)
    np.testing.assert_allclose(post_cov.values, (1.05) * cov.values)


def test_posterior_absolute_view_moves_toward_view():
    cov = _cov()
    hist = pd.Series({"AAPL": 0.10, "MSFT": 0.09, "XOM": 0.05})
    prior, _ = calculate_black_litterman_posterior(hist, cov, market_caps=_caps())
    post, _ = calculate_black_litterman_posterior(
        hist, cov, market_caps=_caps(),
        view_dict={"AAPL": 0.40},          # bullish, well above equilibrium
        view_confidences={"AAPL": 0.90},
    )
    # The strong bullish view should pull AAPL's posterior return up vs. the prior.
    assert post["AAPL"] > prior["AAPL"]


def test_posterior_relative_view_runs_and_is_finite():
    cov = _cov()
    hist = pd.Series({"AAPL": 0.10, "MSFT": 0.09, "XOM": 0.05})
    post, post_cov = calculate_black_litterman_posterior(
        hist, cov, market_caps=_caps(),
        relative_views=[("AAPL", "XOM", 0.05)],
        relative_view_confidences=[0.7],
    )
    assert np.isfinite(post.values).all()
    assert post_cov.shape == (3, 3)
