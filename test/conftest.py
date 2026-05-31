import numpy as np
import pandas as pd
import pytest


def make_synthetic_prices(seed=42, n_days=756, assets=("AAA", "BBB", "CCC"), benchmark="BMK"):
    """Deterministic GBM price panel: assets + a benchmark column, ~3 business years."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    cols = list(assets) + [benchmark]
    frame = {}
    for i, c in enumerate(cols):
        mu = 0.0003 + 0.0001 * i
        sig = 0.010 + 0.002 * i
        daily = rng.normal(mu, sig, n_days)
        frame[c] = 100.0 * np.exp(np.cumsum(daily))
    return pd.DataFrame(frame, index=dates)


@pytest.fixture
def synthetic_prices():
    return make_synthetic_prices()


@pytest.fixture
def sample_data():
    """
    Creates a sample DataFrame to avoid hitting yfinance in tests.
    """
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B')

    bench_returns = np.random.normal(0.0005, 0.01, 252)
    asset_returns = 0.0002 + (1.2 * bench_returns) + np.random.normal(0, 0.005, 252)

    bench_price = 100 * (1 + bench_returns).cumprod()
    asset_price = 150 * (1 + asset_returns).cumprod()

    data = pd.DataFrame({
        'MSFT': asset_price,
        'SPY': bench_price
    }, index=dates)

    return data
