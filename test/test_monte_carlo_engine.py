import numpy as np
import pandas as pd
from conftest import make_synthetic_prices
from quant_reporter.monte_carlo import simulate_portfolio_paths


def test_simulation_is_deterministic_with_seed():
    prices = make_synthetic_prices()
    rets = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    mean = rets.mean().values
    cov = rets.cov().values
    w = np.array([0.5, 0.3, 0.2])
    a = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=42)
    b = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=42)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_simulation_differs_without_seed():
    """Two unseeded runs should produce different results (with overwhelming probability)."""
    prices = make_synthetic_prices()
    rets = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    mean = rets.mean().values
    cov = rets.cov().values
    w = np.array([0.5, 0.3, 0.2])
    a = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=None)
    b = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=None)
    assert not np.allclose(np.asarray(a), np.asarray(b)), "Two unseeded runs should differ"


def test_stress_shock_reduces_final_values():
    """A negative stress shock should reduce median final portfolio value."""
    prices = make_synthetic_prices()
    rets = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    mean = rets.mean().values
    cov = rets.cov().values
    w = np.array([0.5, 0.3, 0.2])
    base_df = simulate_portfolio_paths(w, mean, cov, num_simulations=500, time_horizon=126, initial_investment=10000, seed=7)
    shocked_df = simulate_portfolio_paths(w, mean, cov, num_simulations=500, time_horizon=126, initial_investment=10000, stress_shock=-0.20, seed=7)
    assert shocked_df.iloc[-1].median() < base_df.iloc[-1].median(), \
        "A -20% Day-1 shock should reduce the median final value"


def test_different_seeds_give_different_results():
    """Two different seeds should produce different paths."""
    prices = make_synthetic_prices()
    rets = prices[["AAA", "BBB", "CCC"]].pct_change().dropna()
    mean = rets.mean().values
    cov = rets.cov().values
    w = np.array([0.5, 0.3, 0.2])
    a = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=1)
    b = simulate_portfolio_paths(w, mean, cov, num_simulations=200, time_horizon=126, initial_investment=10000, seed=2)
    assert not np.allclose(np.asarray(a), np.asarray(b)), "Different seeds should give different paths"
