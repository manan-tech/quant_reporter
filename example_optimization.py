"""
Quick optimization example: Min-Vol weights, correlation heatmap, Monte Carlo.
Reports saved to ./reports/ directory.
"""
import logging
import quant_reporter as qr
import os
import pandas as pd

qr.enable_logging(logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Define tickers and get data
tickers = ['AAPL', 'MSFT', 'GOOG', 'GLD']
data = qr.get_data(tickers, '2020-01-01', '2023-12-31')

# 2. Get optimization inputs
mean_returns, cov_matrix, log_returns = qr.get_optimization_inputs(data)

# 3. Define constraints (must sum to 1, no shorting)
num_assets = len(tickers)
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = qr.build_constraints(num_assets, tickers)

# 4. Find the optimal weights
min_vol_weights = qr.find_optimal_portfolio(
    objective_func=qr.objective_min_variance,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    bounds=bounds,
    constraints=constraints,
    risk_free_rate=0.05
)

weights_df = pd.Series(min_vol_weights, index=tickers, name="Weights")
logger.info("Minimum Volatility Weights:")
for t, w in weights_df[weights_df > 0].items():
    logger.info("  %s: %.2f%%", t, w * 100)

# 5. Create and show a plot
fig = qr.plot_correlation_heatmap(log_returns)

# 6. Run Monte Carlo Simulation on the optimal portfolio
mc_path = os.path.join(REPORTS_DIR, 'Optimization_Monte_Carlo.html')
qr.create_monte_carlo_report(
    weights=min_vol_weights,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    num_simulations=500,
    time_horizon=252,
    initial_investment=10000,
    filename=mc_path
)
logger.info("Monte Carlo report saved: %s", mc_path)

# 7. Probability analysis
sim_df = qr.simulate_portfolio_paths(min_vol_weights, mean_returns, cov_matrix, num_simulations=500, time_horizon=252)
_, total_returns = qr.calculate_simulation_metrics(sim_df)
probs = qr.calculate_success_probabilities(total_returns)
logger.info("Success probabilities:")
for k, v in probs.items():
    logger.info("  %s: %s", k, v)