import quant_reporter as qr
import pandas as pd

# 1. Define tickers and get data
tickers = ['AAPL', 'MSFT', 'GOOG', 'GLD']
data = qr.get_data(tickers, '2020-01-01', '2023-12-31')

# 2. Get optimization inputs
mean_returns, cov_matrix, log_returns = qr.get_optimization_inputs(data)

# 3. Define constraints (e.g., must sum to 1, no shorting)
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
print("--- Minimum Volatility Weights ---")
print(weights_df[weights_df > 0].map(lambda x: f"{x:.2%}"))

# 5. Create and show a plot
fig = qr.plot_correlation_heatmap(log_returns)
#fig.show() # Uncomment to display

# 6. Run Monte Carlo Simulation on the optimal portfolio
print("\n--- Running Monte Carlo Simulation on Min Vol Portfolio ---")
qr.create_monte_carlo_report(
    weights=min_vol_weights,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    num_simulations=500,
    time_horizon=252,
    initial_investment=10000,
    filename="monte_carlo_report.html"
)

# 7. (Optional) Calculate and print success probabilities directly
print("\n--- Probability Analysis ---")
# We need to run the simulation manually to get the returns for this print statement
# (The report generator does this internally, but we do it here to show the API)
sim_df = qr.simulate_portfolio_paths(min_vol_weights, mean_returns, cov_matrix, num_simulations=500, time_horizon=252)
_, total_returns = qr.calculate_simulation_metrics(sim_df)
probs = qr.calculate_success_probabilities(total_returns)
print("Probability of exceeding returns:")
for k, v in probs.items():
    print(f"  {k}: {v}")