import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from .html_builder import generate_html_report
from .opt_core import get_portfolio_stats

def simulate_portfolio_paths(weights, mean_returns, cov_matrix, num_simulations=1000, time_horizon=252, initial_investment=10000):
    """
    Simulates future portfolio paths using Geometric Brownian Motion (GBM).
    
    Args:
        weights (list/array): Portfolio weights.
        mean_returns (series): Annualized mean returns of assets.
        cov_matrix (df): Annualized covariance matrix of assets.
        num_simulations (int): Number of paths to simulate.
        time_horizon (int): Number of trading days to simulate (e.g., 252 for 1 year).
        initial_investment (float): Starting value of the portfolio.
        
    Returns:
        simulation_df (pd.DataFrame): DataFrame of shape (time_horizon, num_simulations) with portfolio values.
    """
    print(f"Running {num_simulations} Monte Carlo simulations for {time_horizon} days...")
    
    weights = np.array(weights)
    
    # Calculate portfolio expected return and volatility
    port_return, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix)
    
    # Convert annualized metrics to daily
    daily_return = port_return / 252
    daily_vol = port_vol / np.sqrt(252)
    
    # Simulation logic using GBM: S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
    # We simulate daily steps.
    
    dt = 1 # 1 day step
    
    # Generate random shocks (Brownian motion)
    # Shape: (time_horizon, num_simulations)
    random_shocks = np.random.normal(0, 1, (time_horizon, num_simulations))
    
    # Calculate daily drift and diffusion
    # drift = (mu - 0.5 * sigma^2) * dt
    drift = (daily_return - 0.5 * daily_vol**2) * dt
    
    # diffusion = sigma * sqrt(dt) * Z
    diffusion = daily_vol * np.sqrt(dt) * random_shocks
    
    # Calculate daily log returns
    daily_log_returns = drift + diffusion
    
    # Calculate cumulative returns path
    # We start from 0 log return at t=0
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
    
    # Convert back to price paths
    # Add initial row of zeros for t=0
    cumulative_log_returns = np.vstack([np.zeros((1, num_simulations)), cumulative_log_returns])
    
    simulation_paths = initial_investment * np.exp(cumulative_log_returns)
    
    return pd.DataFrame(simulation_paths)

def calculate_simulation_metrics(simulation_df, confidence_level=0.95, actual_return=None):
    """
    Calculates risk metrics from the final values of the simulation.
    """
    final_values = simulation_df.iloc[-1]
    initial_value = simulation_df.iloc[0, 0]
    
    # Returns distribution
    total_returns = (final_values / initial_value) - 1
    
    mean_return = total_returns.mean()
    median_return = total_returns.median()
    
    # Value at Risk (VaR) - Historical method on simulated data
    # The q-th quantile of the distribution of returns
    var_percentile = np.percentile(total_returns, (1 - confidence_level) * 100)
    
    # Conditional Value at Risk (CVaR) - Expected loss given that loss > VaR
    cvar_percentile = total_returns[total_returns <= var_percentile].mean()
    
    metrics = {
        "Mean Expected Return": f"{mean_return:.2%}",
        "Median Expected Return": f"{median_return:.2%}",
        f"VaR ({confidence_level:.0%})": f"{var_percentile:.2%}",
        f"CVaR ({confidence_level:.0%})": f"{cvar_percentile:.2%}",
        "Best Case (95th percentile)": f"{np.percentile(total_returns, 95):.2%}",
        "Worst Case (5th percentile)": f"{np.percentile(total_returns, 5):.2%}"
    }
    
    if actual_return is not None:
        # Calculate percentile of actual return
        percentile = (total_returns < actual_return).mean()
        metrics["Actual Realized Return"] = f"{actual_return:.2%}"
        metrics["Realized Percentile"] = f"{percentile:.1%}"
    
    return metrics, total_returns

def calculate_success_probabilities(total_returns, thresholds=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]):
    """
    Calculates the probability of the portfolio return exceeding specific thresholds.
    """
    probs = {}
    for t in thresholds:
        prob = (total_returns > t).mean()
        probs[f"Return > {t:.0%}"] = f"{prob:.1%}"
    return probs

def plot_probability_curve(total_returns, actual_return=None):
    """
    Plots the 'Probability of Exceeding Return X' curve (Survival Function).
    """
    print("Plotting probability curve...")
    # Create a range of return thresholds from min to max
    x_values = np.linspace(total_returns.min(), total_returns.max(), 100)
    y_values = [(total_returns > x).mean() for x in x_values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=y_values, 
        mode='lines', 
        name='Probability of Exceeding',
        fill='tozeroy',
        fillcolor='rgba(0, 100, 80, 0.2)',
        line=dict(color='rgba(0, 100, 80, 1)')
    ))
    
    if actual_return is not None:
        actual_prob = (total_returns > actual_return).mean()
        fig.add_trace(go.Scatter(
            x=[actual_return],
            y=[actual_prob],
            mode='markers',
            name='Actual Return',
            marker=dict(color='red', size=10, symbol='star')
        ))
        fig.add_vline(x=actual_return, line_dash="dash", line_color="red", annotation_text="Actual")

    fig.update_layout(
        title='Probability of Exceeding Target Return',
        xaxis_title='Target Return',
        yaxis_title='Probability',
        yaxis_tickformat='.0%',
        xaxis_tickformat='.0%',
        template='plotly_white'
    )
    return fig

def plot_simulation_paths(simulation_df, actual_path=None):
    """
    Plots the first 100 simulation paths and the median path.
    Optionally overlays the actual realized path.
    """
    print("Plotting Monte Carlo paths...")
    fig = go.Figure()
    
    # Plot first 100 paths (or less if fewer sims)
    num_to_plot = min(100, simulation_df.shape[1])
    
    for i in range(num_to_plot):
        fig.add_trace(go.Scatter(
            y=simulation_df.iloc[:, i],
            mode='lines',
            line=dict(width=1, color='rgba(100, 149, 237, 0.2)'), # Cornflower blue, transparent
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Plot Median Path
    median_path = simulation_df.median(axis=1)
    fig.add_trace(go.Scatter(
        y=median_path,
        mode='lines',
        name='Median Predicted Path',
        line=dict(width=3, color='red', dash='dash')
    ))
    
    if actual_path is not None:
        # Ensure actual path starts at same initial value as simulation
        # Simulation starts at initial_investment (e.g. 10000)
        # Actual path is likely normalized to 1.0 or similar. 
        # We re-scale actual path to match simulation start.
        sim_start = simulation_df.iloc[0, 0]
        actual_start = actual_path.iloc[0]
        scaled_actual = actual_path * (sim_start / actual_start)
        
        # Truncate or extend x-axis? 
        # We plot actuals against the simulation steps (0 to T)
        # Assuming actual_path index is dates, we just use range(len)
        
        fig.add_trace(go.Scatter(
            y=scaled_actual.values,
            mode='lines',
            name='Actual Realized Path',
            line=dict(width=4, color='black')
        ))
    
    fig.update_layout(
        title='Monte Carlo: Predicted vs. Actual',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value',
        template='plotly_white',
        hovermode='x'
    )
    return fig

def plot_simulation_distribution(total_returns, actual_return=None):
    """
    Plots the histogram of final simulated returns.
    """
    print("Plotting simulation distribution...")
    fig = px.histogram(
        x=total_returns,
        nbins=50,
        title='Distribution of Final Simulated Returns',
        labels={'x': 'Total Return'},
        opacity=0.75
    )
    
    # Add VaR line
    var_95 = np.percentile(total_returns, 5)
    fig.add_vline(x=var_95, line_dash="dash", line_color="red", 
                  annotation_text=f"VaR (95%): {var_95:.2%}", 
                  annotation_position="top left")
                  
    if actual_return is not None:
        fig.add_vline(x=actual_return, line_dash="solid", line_color="black",
                      annotation_text=f"Actual: {actual_return:.2%}",
                      annotation_position="top right")
    
    fig.update_layout(template='plotly_white')
    return fig

def create_monte_carlo_report(weights, mean_returns, cov_matrix, 
                              num_simulations=1000, time_horizon=252, initial_investment=10000,
                              actual_return=None, actual_path=None,
                              filename="monte_carlo_report.html", title="Monte Carlo Simulation Report"):
    """
    Generates a standalone HTML report for Monte Carlo simulations.
    """
    print("--- Starting Monte Carlo Report Generation ---")
    
    # Run Simulation
    sim_df = simulate_portfolio_paths(weights, mean_returns, cov_matrix, num_simulations, time_horizon, initial_investment)
    
    # Calculate Metrics
    metrics, total_returns = calculate_simulation_metrics(sim_df, actual_return=actual_return)
    probs = calculate_success_probabilities(total_returns)
    
    # Generate Plots
    paths_plot = plot_simulation_paths(sim_df, actual_path=actual_path)
    dist_plot = plot_simulation_distribution(total_returns, actual_return=actual_return)
    prob_curve = plot_probability_curve(total_returns, actual_return=actual_return)
    
    # Build Report
    sections = [{
        "title": "Simulation Results",
        "description": f"Results from {num_simulations} simulations over a {time_horizon}-day horizon.",
        "sidebar": [
            {"title": "Risk Metrics", "type": "metrics", "data": metrics},
            {"title": "Success Probabilities", "type": "metrics", "data": probs}
        ],
        "main_content": [
            {"title": "Projected Paths vs Actual", "type": "plot", "data": paths_plot},
            {"title": "Return Distribution", "type": "plot", "data": dist_plot},
            {"title": "Probability of Exceeding Return", "type": "plot", "data": prob_curve}
        ]
    }]
    
    generate_html_report(sections, title=title, filename=filename)
    print(f"--- Monte Carlo Report Generated: {filename} ---")
