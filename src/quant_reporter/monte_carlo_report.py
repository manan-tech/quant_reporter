import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .html_builder import generate_html_report
from .opt_core import get_portfolio_stats

def simulate_portfolio_paths(weights, mean_returns, cov_matrix, num_simulations=1000, time_horizon=252, initial_investment=10000, stress_shock=0.0):
    """
    Simulates future portfolio paths using Geometric Brownian Motion (GBM),
    with an optional Day 1 stress shock.
    """
    weights = np.array(weights)
    port_return, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix)
    
    daily_return = port_return / 252
    daily_vol = port_vol / np.sqrt(252)
    dt = 1 
    
    random_shocks = np.random.normal(0, 1, (time_horizon, num_simulations))
    drift = (daily_return - 0.5 * daily_vol**2) * dt
    diffusion = daily_vol * np.sqrt(dt) * random_shocks
    
    daily_log_returns = drift + diffusion
    
    # Apply initial stress shock to the first day's returns
    if stress_shock != 0.0:
        # Convert % shock (-0.20) to log return shock
        log_shock = np.log(1 + stress_shock)
        daily_log_returns[0, :] += log_shock
        
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
    cumulative_log_returns = np.vstack([np.zeros((1, num_simulations)), cumulative_log_returns])
    
    simulation_paths = initial_investment * np.exp(cumulative_log_returns)
    return pd.DataFrame(simulation_paths)

def calculate_time_to_target(simulation_df, target_return, initial_investment=10000):
    """
    Calculates the median time (in days) to achieve a target return.
    """
    target_value = initial_investment * (1 + target_return)
    
    # Find the day index where the value first exceeds target
    # idxmax returns the first index where arg is true. 
    # If it never exceeds, it returns 0.
    reached_mask = simulation_df >= target_value
    days_to_target = reached_mask.idxmax(axis=0)
    
    # Filter out paths that never reached it
    ever_reached = reached_mask.any(axis=0)
    valid_days = days_to_target[ever_reached]
    
    if len(valid_days) == 0:
        return None, 0.0
    
    return valid_days.median(), ever_reached.mean()

def calculate_simulation_metrics(simulation_df, confidence_level=0.95, actual_return=None):
    final_values = simulation_df.iloc[-1]
    initial_value = simulation_df.iloc[0, 0]
    total_returns = (final_values / initial_value) - 1
    
    mean_return = total_returns.mean()
    median_return = total_returns.median()
    var_percentile = np.percentile(total_returns, (1 - confidence_level) * 100)
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
        percentile = (total_returns < actual_return).mean()
        metrics["Actual Realized Return"] = f"{actual_return:.2%}"
        metrics["Realized Percentile"] = f"{percentile:.1%}"
    
    return metrics, total_returns

def calculate_success_probabilities(total_returns, thresholds=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50]):
    probs = {}
    for t in thresholds:
        prob = (total_returns > t).mean()
        probs[f"Return > {t:.0%}"] = f"{prob:.1%}"
    return probs

def plot_probability_curve(total_returns, actual_return=None):
    x_values = np.linspace(total_returns.min(), total_returns.max(), 100)
    y_values = [(total_returns > x).mean() for x in x_values]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=y_values, mode='lines', name='Probability of Exceeding', fill='tozeroy',
        fillcolor='rgba(0, 100, 80, 0.2)', line=dict(color='rgba(0, 100, 80, 1)')
    ))
    
    if actual_return is not None:
        actual_prob = (total_returns > actual_return).mean()
        fig.add_trace(go.Scatter(
            x=[actual_return], y=[actual_prob], mode='markers', name='Actual Return',
            marker=dict(color='red', size=10, symbol='star')
        ))
        fig.add_vline(x=actual_return, line_dash="dash", line_color="red", annotation_text="Actual")

    fig.update_layout(
        title='Probability of Exceeding Target Return', xaxis_title='Target Return', yaxis_title='Probability',
        yaxis_tickformat='.0%', xaxis_tickformat='.0%', template='plotly_white'
    )
    return fig

def plot_simulation_bands(simulation_df):
    """
    Plots confidence bands (5th, 25th, Median, 75th, 95th percentiles) instead of individual spaghetti paths.
    """
    logger.debug("Plotting Monte Carlo confidence bands...")
    percentiles = simulation_df.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
    
    fig = go.Figure()
    
    x = list(percentiles.index)
    x_rev = x[::-1]
    
    # 5th to 95th band (light band)
    y_upper_95 = list(percentiles[0.95])
    y_lower_95 = list(percentiles[0.05])[::-1]
    
    fig.add_trace(go.Scatter(
        x=x + x_rev,
        y=y_upper_95 + y_lower_95,
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='90% Confidence Interval'
    ))
    
    # 25th to 75th band (darker band)
    y_upper_75 = list(percentiles[0.75])
    y_lower_75 = list(percentiles[0.25])[::-1]
    
    fig.add_trace(go.Scatter(
        x=x + x_rev,
        y=y_upper_75 + y_lower_75,
        fill='toself',
        fillcolor='rgba(100, 149, 237, 0.4)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='50% Confidence Interval'
    ))
    
    # Median line
    fig.add_trace(go.Scatter(
        x=x, y=percentiles[0.5],
        line=dict(color='rgb(31, 119, 180)', width=2),
        mode='lines',
        name='Median Path'
    ))
    
    fig.update_layout(
        title='Simulated Value Pathways (Confidence Bands)',
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        hovermode="x"
    )
    return fig

def plot_simulation_distribution(total_returns, actual_return=None):
    fig = px.histogram(x=total_returns, nbins=50, title='Distribution of Final Simulated Returns', opacity=0.75)
    var_95 = np.percentile(total_returns, 5)
    fig.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text=f"VaR (95%): {var_95:.2%}", annotation_position="top left")
    if actual_return is not None:
         fig.add_vline(x=actual_return, line_dash="solid", line_color="black", annotation_text=f"Actual: {actual_return:.2%}", annotation_position="top right")
    fig.update_layout(template='plotly_white', xaxis_title="Total Return", yaxis_title="Count")
    return fig

def compute_monte_carlo_analysis(ctx: ReportContext, num_simulations=5000, time_horizon=252, initial_investment=10000):
    logger.info("Computing Monte Carlo...")
    
    user_weights = [ctx.user_friendly_weights.get(t, 0) for t in ctx.friendly_tickers]
    
    # Base Simulation
    sim_df = simulate_portfolio_paths(user_weights, ctx.mean_returns, ctx.cov_matrix, num_simulations, time_horizon, initial_investment)
    metrics, total_returns = calculate_simulation_metrics(sim_df)
    probs = calculate_success_probabilities(total_returns)
    
    # Time-to-Target Analysis
    targets = [0.05, 0.10, 0.20, 0.50]
    ttt_data = {}
    for t in targets:
        med_days, prob = calculate_time_to_target(sim_df, t, initial_investment)
        ttt_data[f"Target {t:.0%}"] = f"{med_days:.0f} days" if med_days else "Not Reached"
        ttt_data[f"Prob of reaching {t:.0%}"] = f"{prob:.1%}"
        
    ttt_df = pd.DataFrame.from_dict({"Time-to-Target": ttt_data}, orient='index').T
    ttt_html = ttt_df.to_html(classes='metrics-table')
    
    # Stress Scenarios (-10%, -20% initial shock)
    stress_results = []
    for shock in [-0.10, -0.20]:
        stress_df = simulate_portfolio_paths(user_weights, ctx.mean_returns, ctx.cov_matrix, num_simulations, time_horizon, initial_investment, stress_shock=shock)
        _, stress_returns = calculate_simulation_metrics(stress_df)
        prob_recovery = (stress_returns > 0).mean() # End above initial investment
        stress_results.append({
            "Scenario": f"{int(shock*100)}% Day-1 Shock",
            "Probability of Full Recovery": f"{prob_recovery:.1%}",
            "Median End Value": f"${stress_df.iloc[-1].median():,.0f}"
        })
        
    stress_df_out = pd.DataFrame(stress_results).set_index("Scenario")
    stress_html = stress_df_out.to_html(classes='metrics-table')

    bands_plot = plot_simulation_bands(sim_df)
    dist_plot = plot_simulation_distribution(total_returns)
    prob_curve = plot_probability_curve(total_returns)
    
    sections = [{
        "title": "Stochastic Return Forecasting",
        "description": f"Forecast based on {num_simulations} paths over {time_horizon} days.",
        "sidebar": [
            {"title": "Overall Simulation Metrics", "type": "metrics", "data": metrics},
            {"title": "Success Probabilities", "type": "metrics", "data": probs},
            {"title": "Time-to-Target", "type": "table_html", "data": ttt_html},
            {"title": "Stress Testing (1YR Recovery)", "type": "table_html", "data": stress_html}
        ],
        "main_content": [
            {"title": "Projected Value Bands", "type": "plot", "data": bands_plot},
            {"title": "Distribution of Expected Returns", "type": "plot", "data": dist_plot},
            {"title": "Probability Exceedance Curve", "type": "plot", "data": prob_curve}
        ]
    }]
    
    return sections

def create_monte_carlo_report(portfolio_dict, benchmark_ticker, train_start, train_end, 
                              filename="Monte_Carlo_Report.html", **kwargs):
    # Extract report-specific kwargs
    num_simulations = kwargs.get('num_simulations', 5000)
    time_horizon = kwargs.get('time_horizon', 252)
    initial_investment = kwargs.get('initial_investment', 10000)
    
    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_monte_carlo_analysis(ctx, 
                                            num_simulations=num_simulations, 
                                            time_horizon=time_horizon, 
                                            initial_investment=initial_investment)
    generate_html_report(sections, title="Monte Carlo Projection", filename=filename)
    logger.info("Monte Carlo generated.")
