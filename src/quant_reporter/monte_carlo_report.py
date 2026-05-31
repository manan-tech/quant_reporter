import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .html_builder import generate_html_report
from .monte_carlo import (
    simulate_portfolio_paths,
    calculate_simulation_metrics,
    calculate_success_probabilities,
    plot_simulation_distribution,
    plot_probability_curve,
)


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


def compute_monte_carlo_analysis(ctx: ReportContext, num_simulations=5000, time_horizon=252, initial_investment=10000, seed=42):
    logger.info("Computing Monte Carlo...")

    user_weights = [ctx.user_friendly_weights.get(t, 0) for t in ctx.friendly_tickers]

    # Base Simulation — thread risk_free_rate from context; seed for reproducibility
    sim_df = simulate_portfolio_paths(
        user_weights, ctx.mean_returns, ctx.cov_matrix,
        num_simulations, time_horizon, initial_investment,
        risk_free_rate=ctx.risk_free_rate, seed=seed
    )
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
    # Each stress run uses a deterministic seed derived from the base seed so
    # stress results are also reproducible while differing from the base run.
    stress_results = []
    for i, shock in enumerate([-0.10, -0.20]):
        stress_seed = None if seed is None else seed + i + 1
        stress_df = simulate_portfolio_paths(
            user_weights, ctx.mean_returns, ctx.cov_matrix,
            num_simulations, time_horizon, initial_investment,
            stress_shock=shock, risk_free_rate=ctx.risk_free_rate, seed=stress_seed
        )
        _, stress_returns = calculate_simulation_metrics(stress_df)
        prob_recovery = (stress_returns > 0).mean()  # End above initial investment
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
    seed = kwargs.get('seed', 42)

    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_monte_carlo_analysis(ctx,
                                            num_simulations=num_simulations,
                                            time_horizon=time_horizon,
                                            initial_investment=initial_investment,
                                            seed=seed)
    generate_html_report(sections, title="Monte Carlo Projection", filename=filename)
    logger.info("Monte Carlo generated.")
