import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .metrics import calculate_metrics
from .plotting import (
    plot_cumulative_returns, 
    plot_rolling_volatility, 
    plot_regression, 
    plot_rolling_sharpe, 
    plot_monthly_distribution,
    plot_yearly_returns
)
from .opt_plotting import plot_portfolio_vs_constituents, plot_drawdown_comparison
from .html_builder import generate_html_report
from .opt_core import get_portfolio_price

def plot_correlation_heatmap(log_returns):
    """
    Generates a Plotly heatmap of the asset correlation matrix.
    """
    logger.debug("Plotting Correlation Heatmap for Portfolio Constituents...")
    corr_matrix = log_returns.corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale='RdYlGn', title='Constituent Correlation Heatmap'
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_drawdown(cumulative_returns, name):
    """
    Plots the drawdown 'underwater' curve for the portfolio.
    """
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, name=name,
        mode='lines', fill='tozeroy', line=dict(color='red')
    ))
    fig.update_layout(
        title=f'{name} Historical Drawdown',
        xaxis_title='Date', yaxis_title='Drawdown',
        yaxis_tickformat='.1%', hovermode='x unified', template='plotly_white'
    )
    return fig

def compute_portfolio_analysis(ctx: ReportContext):
    """
    Computes analysis explicitly for the portfolio vs benchmark report.
    """
    logger.info("Computing Portfolio Analysis...")
    
    # Calculate Portfolio Prices
    portfolio_prices = get_portfolio_price(
        ctx.price_data_full[ctx.friendly_tickers], 
        ctx.user_friendly_weights
    )
    
    # Add Portfolio to price_data temporarily for calculate_metrics
    eval_data = ctx.price_data_full.copy()
    eval_data['Portfolio'] = portfolio_prices
    
    # Generate Metrics and Base Plot Data
    metrics, plot_data = calculate_metrics(
        eval_data, 'Portfolio', ctx.friendly_benchmark, ctx.risk_free_rate
    )
    
    daily_returns_df = plot_data['daily_returns']
    cumulative_returns_df = plot_data['cumulative_returns']
    
    sidebar_items = [
        {"title": "Risk & Return Dashboard", "type": "metrics", "data": metrics}
    ]
    
    # Standard Portfolio vs Benchmark content
    main_content = [
        {
            "title": "Cumulative Returns", 
            "type": "plot", 
            "data": plot_cumulative_returns(plot_data, 'Portfolio', ctx.friendly_benchmark)
        },
        {
            "title": "Annual Returns", 
            "type": "plot", 
            "data": plot_yearly_returns(plot_data, 'Portfolio', ctx.friendly_benchmark)
        },
        {
            "title": "Drawdown Analysis",
            "type": "plot",
            "data": plot_drawdown(cumulative_returns_df['Asset'], "Portfolio")
        },
        {
            "title": "Rolling Volatility", 
            "type": "plot", 
            "data": plot_rolling_volatility(plot_data, 'Portfolio', ctx.friendly_benchmark)
        },
        {
            "title": "Rolling Sharpe Ratio", 
            "type": "plot", 
            "data": plot_rolling_sharpe(plot_data, 'Portfolio')
        },
        {
            "title": "Alpha/Beta Regression", 
            "type": "plot", 
            "data": plot_regression(plot_data, metrics, 'Portfolio', ctx.friendly_benchmark)
        },
        {
            "title": "Monthly Returns Distribution", 
            "type": "plot", 
            "data": plot_monthly_distribution(plot_data, 'Portfolio')
        },
    ]

    # Constituent Analysis Block
    tickers_to_plot = ['Portfolio'] + ctx.friendly_tickers + [ctx.friendly_benchmark]
    tickers_to_plot = [t for t in tickers_to_plot if t in eval_data.columns]
    
    all_daily_returns = eval_data[tickers_to_plot].pct_change().dropna()
    all_cumulative_returns = (1 + all_daily_returns).cumprod()
    
    constituent_content = [
        {
            "title": "Portfolio vs. Constituent Performance",
            "type": "plot",
            "data": plot_portfolio_vs_constituents(all_cumulative_returns)
        },
        {
            "title": "Constituent Correlation",
            "type": "plot",
            "data": plot_correlation_heatmap(ctx.log_returns)
        }
    ]
    
    sections = [
        {
            "title": "Portfolio Performance Summary",
            "description": f"Holistic performance analysis of the portfolio against '{ctx.friendly_benchmark}'.",
            "sidebar": sidebar_items,
            "main_content": main_content
        },
        {
            "title": "Constituent Breakdown",
            "description": "How individual assets contribute to and correlate within the portfolio.",
            "main_content": constituent_content
        }
    ]
    
    return sections

def create_portfolio_report(portfolio_dict, benchmark_ticker, train_start, train_end, 
                            filename="Portfolio_Report.html", **kwargs):
    """
    Standalone entry point to generate the Portfolio Report.
    """
    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_portfolio_analysis(ctx)
    generate_html_report(sections, title="Portfolio Report", filename=filename)
    logger.info("Portfolio report complete.")