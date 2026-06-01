import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .analytics import format_metrics
from .plotting import (
    plot_cumulative_returns,
    plot_rolling_volatility,
    plot_regression,
    plot_rolling_sharpe,
    plot_monthly_distribution,
    plot_yearly_returns
)
from .opt_plotting import (
    plot_portfolio_vs_constituents,
    plot_correlation_heatmap,
)
from .html_builder import generate_html_report


def _plot_drawdown_from_curve(drawdown_curve, name):
    """
    Plots the drawdown 'underwater' curve for the portfolio using a pre-computed curve.
    The curve is sourced from ctx.analytics.drawdown.curve so it is consistent with
    the rest of the report.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_curve.index, y=drawdown_curve, name=name,
        mode='lines', fill='tozeroy', line=dict(color='red')
    ))
    fig.update_layout(
        title=f'{name} Historical Drawdown',
        xaxis_title='Date', yaxis_title='Drawdown',
        yaxis_tickformat='.1%', hovermode='x unified', template='plotly_white'
    )
    return fig


def _titled(fig, title):
    """Return fig with its layout title updated (non-mutating wrapper)."""
    fig.update_layout(title=title)
    return fig


def _build_plot_data(ctx):
    """
    Assemble the plot_data dict expected by the shared plotting helpers, sourcing
    Portfolio and Benchmark series from ctx.analytics (the single-source core).
    Per-constituent series are still derived from raw price data.
    """
    # Portfolio & Benchmark daily returns / growth from the analytics core
    core_daily = ctx.analytics.returns.daily   # cols: ['Portfolio', 'Benchmark']
    core_growth = ctx.analytics.returns.growth  # cols: ['Portfolio', 'Benchmark']

    # Rename to 'Asset'/'Benchmark' for the legacy plotting helpers
    daily_returns_df = core_daily.rename(columns={"Portfolio": "Asset"})
    cumulative_returns_df = core_growth.rename(columns={"Portfolio": "Asset"})

    asset_daily = daily_returns_df["Asset"]
    rfr_daily = ctx.risk_free_rate / 252
    excess = asset_daily - rfr_daily

    rolling_sharpe = (
        (excess.rolling(window=60).mean() * 252)
        / (excess.rolling(window=60).std() * np.sqrt(252))
    )

    monthly_returns = (
        cumulative_returns_df["Asset"]
        .resample("ME")
        .last()
        .pct_change()
        .dropna()
    )

    # Yearly returns need Asset + Benchmark columns
    asset_growth = cumulative_returns_df["Asset"]
    bench_growth = cumulative_returns_df["Benchmark"]
    growth_for_yearly = pd.concat(
        [asset_growth.rename("Asset"), bench_growth.rename("Benchmark")], axis=1
    )
    yearly_returns = (
        growth_for_yearly
        .resample("YE")
        .last()
        .pct_change()
        .dropna()
    )

    return {
        "daily_returns": daily_returns_df,
        "cumulative_returns": cumulative_returns_df,
        "monthly_returns": monthly_returns,
        "rolling_sharpe": rolling_sharpe,
        "yearly_returns": yearly_returns,
    }


def compute_portfolio_analysis(ctx: ReportContext):
    """
    Computes analysis explicitly for the portfolio vs benchmark report.
    Numbers are sourced from ctx.analytics (the single-source analytics core);
    no inline recomputation of Portfolio/Benchmark returns, drawdown, or metrics.
    """
    logger.info("Computing Portfolio Analysis...")

    # --- Metrics dashboard from the analytics core ---
    metrics = format_metrics(ctx.analytics.metrics)

    sidebar_items = [
        {"title": "Risk & Return Dashboard", "type": "metrics", "data": metrics}
    ]

    # --- Build plot_data for shared plotting helpers ---
    plot_data = _build_plot_data(ctx)

    # --- Drawdown: use pre-computed curve from ctx.analytics ---
    drawdown_curve = ctx.analytics.drawdown.curve

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
            "data": _plot_drawdown_from_curve(drawdown_curve, "Portfolio")
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

    # --- Constituent Analysis Block ---
    # Portfolio & Benchmark growth come from the core; per-constituent from price data
    portfolio_growth = ctx.analytics.returns.growth["Portfolio"]
    benchmark_growth = ctx.analytics.returns.growth["Benchmark"]

    constituent_prices = ctx.price_data_full[ctx.friendly_tickers]
    constituent_daily = constituent_prices.pct_change().dropna()
    constituent_growth = (1 + constituent_daily).cumprod()

    # The benchmark may already be one of the holdings (e.g. a 60/40 of SPY+AGG
    # benchmarked against SPY). In that case its growth line is already present
    # as a constituent column, so don't concat a duplicate (Plotly rejects
    # duplicate column names).
    frames = [portfolio_growth.rename("Portfolio"), constituent_growth]
    if ctx.friendly_benchmark not in constituent_growth.columns:
        frames.append(benchmark_growth.rename(ctx.friendly_benchmark))
    all_cumulative_returns = pd.concat(frames, axis=1).dropna(how="all")

    constituent_content = [
        {
            "title": "Portfolio vs. Constituent Performance",
            "type": "plot",
            "data": plot_portfolio_vs_constituents(all_cumulative_returns)
        },
        {
            "title": "Constituent Correlation (train period)",
            "type": "plot",
            "data": _titled(
                plot_correlation_heatmap(ctx.log_returns),
                "Constituent Correlation Heatmap (train period)"
            )
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