# src/quant_reporter/backtest_report.py
"""Interactive backtest report (SP-Strategy).

Reuses the established plotly_white style and html_builder.generate_html_report.
Every number comes from the BacktestResult / metrics surface; nothing is recomputed.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .html_builder import generate_html_report

_PCT_KEYS = {"CAGR", "Volatility", "Max Drawdown", "Avg Drawdown", "VaR (95%)",
             "CVaR (95%)", "Downside Dev", "Tracking Error", "Hit Rate"}


def plot_wealth(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.wealth.index, y=result.wealth.values, name=result.name))
    if result.benchmark is not None:
        fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark.values,
                                 name="Benchmark"))
    fig.update_layout(title="Growth of $1", template="plotly_white", hovermode="x unified")
    return fig


def plot_drawdown(result):
    growth = result.wealth
    uw = (growth - growth.cummax()) / growth.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uw.index, y=uw.values, fill="tozeroy", name="Drawdown"))
    fig.update_layout(title="Underwater Drawdown", template="plotly_white")
    return fig


def plot_weights(result):
    w = result.weights
    fig = go.Figure()
    for col in w.columns:
        fig.add_trace(go.Scatter(x=w.index, y=w[col].values, stackgroup="one", name=str(col)))
    fig.update_layout(title="Weights Over Time", template="plotly_white")
    return fig


def plot_turnover(result):
    t = result.turnover
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(t.index), y=t.values, name="Turnover"))
    fig.update_layout(title="Turnover per Rebalance", template="plotly_white")
    return fig


def plot_rolling(result, window=63):
    r = result.returns
    roll_sharpe = (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)
    roll_vol = r.rolling(window).std() * np.sqrt(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values,
                             name=f"{window}d Rolling Sharpe"))
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol.values,
                             name=f"{window}d Rolling Vol", yaxis="y2"))
    fig.update_layout(title="Rolling Sharpe & Vol", template="plotly_white",
                      yaxis2=dict(overlaying="y", side="right", title="Vol"))
    return fig


def _fmt(key, val):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return "N/A"
    return f"{val:.2%}" if key in _PCT_KEYS else f"{val:.2f}"


def _metrics_table_html(result):
    rows = "".join(f"<tr><td>{k}</td><td>{_fmt(k, v)}</td></tr>"
                   for k, v in result.metrics.items())
    return f'<table class="metrics-table">{rows}</table>'


def build_sections(result):
    return [{
        "title": f"Backtest — {result.name}",
        "main_content": [
            {"type": "plot", "title": "Growth of $1", "data": plot_wealth(result)},
            {"type": "plot", "title": "Underwater Drawdown", "data": plot_drawdown(result)},
            {"type": "plot", "title": "Weights Over Time", "data": plot_weights(result)},
            {"type": "plot", "title": "Turnover per Rebalance", "data": plot_turnover(result)},
            {"type": "plot", "title": "Rolling Sharpe & Vol", "data": plot_rolling(result)},
            {"type": "table_html", "title": "Performance Metrics",
             "data": _metrics_table_html(result)},
            {"type": "table_html", "title": "Trade Blotter",
             "data": result.blotter.to_html(classes="metrics-table", index=False)},
        ],
    }]


def _comparison_section(results):
    from .performance_stats import compare_strategies_oos
    oos = {nm: res.returns for nm, res in results.items()}
    comp = compare_strategies_oos(oos, n_trials=len(results))
    summary_df = pd.DataFrame(comp["summary"]).T
    fig = go.Figure()
    for nm, res in results.items():
        fig.add_trace(go.Scatter(x=res.wealth.index, y=res.wealth.values, name=nm))
    fig.update_layout(title="Strategy Wealth Comparison", template="plotly_white")
    return {
        "title": "OOS Strategy Comparison",
        "description": f"Best by DSR: {comp['best_by_dsr']}",
        "main_content": [
            {"type": "plot", "title": "Wealth Comparison", "data": fig},
            {"type": "table_html", "title": "OOS Stats (SR / PSR / DSR)",
             "data": summary_df.to_html(classes="metrics-table", border=0,
                                        float_format=lambda x: f"{x:.3f}")},
        ],
    }


def create_backtest_report(result_or_results, path="backtest_report.html",
                           open_browser=False, recommendation=None):
    if isinstance(result_or_results, dict):
        sections = []
        for res in result_or_results.values():
            sections.extend(build_sections(res))
        sections.append(_comparison_section(result_or_results))
        title = "Strategy Comparison Report"
    else:
        sections = build_sections(result_or_results)
        title = f"Backtest Report — {result_or_results.name}"
    if recommendation is not None:
        from .recommendation_report import build_recommendation_section
        sections.append(build_recommendation_section(recommendation))
    generate_html_report(sections, title=title, filename=path)
    if open_browser:
        import os
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return path
