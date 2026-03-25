import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)
import plotly.io as pio
from plotly.subplots import make_subplots

# Import the core math function this file needs
from .opt_core import get_portfolio_stats

# --- All Optimization Plotting Functions ---

def plot_efficient_frontier(mean_returns, cov_matrix, optimal_portfolios, frontier_curve, risk_free_rate=0.02):
    """
    Generates a Plotly scatter plot of the efficient frontier.
    """
    logger.debug("Plotting Efficient Frontier...")
    
    num_ports = 2500
    all_weights = np.zeros((num_ports, len(mean_returns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    # Vectorized generation of random portfolios
    weights = np.random.random((num_ports, len(mean_returns)))
    weights /= weights.sum(axis=1)[:, np.newaxis]
    
    # Vectorized return and volatility calculation
    ret_arr = np.dot(weights, mean_returns)
    
    # Volatility is a bit trickier to vectorize fully without a loop for the quadratic form, 
    # but we can use einsum for speed: diag(w @ Sigma @ w.T)
    # However, a simple way that is still fast is:
    # vol = sqrt( sum( (w @ Sigma) * w, axis=1 ) )
    vol_arr = np.sqrt(np.sum(np.dot(weights, cov_matrix) * weights, axis=1))
    
    sharpe_arr = (ret_arr - risk_free_rate) / vol_arr
    sharpe_arr[vol_arr <= 0.00001] = 0 # Handle division by zero case

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=vol_arr, y=ret_arr, mode='markers',
        marker=dict(color=sharpe_arr, colorscale='Viridis', showscale=True, size=5, opacity=0.3, colorbar=dict(title="Sharpe Ratio")),
        name='Random Portfolios', text=[f"Sharpe: {s:.2f}" for s in sharpe_arr]
    ))
    
    fig.add_trace(go.Scatter(
        x=frontier_curve['Volatility'], y=frontier_curve['Return'], mode='lines',
        name='Efficient Frontier', line=dict(color='black', width=3)
    ))
    
    for name, data in optimal_portfolios.items():
        port_ret, port_vol, _ = get_portfolio_stats(data['weights_arr'], mean_returns, cov_matrix, risk_free_rate)
        fig.add_trace(go.Scatter(
            x=[port_vol], y=[port_ret], mode='markers', 
            marker=dict(color=data['color'], size=12, symbol='star', line=dict(width=1, color='Black')),
            name=name
        ))
    
    if "Max Sharpe (Unconstrained)" in optimal_portfolios:
        max_sharpe_data = optimal_portfolios["Max Sharpe (Unconstrained)"]
        msr_ret, msr_vol, _ = get_portfolio_stats(max_sharpe_data['weights_arr'], mean_returns, cov_matrix, risk_free_rate)
        cml_x = [0, msr_vol * 1.5]
        cml_y = [risk_free_rate, risk_free_rate + (msr_ret - risk_free_rate) / msr_vol * (msr_vol * 1.5)]
        
        fig.add_trace(go.Scatter(
            x=cml_x, y=cml_y, mode='lines',
            name='Capital Market Line (CML)', line=dict(color='black', dash='dash')
        ))
    
    fig.update_layout(
        title='Efficient Frontier & Optimal Portfolios',
        xaxis_title='Annualized Volatility (Risk)', yaxis_title='Annualized Return',
        hovermode='closest', template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.7)")
    )
    return fig

def plot_correlation_heatmap(log_returns):
    """
    Generates a Plotly heatmap of the asset correlation matrix.
    """
    logger.debug("Plotting Correlation Heatmap...")
    corr_matrix = log_returns.corr()
    fig = px.imshow(
        corr_matrix, text_auto=".2f",
        color_continuous_scale='RdYlGn', title='Asset Correlation Heatmap'
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_cumulative_comparison(cumulative_returns_df, benchmark_ticker):
    """
    Plots the "Growth of $1" for all optimized portfolios.
    """
    logger.debug("Plotting Strategy Cumulative Returns...")
    fig = go.Figure()
    
    for col in cumulative_returns_df.columns:
        fig.add_trace(go.Scatter(
            x=cumulative_returns_df.index, y=cumulative_returns_df[col], name=col,
            mode='lines', line=dict(width=2, dash=('dot' if col == benchmark_ticker else 'solid'))
        ))
    
    fig.update_layout(
        title='Strategy Performance: Cumulative Returns (Growth of $1)',
        xaxis_title='Date', yaxis_title='Cumulative Growth',
        hovermode='x unified', template='plotly_white'
    )
    return fig

def plot_drawdown_comparison(drawdown_df, benchmark_ticker):
    """
    Plots the drawdown "underwater" curves for all portfolios.
    """
    logger.debug("Plotting Strategy Drawdown...")
    fig = go.Figure()
    
    portfolios_to_plot = [col for col in drawdown_df.columns if col != benchmark_ticker]
    
    for col in portfolios_to_plot:
        fig.add_trace(go.Scatter(
            x=drawdown_df.index, y=drawdown_df[col], name=col,
            mode='lines', line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Strategy Performance: Drawdown',
        xaxis_title='Date', yaxis_title='Drawdown',
        yaxis_tickformat='.1%', hovermode='x unified', template='plotly_white'
    )
    return fig

def plot_rolling_sharpe(rolling_sharpe_df, benchmark_ticker):
    """
    Plots the 60-day rolling sharpe ratio for all strategies.
    """
    logger.debug("Plotting Rolling Sharpe...")
    fig = go.Figure()
    
    portfolios_to_plot = [col for col in rolling_sharpe_df.columns if col != benchmark_ticker]
    
    for col in portfolios_to_plot:
        fig.add_trace(go.Scatter(
            x=rolling_sharpe_df.index, y=rolling_sharpe_df[col], name=col,
            mode='lines', line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Strategy Performance: 60-Day Rolling Sharpe Ratio',
        xaxis_title='Date', yaxis_title='Sharpe Ratio',
        hovermode='x unified', template='plotly_white'
    )
    return fig

def plot_composition_pies(optimal_portfolios):
    """
    Plots side-by-side pie charts of portfolio weights.
    """
    logger.debug("Plotting Composition Pies...")
    
    port_names = list(optimal_portfolios.keys())
    n = len(port_names)
    
    short_names = {
        "Equal Weight (Baseline)": "Equal Wt",
        "Minimum Volatility": "Min Vol",
        "Balanced (40% Cap)": "Balanced (Asset Cap)",
        "Max Sharpe (Unconstrained)": "Max Sharpe",
        "Sector Balanced": "Balanced (Sector Cap)",
        "User Portfolio": "User",
        "Black-Litterman (Mkt Caps)": "Black-Litterman",
    }
    
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{'type':'domain'}] * n],
        subplot_titles=[short_names.get(name, name) for name in port_names],
        horizontal_spacing=0.03
    )
    
    for i, name in enumerate(port_names):
        weights_dict = optimal_portfolios[name]['weights_dict']
        labels = [ticker for ticker, weight in weights_dict.items() if weight > 0.001]
        values = [weight for ticker, weight in weights_dict.items() if weight > 0.001]
        
        fig.add_trace(go.Pie(
            labels=labels, values=values, name=name, hole=0.3,
            textinfo='percent+label',
            textposition='inside',
        ), row=1, col=i+1)
    
    fig.update_layout(
        title_text='Portfolio Strategy Compositions (by Asset)',
        width=max(900, n * 200),
        height=350,
        showlegend=False,
    )
    return fig

def plot_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, tickers, risk_free_rate=0.02):
    """
    Plots a 100% stacked bar chart of portfolio risk contribution.
    """
    logger.debug("Plotting Risk Contribution...")
    
    risk_data = []
    port_names = list(optimal_portfolios.keys())
    
    for name in port_names:
        data = optimal_portfolios[name]
        weights = data['weights_arr']
        port_ret, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        port_variance = port_vol**2
        marginal_contrib = np.dot(cov_matrix, weights)
        
        for i, ticker in enumerate(tickers):
            contrib = weights[i] * marginal_contrib[i]
            pct_contrib = contrib / port_variance if port_variance > 0 else 0
            risk_data.append({'Portfolio': name, 'Ticker': ticker, 'Risk Contribution': pct_contrib})
            
    risk_df = pd.DataFrame(risk_data)
    
    fig = px.bar(
        risk_df, x='Portfolio', y='Risk Contribution', color='Ticker',
        title='Portfolio Risk Contribution (by Asset)',
        # Re-order x-axis to match the legend of other plots
        category_orders={"Portfolio": port_names}
    )
    fig.update_layout(
        yaxis_tickformat='.0%', yaxis_title='Percent of Total Risk',
        xaxis_title=None, template='plotly_white', barmode='stack'
    )
    return fig

def plot_monthly_heatmaps(eval_data, benchmark_ticker):
    """
    Plots a heatmap of monthly returns for each strategy.
    """
    logger.debug("Plotting Monthly Returns Heatmaps...")
    
    port_names = [col for col in eval_data.columns if col != benchmark_ticker]
    
    daily_returns_df = eval_data.pct_change().dropna()
    monthly_returns = daily_returns_df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.strftime('%b')
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=len(port_names), cols=1,
        subplot_titles=port_names, vertical_spacing=0.05
    )

    for i, name in enumerate(port_names):
        pivot = monthly_returns.pivot_table(index='Year', columns='Month', values=name)
        pivot = pivot.reindex(columns=month_order)
        
        fig.add_trace(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale='RdYlGn', zmid=0,
            text=pivot.map(lambda x: f"{x:.1%}" if not pd.isna(x) else ""),
            texttemplate="%{text}", name=name,
            showscale=(i == 0)
        ), row=i+1, col=1)

    fig.update_layout(
        title_text='Strategy Monthly Returns Heatmap',
        height=180 * len(port_names), template='plotly_white'
    )
    return fig

def plot_portfolio_vs_constituents(all_cumulative_returns):
    """
    Plots the cumulative returns for the portfolio and all its underlying assets.
    """
    logger.debug("Plotting Portfolio vs. Constituents...")
    fig = go.Figure()
    
    for col in all_cumulative_returns.columns:
        fig.add_trace(go.Scatter(
            x=all_cumulative_returns.index,
            y=all_cumulative_returns[col],
            name=col,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Portfolio vs. Constituent Performance',
        xaxis_title='Date',
        yaxis_title='Cumulative Growth',
        hovermode='x unified',
        template='plotly_white'
    )
    return fig

def plot_sector_allocation_pies(optimal_portfolios, friendly_sector_map):
    """
    Plots side-by-side pie charts of portfolio weights aggregated by sector.
    """
    logger.debug("Plotting Sector Allocation Pies...")
    if not friendly_sector_map:
        return go.Figure().update_layout(title="Sector Allocation (No sector_map provided)")

    port_names = list(optimal_portfolios.keys())
    n = len(port_names)
    
    short_names = {
        "Equal Weight (Baseline)": "Equal Wt", "Minimum Volatility": "Min Vol",
        "Balanced (40% Cap)": "Balanced (Asset Cap)", "Max Sharpe (Unconstrained)": "Max Sharpe",
        "Sector Balanced": "Balanced (Sector Cap)", "User Portfolio": "User",
        "Black-Litterman (Mkt Caps)": "Black-Litterman",
    }
    
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{'type':'domain'}] * n],
        subplot_titles=[short_names.get(name, name) for name in port_names],
        horizontal_spacing=0.03
    )
    
    for i, name in enumerate(port_names):
        weights_dict = optimal_portfolios[name]['weights_dict']
        
        # Aggregate weights by sector
        sector_weights = {}
        for ticker, weight in weights_dict.items():
            if weight > 0.001:
                sector = friendly_sector_map.get(ticker, "Other")
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        labels = list(sector_weights.keys())
        values = list(sector_weights.values())
        
        fig.add_trace(go.Pie(
            labels=labels, values=values, name=name, hole=0.3,
            textinfo='percent+label',
            textposition='inside',
        ), row=1, col=i+1)

    fig.update_layout(
        title_text='Portfolio Strategy Compositions (by Sector)',
        width=max(900, n * 200),
        height=350,
        showlegend=False,
    )
    return fig

def plot_sector_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, tickers, friendly_sector_map, risk_free_rate=0.02):
    """
    Plots a 100% stacked bar chart of portfolio risk contribution by sector.
    """
    logger.debug("Plotting Sector Risk Contribution...")
    if not friendly_sector_map:
        return go.Figure().update_layout(title="Sector Risk Contribution (No sector_map provided)")

    risk_data = []
    port_names = list(optimal_portfolios.keys())
    
    for name in port_names:
        data = optimal_portfolios[name]
        weights = data['weights_arr']
        port_ret, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
        port_variance = port_vol**2
        marginal_contrib = np.dot(cov_matrix, weights)
        
        for i, ticker in enumerate(tickers):
            contrib = weights[i] * marginal_contrib[i]
            pct_contrib = contrib / port_variance if port_variance > 0 else 0
            risk_data.append({
                'Portfolio': name,
                'Ticker': ticker,
                'Risk Contribution': pct_contrib
            })
            
    risk_df = pd.DataFrame(risk_data)
    
    # Map tickers to sectors
    risk_df['Sector'] = risk_df['Ticker'].map(friendly_sector_map).fillna("Other")
    
    # Aggregate by sector
    sector_risk_df = risk_df.groupby(['Portfolio', 'Sector'])['Risk Contribution'].sum().reset_index()

    fig = px.bar(
        sector_risk_df,
        x='Portfolio',
        y='Risk Contribution',
        color='Sector',
        title='Portfolio Risk Contribution (by Sector)',
        category_orders={"Portfolio": port_names} # Match other plots
    )
    
    fig.update_layout(
        yaxis_tickformat='.0%',
        yaxis_title='Percent of Total Risk',
        xaxis_title=None,
        template='plotly_white',
        barmode='stack'
    )
    return fig


# --- Rebalancing Specific Plots ---

def plot_rebalancing_history(weight_history_df, title="Portfolio Weight Evolution (Rebalancing History)"):
    """
    Plots a stacked area chart showing how asset weights change over time
    due to price drift and rebalancing.
    """
    logger.debug("Plotting Rebalancing History...")
    
    # Sort columns by average weight to make the chart cleaner
    avg_weights = weight_history_df.mean().sort_values(ascending=False)
    sorted_cols = avg_weights.index.tolist()
    df_sorted = weight_history_df[sorted_cols]
    
    fig = go.Figure()
    
    for col in sorted_cols:
        fig.add_trace(go.Scatter(
            x=df_sorted.index,
            y=df_sorted[col],
            name=col,
            mode='lines',
            stackgroup='one', # This makes it a stacked area chart
            line=dict(width=0.5),
            hovertemplate='%{y:.2%}'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Weight',
        yaxis_tickformat='.0%',
        hovermode='x unified',
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80)
    )
    return fig


# --- Black-Litterman Specific Plots ---

def plot_bl_return_comparison(equilibrium_returns, posterior_returns, view_dict=None):
    """
    Grouped bar chart: Equilibrium Returns vs BL Posterior Returns for each asset.
    Assets with active views are highlighted with separate legend entries.
    """
    logger.debug("Plotting BL Return Comparison...")
    
    tickers = list(equilibrium_returns.index)
    eq_vals = equilibrium_returns.values
    post_vals = posterior_returns.reindex(equilibrium_returns.index).values
    
    # Sort by absolute shift so biggest impacts are most visible
    shifts = post_vals - eq_vals
    sort_idx = np.argsort(np.abs(shifts))[::-1]
    tickers_sorted = [tickers[i] for i in sort_idx]
    eq_sorted = eq_vals[sort_idx]
    post_sorted = post_vals[sort_idx]
    
    view_tickers = set(view_dict.keys()) if view_dict else set()
    
    # Split into view vs non-view indices for separate legend entries
    view_x, view_y, view_text = [], [], []
    noview_x, noview_y, noview_text = [], [], []
    for t, p in zip(tickers_sorted, post_sorted):
        if t in view_tickers:
            view_x.append(t)
            view_y.append(p)
            view_text.append(f"{p:.1%}")
        else:
            noview_x.append(t)
            noview_y.append(p)
            noview_text.append(f"{p:.1%}")
    
    fig = go.Figure()
    
    # Equilibrium bars
    fig.add_trace(go.Bar(
        name='Equilibrium (Market Implied)',
        x=tickers_sorted, y=eq_sorted,
        marker_color='rgba(99, 110, 250, 0.6)',
        text=[f"{v:.1%}" for v in eq_sorted],
        textposition='outside'
    ))
    
    # Posterior bars for assets WITH views (red)
    if view_x:
        fig.add_trace(go.Bar(
            name='BL Posterior (With View)',
            x=view_x, y=view_y,
            marker_color='rgba(255, 107, 107, 0.85)',
            text=view_text,
            textposition='outside'
        ))
    
    # Posterior bars for assets WITHOUT views (green)
    if noview_x:
        fig.add_trace(go.Bar(
            name='BL Posterior (No View)',
            x=noview_x, y=noview_y,
            marker_color='rgba(0, 204, 150, 0.7)',
            text=noview_text,
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Black-Litterman: Equilibrium vs Posterior Expected Returns',
        xaxis_title='Asset',
        yaxis_title='Annualized Expected Return',
        yaxis_tickformat='.0%',
        barmode='group',
        template='plotly_white',
        xaxis_tickangle=-45,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(b=120),
    )
    return fig


def plot_bl_view_impact(equilibrium_returns, posterior_returns, view_dict):
    """
    Horizontal bar chart showing the shift (posterior - equilibrium) for each asset.
    Only shows assets with material shifts. View assets are labeled.
    """
    logger.debug("Plotting BL View Impact...")
    
    tickers = list(equilibrium_returns.index)
    shifts = posterior_returns.reindex(equilibrium_returns.index).values - equilibrium_returns.values
    
    # Sort by shift magnitude
    sort_idx = np.argsort(shifts)
    tickers_sorted = [tickers[i] for i in sort_idx]
    shifts_sorted = shifts[sort_idx]
    
    view_tickers = set(view_dict.keys()) if view_dict else set()
    
    colors = []
    for i, t in enumerate(tickers_sorted):
        if t in view_tickers:
            colors.append('rgba(255, 107, 107, 0.9)' if shifts_sorted[i] >= 0 else 'rgba(255, 71, 87, 0.9)')
        else:
            colors.append('rgba(0, 204, 150, 0.6)' if shifts_sorted[i] >= 0 else 'rgba(99, 110, 250, 0.6)')
    
    # Add view annotation
    labels = []
    for t in tickers_sorted:
        if t in view_tickers:
            labels.append(f"{t} ★ (View: {view_dict[t]:+.0%})")
        else:
            labels.append(t)
    
    fig = go.Figure(go.Bar(
        y=labels,
        x=shifts_sorted,
        orientation='h',
        marker_color=colors,
        text=[f"{s:+.2%}" for s in shifts_sorted],
        textposition='outside'
    ))
    
    fig.add_vline(x=0, line_width=2, line_color="gray")
    
    fig.update_layout(
        title='Black-Litterman: View Impact on Expected Returns',
        xaxis_title='Shift from Equilibrium (Posterior − Equilibrium)',
        xaxis_tickformat='.1%',
        template='plotly_white',
        height=max(400, len(tickers) * 25),
        annotations=[dict(
            text="★ = active investor view  |  Positive = BL raised expected return",
            xref="paper", yref="paper", x=0.5, y=-0.08,
            showarrow=False, font=dict(size=11, color="gray")
        )]
    )
    return fig


def plot_bl_weights_comparison(market_weights, bl_weights_dict, view_dict=None):
    """
    Grouped bar chart: Market-Cap Weights vs BL Optimized Weights.
    """
    logger.debug("Plotting BL Weights Comparison...")
    
    tickers = list(bl_weights_dict.keys())
    bl_vals = np.array(list(bl_weights_dict.values()))
    mkt_vals = market_weights.reindex(tickers).fillna(0).values
    
    # Sort by BL weight (descending)
    sort_idx = np.argsort(bl_vals)[::-1]
    tickers_sorted = [tickers[i] for i in sort_idx]
    bl_sorted = bl_vals[sort_idx]
    mkt_sorted = mkt_vals[sort_idx]
    
    # Only show assets with > 0.5% weight in either
    mask = (bl_sorted > 0.005) | (mkt_sorted > 0.005)
    tickers_filtered = [t for t, m in zip(tickers_sorted, mask) if m]
    bl_filtered = bl_sorted[mask]
    mkt_filtered = mkt_sorted[mask]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Market-Cap Weights',
        x=tickers_filtered, y=mkt_filtered,
        marker_color='rgba(99, 110, 250, 0.6)',
        text=[f"{v:.1%}" for v in mkt_filtered],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='BL Optimized Weights',
        x=tickers_filtered, y=bl_filtered,
        marker_color='rgba(0, 204, 150, 0.8)',
        text=[f"{v:.1%}" for v in bl_filtered],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Black-Litterman: Market-Cap Weights vs Optimized Weights',
        xaxis_title='Asset',
        yaxis_title='Portfolio Weight',
        yaxis_tickformat='.0%',
        barmode='group',
        template='plotly_white',
    )
    return fig