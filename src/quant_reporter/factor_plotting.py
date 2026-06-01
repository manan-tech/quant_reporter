import plotly.graph_objects as go

def plot_factor_portfolio_growth(portfolio_returns, portfolio_name):
    cumulative_returns = (1 + portfolio_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index,
        y=cumulative_returns.values,
        mode='lines',
        name=portfolio_name,
        line=dict(color='#1976D2', width=2)
    ))
    fig.update_layout(
        title="Cumulative Portfolio Growth",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (Growth of $1)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_factor_loadings(regression_results):
    fig = go.Figure()
    beta_labels = {
        'Mkt-RF': 'Market',
        'SMB': 'Size (SMB)',
        'HML': 'Value (HML)'
    }

    for factor_name in regression_results['betas'].index:
        label = beta_labels.get(factor_name, factor_name)
        beta_value = regression_results['betas'][factor_name]
        pvalue = regression_results['factor_pvalues'][factor_name]

        # Color based on significance
        if pvalue < 0.01:
            color = '#2E7D32'  # Green - highly significant
        elif pvalue < 0.05:
            color = '#1976D2'  # Blue - significant
        elif pvalue < 0.10:
            color = '#F57C00'  # Orange - marginally significant
        else:
            color = '#757575'  # Gray - not significant

        fig.add_trace(go.Bar(
            name=label,
            x=[label],
            y=[beta_value],
            marker_color=color,
            text=f'{beta_value:.3f}<br>p={pvalue:.3f}',
            textposition='outside'
        ))

    fig.update_layout(
        title="Factor Loadings (Betas) and Significance",
        yaxis_title="Beta Coefficient",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_cumulative_contributions(attribution):
    # Only keep the columns that actually exist in the dataframe
    expected_cols = ['Mkt-RF_Contribution', 'SMB_Contribution', 'HML_Contribution', 'Alpha_Contribution']
    available_cols = [col for col in expected_cols if col in attribution.columns]
    
    attribution_cum = (1 + attribution[available_cols]).cumprod() - 1

    fig = go.Figure()

    colors = {
        'Mkt-RF_Contribution': '#1976D2',
        'SMB_Contribution': '#7B1FA2',
        'HML_Contribution': '#F57C00',
        'Alpha_Contribution': '#2E7D32'
    }

    labels = {
        'Mkt-RF_Contribution': 'Market Factor',
        'SMB_Contribution': 'Size Factor',
        'HML_Contribution': 'Value Factor',
        'Alpha_Contribution': 'Alpha'
    }

    for col in available_cols:
        fig.add_trace(go.Scatter(
            x=attribution_cum.index,
            y=attribution_cum[col],
            mode='lines',
            name=labels.get(col, col),
            line=dict(color=colors.get(col, '#000000'), width=2),
            stackgroup='one'
        ))

    fig.update_layout(
        title="Cumulative Factor Contributions",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_brinson_attribution(brinson_results):
    fig = go.Figure()
    sectors = [idx for idx in brinson_results.index if idx != 'Total']

    fig.add_trace(go.Bar(
        name='Allocation Effect',
        x=sectors,
        y=brinson_results.loc[sectors, 'Allocation_Effect'] * 100,
        marker_color='#1976D2'
    ))

    fig.add_trace(go.Bar(
        name='Selection Effect',
        x=sectors,
        y=brinson_results.loc[sectors, 'Selection_Effect'] * 100,
        marker_color='#7B1FA2'
    ))

    fig.add_trace(go.Bar(
        name='Interaction Effect',
        x=sectors,
        y=brinson_results.loc[sectors, 'Interaction_Effect'] * 100,
        marker_color='#F57C00'
    ))

    fig.update_layout(
        title="Brinson Attribution by Sector (Basis Points)",
        xaxis_title="Sector",
        yaxis_title="Effect (bps)",
        barmode='stack',
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def plot_asset_allocation(portfolio_weights):
    asset_labels = list(portfolio_weights.keys())
    asset_weights = list(portfolio_weights.values())

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=asset_labels,
        values=asset_weights,
        name="Asset Allocation",
        marker=dict(
            colors=[
                '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5'
            ]
        ),
        hovertemplate='%{label}<br>%{value:.1%}<extra></extra>',
        textinfo='percent+label'
    ))

    fig.update_layout(
        title="Portfolio Asset Allocation",
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    return fig

def plot_sector_allocation(sector_weights):
    sector_labels = list(sector_weights.keys())
    sector_values = list(sector_weights.values())

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=sector_labels,
        values=sector_values,
        name="Sector Allocation",
        marker=dict(
            colors=[
                '#17becf', '#9edae5', '#bcbd22', '#dbdb8d', '#1f77b4',
                '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a'
            ]
        ),
        hovertemplate='%{label}<br>%{value:.1%}<extra></extra>',
        textinfo='percent+label'
    ))

    fig.update_layout(
        title="Portfolio Sector Allocation",
        height=500,
        showlegend=False,
        template="plotly_white"
    )
    return fig


def plot_rolling_exposures(rolling_betas):
    """
    Plots rolling factor beta exposures over time.
    rolling_betas: DataFrame with DatetimeIndex, columns = factor names like 'Mkt-RF', 'SMB', 'HML'.
    """
    colors = ['#1976D2', '#7B1FA2', '#F57C00', '#2E7D32', '#D32F2F']
    fig = go.Figure()
    for i, col in enumerate(rolling_betas.columns):
        fig.add_trace(go.Scatter(
            x=rolling_betas.index,
            y=rolling_betas[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=1.5)
        ))
    fig.update_layout(
        title="Rolling Factor Exposures (6-Month Window)",
        xaxis_title="Date",
        yaxis_title="Beta",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig
