import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import numpy as np

def plot_cumulative_returns(plot_data, asset_ticker, benchmark_ticker):
    cumulative_returns = plot_data['cumulative_returns']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns['Asset'], 
        name=f'{asset_ticker} Cumulative Returns'
    ))
    fig.add_trace(go.Scatter(
        x=cumulative_returns.index, 
        y=cumulative_returns['Benchmark'], 
        name=f'{benchmark_ticker} Cumulative Returns'
    ))
    fig.update_layout(title='Performance vs. Benchmark', hovermode='x unified', template='plotly_white')
    return fig

def plot_rolling_volatility(plot_data, asset_ticker, benchmark_ticker):
    daily_returns = plot_data['daily_returns']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_returns.index, 
        y=(daily_returns['Asset'].rolling(30).std() * 252**0.5), 
        name=f'{asset_ticker} 30-Day Rolling Vol'
    ))
    fig.add_trace(go.Scatter(
        x=daily_returns.index, 
        y=(daily_returns['Benchmark'].rolling(30).std() * 252**0.5), 
        name=f'{benchmark_ticker} 30-Day Rolling Vol'
    ))
    fig.update_layout(title='Rolling Volatility (Annualized)', hovermode='x unified', template='plotly_white')
    return fig

def plot_regression(plot_data, metrics, asset_ticker, benchmark_ticker):
    daily_returns = plot_data['daily_returns']
    fig = px.scatter(
        daily_returns, 
        x='Benchmark', 
        y='Asset', 
        title=f'{asset_ticker} vs {benchmark_ticker} Daily Returns (Beta)',
        opacity=0.5,
        trendline='ols'
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_rolling_sharpe(plot_data, asset_ticker):
    rolling_sharpe = plot_data['rolling_sharpe']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, 
        y=rolling_sharpe, 
        name='Rolling Sharpe'
    ))
    fig.update_layout(title=f'{asset_ticker} 60-Day Rolling Sharpe Ratio', hovermode='x unified', template='plotly_white')
    return fig

def plot_monthly_distribution(plot_data, asset_ticker):
    monthly_returns = plot_data['monthly_returns'].dropna()
    if monthly_returns.empty:
        return go.Figure().update_layout(title=f'{asset_ticker} Monthly Returns Distribution (No Data)')
    
    # Clip outliers for better visualization
    q_01 = monthly_returns.quantile(0.01)
    q_99 = monthly_returns.quantile(0.99)
    plot_data_filtered = monthly_returns[(monthly_returns >= q_01) & (monthly_returns <= q_99)]
        
    fig = px.histogram(
        plot_data_filtered, 
        x=plot_data_filtered,
        title=f'{asset_ticker} Monthly Returns Distribution (Outliers Clipped)'
    )
    
    mean_val = monthly_returns.mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                  annotation_text=f'Mean: {mean_val:.2%}', 
                  annotation_position="top right")
    
    fig.update_layout(
        xaxis_title='Monthly Return',
        yaxis_title='Frequency (Count of Months)',
        template='plotly_white'
    )
    return fig

def plot_yearly_returns(plot_data, asset_ticker, benchmark_ticker):
    yearly_returns = plot_data['yearly_returns']
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_returns.index.year,
        y=yearly_returns['Asset'],
        name=asset_ticker,
    ))
    fig.add_trace(go.Bar(
        x=yearly_returns.index.year,
        y=yearly_returns['Benchmark'],
        name=benchmark_ticker
    ))
    fig.update_layout(title='Annual Returns Comparison', barmode='group', hovermode='x unified', template='plotly_white')
    return fig