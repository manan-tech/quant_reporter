import pandas as pd
import numpy as np
from scipy import stats

def calculate_max_drawdown(cumulative_returns):
    """Calculates the max drawdown from a cumulative returns series."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.02):
    """Calculates the annualized Sortino Ratio."""
    annualized_mean_return = daily_returns.mean() * 252
    
    downside_returns = daily_returns[daily_returns < 0]
    annualized_downside_std = downside_returns.std() * np.sqrt(252)
    
    if annualized_downside_std == 0:
        return np.nan
        
    sortino_ratio = (annualized_mean_return - risk_free_rate) / annualized_downside_std
    return sortino_ratio

def calculate_var_cvar(daily_returns, confidence_level=0.95):
    """
    Calculates the historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    if daily_returns.empty:
        return np.nan, np.nan
    var = daily_returns.quantile(1 - confidence_level)
    cvar = daily_returns[daily_returns <= var].mean()
    
    return var, cvar

def calculate_metrics(data, asset_col, benchmark_col, risk_free_rate=0.02):
    """
    Calculates key performance and risk metrics for a single asset against a benchmark.
    """
    if asset_col not in data.columns or benchmark_col not in data.columns:
        raise ValueError(f"Columns '{asset_col}' or '{benchmark_col}' not in DataFrame.")
        
    # --- 1. Prepare Returns Data ---
    asset_prices = data[asset_col]
    benchmark_prices = data[benchmark_col]

    asset_returns_raw = asset_prices.pct_change()
    benchmark_returns_raw = benchmark_prices.pct_change()
    
    daily_returns_df = pd.DataFrame({
        'Asset': asset_returns_raw,
        'Benchmark': benchmark_returns_raw
    }).dropna()
    
    daily_returns = daily_returns_df['Asset']
    
    cumulative_returns_df = (1 + daily_returns_df).cumprod()
    cumulative_returns = cumulative_returns_df['Asset']
    
    log_returns_data = data[[asset_col, benchmark_col]].copy()
    log_returns = np.log(log_returns_data / log_returns_data.shift(1)).dropna()
    log_returns.columns = ['Asset', 'Benchmark'] # Standardize

    monthly_returns = asset_prices.resample('ME').last().pct_change().dropna()
    
    # --- 2. Calculations ---
    asset_vol = daily_returns.std() * np.sqrt(252)
    bench_vol = daily_returns_df['Benchmark'].std() * np.sqrt(252)
    
    if daily_returns_df['Benchmark'].std() == 0 or daily_returns.std() == 0:
        beta = 0.0
        alpha = 0.0
    else:
        lin_reg = stats.linregress(daily_returns_df['Benchmark'], daily_returns)
        beta = lin_reg.slope
        alpha = lin_reg.intercept * 252 # Annualize

    total_days = (asset_prices.index[-1] - asset_prices.index[0]).days
    years = max(total_days / 365.25, 1)
    
    cagr_asset = (asset_prices.iloc[-1] / asset_prices.iloc[0])**(1/years) - 1
    cagr_bench = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0])**(1/years) - 1
    
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
    sortino_ratio = calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    
    calmar_ratio = cagr_asset / abs(max_drawdown) if max_drawdown != 0 else np.nan
    skew = daily_returns.skew()
    kurtosis = daily_returns.kurtosis()
    
    var_95, cvar_95 = calculate_var_cvar(daily_returns, confidence_level=0.95)
    
    rolling_sharpe = (excess_returns.rolling(window=60).mean() * 252) / (excess_returns.rolling(window=60).std() * np.sqrt(252))

    yearly_prices = data[[asset_col, benchmark_col]].resample('YE').last()
    yearly_returns = yearly_prices.pct_change().dropna()
    yearly_returns.columns = ['Asset', 'Benchmark'] # Standardize
    
    metrics = {
        "CAGR (Asset)": f"{cagr_asset:.2%}",
        "CAGR (Benchmark)": f"{cagr_bench:.2%}",
        "Annualized Volatility (Asset)": f"{asset_vol:.2%}",
        "Annualized Volatility (Bench)": f"{bench_vol:.2%}",
        "Sharpe Ratio (Asset)": f"{sharpe_ratio:.2f}",
        "Sortino Ratio (Asset)": f"{sortino_ratio:.2f}",
        "Calmar Ratio (Asset)": f"{calmar_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Beta (vs Benchmark)": f"{beta:.2f}",
        "Alpha (Annualized)": f"{alpha:.2%}",
        "Skew": f"{skew:.2f}",
        "Kurtosis": f"{kurtosis:.2f}",
        "VaR (95%)": f"{var_95:.2%}",
        "CVaR (95%)": f"{cvar_95:.2%}"
    }
    
    plot_data = {
        "daily_returns": daily_returns_df,
        "cumulative_returns": cumulative_returns_df,
        "monthly_returns": monthly_returns,
        "log_returns": log_returns,
        "rolling_sharpe": rolling_sharpe,
        "yearly_returns": yearly_returns
    }
    
    return metrics, plot_data