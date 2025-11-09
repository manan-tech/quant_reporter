import pandas as pd
from .data import get_data
from .metrics import calculate_metrics
from .plotting import (
    plot_cumulative_returns, 
    plot_rolling_volatility, 
    plot_regression, 
    plot_rolling_sharpe, 
    plot_monthly_distribution,
    plot_yearly_returns
)
from .opt_plotting import plot_portfolio_vs_constituents
from .html_builder import generate_html_report
from .opt_core import get_portfolio_price, calculate_rolling_returns

def create_full_report(assets, benchmark_ticker, start_date, end_date, 
                       filename="report.html", 
                       display_names=None,
                       risk_free_rate=0.02):
    """
    Runs the full data-to-HTML report generation pipeline.
    """
    print(f"--- Starting Full Portfolio Report ---")
    
    asset_col_name = ""
    is_portfolio = False
    portfolio_dict_raw = {}

    if isinstance(assets, str):
        is_portfolio = False
        asset_col_name = assets
        all_tickers = [asset_col_name, benchmark_ticker]
    elif isinstance(assets, dict):
        is_portfolio = True
        asset_col_name = "Portfolio"
        portfolio_dict_raw = assets
        all_tickers = list(portfolio_dict_raw.keys()) + [benchmark_ticker]
    else:
        print("Error: 'assets' parameter must be a string or a dict")
        return

    price_data = get_data(all_tickers, start_date, end_date)
    if price_data is None: return

    friendly_benchmark = benchmark_ticker
    friendly_asset_col = asset_col_name
    friendly_tickers = list(portfolio_dict_raw.keys())
    
    if display_names:
        price_data.rename(columns=display_names, inplace=True)
        friendly_benchmark = display_names.get(benchmark_ticker, benchmark_ticker)
        friendly_tickers = [display_names.get(t, t) for t in friendly_tickers]
        if is_portfolio:
            friendly_asset_col = "Portfolio"
        else:
            friendly_asset_col = display_names.get(asset_col_name, asset_col_name)

    if is_portfolio:
        user_friendly_weights = {display_names.get(k, k): v for k, v in portfolio_dict_raw.items()} if display_names else portfolio_dict_raw
        if not abs(sum(user_friendly_weights.values()) - 1.0) > 1e-5: # Check if sum is not 1
             print(f"Warning: Portfolio weights sum to {sum(user_friendly_weights.values())}, not 1.0. Proceeding anyway.")
        
        price_data[friendly_asset_col] = get_portfolio_price(price_data[friendly_tickers], user_friendly_weights)

    try:
        metrics, plot_data = calculate_metrics(price_data, friendly_asset_col, friendly_benchmark, risk_free_rate)
    except Exception as e:
        print(f"Error calculating metrics for {friendly_asset_col}: {e}")
        return

    # --- Build Report Sections ---
    sidebar_items = [
        {"title": f"Key Metrics ({friendly_asset_col})", "type": "metrics", "data": metrics}
    ]
    
    main_content = [
        {"title": "Cumulative Returns", "type": "plot", "data": plot_cumulative_returns(plot_data, friendly_asset_col, friendly_benchmark)},
        {"title": "Annual Returns", "type": "plot", "data": plot_yearly_returns(plot_data, friendly_asset_col, friendly_benchmark)},
        {"title": "Rolling Volatility", "type": "plot", "data": plot_rolling_volatility(plot_data, friendly_asset_col, friendly_benchmark)},
        {"title": "Rolling Sharpe Ratio", "type": "plot", "data": plot_rolling_sharpe(plot_data, friendly_asset_col)},
        {"title": "Alpha/Beta Regression", "type": "plot", "data": plot_regression(plot_data, metrics, friendly_asset_col, friendly_benchmark)},
        {"title": "Monthly Returns Distribution", "type": "plot", "data": plot_monthly_distribution(plot_data, friendly_asset_col)},
    ]
    
    if is_portfolio:
        tickers_to_plot = [friendly_asset_col] + friendly_tickers + [friendly_benchmark]
        # Filter out any tickers not in price_data (e.g. SPY if it was in the portfolio)
        tickers_to_plot = [t for t in tickers_to_plot if t in price_data.columns]
        
        daily_returns = price_data[tickers_to_plot].pct_change().dropna()
        all_cumulative_returns = (1 + daily_returns).cumprod()
        
        main_content.insert(0, {
            "title": "Portfolio vs. Constituent Performance",
            "type": "plot",
            "data": plot_portfolio_vs_constituents(all_cumulative_returns)
        })

    sections = [{
        "title": "Full Period Analysis",
        "description": f"Analysis of the asset '{friendly_asset_col}' against the benchmark '{friendly_benchmark}' from {start_date} to {end_date}.",
        "sidebar": sidebar_items,
        "main_content": main_content
    }]
    
    generate_html_report(sections, title=f"Portfolio Report: {friendly_asset_col}", filename=filename)
    print(f"--- Report for {friendly_asset_col} complete. ---")