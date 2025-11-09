import pandas as pd
import numpy as np
import traceback
from .data import get_data
from .metrics import calculate_metrics
from .html_builder import generate_html_report

# Import from new utility modules
from .opt_core import (
    get_risk_free_rate,
    get_optimization_inputs,
    get_portfolio_stats,
    objective_neg_sharpe,
    objective_min_variance,
    find_optimal_portfolio,
    calculate_efficient_frontier_curve,
    calculate_rolling_returns,
    get_portfolio_price,
    build_constraints
)
from .opt_plotting import (
    plot_efficient_frontier,
    plot_correlation_heatmap,
    plot_cumulative_comparison,
    plot_drawdown_comparison,
    plot_rolling_sharpe,
    plot_composition_pies,
    plot_risk_contribution,
    plot_monthly_heatmaps,
    plot_sector_allocation_pies,  
    plot_sector_risk_contribution 
)

def create_optimization_report(portfolio_dict, benchmark_ticker, start_date, end_date, 
                               filename="optimization_report.html", 
                               risk_free_rate="auto",
                               display_names=None,
                               sector_map=None,
                               sector_caps=None,
                               sector_mins=None):
    """
    Runs the full portfolio optimization and generates an HTML report.
    """
    print("--- Starting Portfolio Optimization ---")
    
    try:
        if isinstance(risk_free_rate, str) and risk_free_rate.lower() == 'auto':
            rfr = get_risk_free_rate()
        elif isinstance(risk_free_rate, (int, float)):
            rfr = risk_free_rate
        else: rfr = 0.02
        
        tickers = list(portfolio_dict.keys()) # <-- Raw tickers
        num_assets = len(tickers)
        print(f"--- Using Risk-Free Rate: {rfr:.2%} ---")
        
        all_tickers = tickers + [benchmark_ticker]
        price_data = get_data(all_tickers, start_date, end_date)
        if price_data is None: raise ValueError("Could not fetch price data.")
        
        friendly_benchmark = benchmark_ticker
        friendly_tickers = tickers
        friendly_sector_map = None
        
        if display_names:
            print("Applying display names...")
            price_data.rename(columns=display_names, inplace=True)
            friendly_benchmark = display_names.get(benchmark_ticker, benchmark_ticker)
            friendly_tickers = [display_names.get(t, t) for t in tickers]
            if sector_map:
                # Create the friendly_sector_map for plotting
                friendly_sector_map = {display_names.get(t, t): s for t, s in sector_map.items()}
        
        mean_returns, cov_matrix, log_returns = get_optimization_inputs(price_data[friendly_tickers])
        
        # --- 4. Define Constraints ---
        bounds_uncon = tuple((0, 1) for _ in range(num_assets))
        cons_uncon = build_constraints(num_assets, tickers)
        bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
        cons_bal = build_constraints(num_assets, tickers)
        cons_sector = build_constraints(num_assets, tickers, sector_map, sector_caps, sector_mins)
        
        # --- 5. Optimize ---
        print("Optimizing for Minimum Volatility...")
        min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, mean_returns, cov_matrix, bounds_uncon, cons_uncon, rfr)
        min_vol_weights_dict = {t: w for t, w in zip(friendly_tickers, min_vol_weights_arr)}

        print("Optimizing for Maximum Sharpe Ratio...")
        max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, mean_returns, cov_matrix, bounds_uncon, cons_uncon, rfr)
        max_sharpe_weights_dict = {t: w for t, w in zip(friendly_tickers, max_sharpe_weights_arr)}

        print("Optimizing for Balanced Portfolio (Max Sharpe w/ 40% cap)...")
        bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, mean_returns, cov_matrix, bounds_bal, cons_bal, rfr)
        bal_weights_dict = {t: w for t, w in zip(friendly_tickers, bal_weights_arr)}
        
        print("Calculating Equal Weight Portfolio...")
        equal_weights_arr = np.array([1./num_assets] * num_assets)
        equal_weights_dict = {t: w for t, w in zip(friendly_tickers, equal_weights_arr)}

        # --- 6. Evaluate Performance ---
        print("Evaluating optimized portfolios...")
        
        eval_data = (price_data[[friendly_benchmark]] / price_data[[friendly_benchmark]].iloc[0]).copy()
        
        eval_data['Min Vol Portfolio'] = get_portfolio_price(price_data[friendly_tickers], min_vol_weights_dict)
        eval_data['Max Sharpe Portfolio'] = get_portfolio_price(price_data[friendly_tickers], max_sharpe_weights_dict)
        eval_data['Balanced Portfolio'] = get_portfolio_price(price_data[friendly_tickers], bal_weights_dict)
        eval_data['Equal Wt Portfolio'] = get_portfolio_price(price_data[friendly_tickers], equal_weights_dict)

        min_vol_metrics, _ = calculate_metrics(eval_data, 'Min Vol Portfolio', friendly_benchmark, rfr)
        max_sharpe_metrics, _ = calculate_metrics(eval_data, 'Max Sharpe Portfolio', friendly_benchmark, rfr)
        balanced_metrics, _ = calculate_metrics(eval_data, 'Balanced Portfolio', friendly_benchmark, rfr)
        equal_wt_metrics, _ = calculate_metrics(eval_data, 'Equal Wt Portfolio', friendly_benchmark, rfr)

        # --- 7. Store results ---
        optimal_portfolios = {
            "Equal Weight (Baseline)": {"weights_arr": equal_weights_arr, "weights_dict": equal_weights_dict, "metrics": equal_wt_metrics, "color": "blue"},
            "Minimum Volatility": {"weights_arr": min_vol_weights_arr, "weights_dict": min_vol_weights_dict, "metrics": min_vol_metrics, "color": "green"},
            "Balanced (40% Cap)": {"weights_arr": bal_weights_arr, "weights_dict": bal_weights_dict, "metrics": balanced_metrics, "color": "orange"},
            "Max Sharpe (Unconstrained)": {"weights_arr": max_sharpe_weights_arr, "weights_dict": max_sharpe_weights_dict, "metrics": max_sharpe_metrics, "color": "red"}
        }
        
        if sector_map and (sector_caps or sector_mins):
            print("Optimizing for Sector Balanced Portfolio...")
            sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, mean_returns, cov_matrix, bounds_uncon, cons_sector, rfr)
            sec_bal_weights_dict = {t: w for t, w in zip(friendly_tickers, sec_bal_weights_arr)}
            eval_data['Sector Balanced'] = get_portfolio_price(price_data[friendly_tickers], sec_bal_weights_dict)
            sec_bal_metrics, _ = calculate_metrics(eval_data, 'Sector Balanced', friendly_benchmark, rfr)
            optimal_portfolios["Sector Balanced"] = {"weights_arr": sec_bal_weights_arr, "weights_dict": sec_bal_weights_dict, "metrics": sec_bal_metrics, "color": "purple"}

        # --- 8. Calculate Data for Tables/Plots ---
        rolling_returns_df = calculate_rolling_returns(eval_data)
        rolling_returns_html = rolling_returns_df.to_html(classes='metrics-table')
        
        benchmark_log_return = np.log(price_data[friendly_benchmark] / price_data[friendly_benchmark].shift(1)).dropna()
        aligned_log_returns, aligned_benchmark = log_returns.align(benchmark_log_return, join='inner', axis=0)
        asset_corr_df = aligned_log_returns.corrwith(aligned_benchmark).to_frame(name='Correlation')
        asset_corr_html = asset_corr_df.map(lambda x: f"{x:.2f}").to_html(classes='metrics-table')
        
        daily_returns_df = eval_data.pct_change().dropna()
        excess_returns_df = daily_returns_df - (rfr / 252)
        rolling_sharpe_df = (excess_returns_df.rolling(60).mean() * 252) / (excess_returns_df.rolling(60).std() * np.sqrt(252))
        
        cumulative_returns_df = eval_data
        drawdown_df = pd.DataFrame()
        for col in cumulative_returns_df.columns:
            peak = cumulative_returns_df[col].cummax()
            drawdown_df[col] = (cumulative_returns_df[col] - peak) / peak
            
        frontier_curve = calculate_efficient_frontier_curve(mean_returns, cov_matrix)

        # --- 9. Build Report Sections ---
        sidebar_items = [
            {"title": "Rolling Returns", "type": "table_html", "data": rolling_returns_html},
            {"title": "Asset-Benchmark Correlation", "type": "table_html", "data": asset_corr_html}
        ]
        
        main_content = [
            {"title": "Strategy Compositions (by Asset)", "type": "plot", "data": plot_composition_pies(optimal_portfolios)},
            {"title": "Strategy Compositions (by Sector)", "type": "plot", "data": plot_sector_allocation_pies(optimal_portfolios, friendly_sector_map)},
            {"title": "Portfolio Risk Contribution (by Asset)", "type": "plot", "data": plot_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, friendly_tickers, rfr)},
            {"title": "Portfolio Risk Contribution (by Sector)", "type": "plot", "data": plot_sector_risk_contribution(optimal_portfolios, mean_returns, cov_matrix, friendly_tickers, friendly_sector_map, rfr)},
            {"title": "Strategy Cumulative Returns", "type": "plot", "data": plot_cumulative_comparison(cumulative_returns_df, friendly_benchmark)},
            {"title": "Strategy Drawdown", "type": "plot", "data": plot_drawdown_comparison(drawdown_df, friendly_benchmark)},
            {"title": "Strategy Rolling Sharpe Ratio", "type": "plot", "data": plot_rolling_sharpe(rolling_sharpe_df, friendly_benchmark)},
            {"title": "Strategy Monthly Returns Heatmap", "type": "plot", "data": plot_monthly_heatmaps(eval_data, friendly_benchmark)},
            {"title": "Efficient Frontier", "type": "plot", "data": plot_efficient_frontier(mean_returns, cov_matrix, optimal_portfolios, frontier_curve, rfr)},
            {"title": "Asset Correlation Heatmap", "type": "plot", "data": plot_correlation_heatmap(log_returns)},
            {"title": "Strategy Metrics", "type": "metrics_grid", "data": optimal_portfolios}
        ]
        
        sections = [{
            "title": "Optimization Analysis",
            "description": f"Optimization performed on the period {start_date} to {end_date}.",
            "sidebar": sidebar_items,
            "main_content": main_content
        }]
        
        # 10. Generate Report
        generate_html_report(sections, title="Portfolio Optimization Report", filename=filename)
        
        print(f"--- Optimization Report Generated: {filename} ---")
        
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        traceback.print_exc()