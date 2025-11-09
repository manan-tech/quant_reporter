import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta

# --- Import from our own package ---
from .data import get_data
from .metrics import calculate_metrics
from .html_builder import generate_html_report

# --- Import from main report plotting ---
from .plotting import (
    plot_cumulative_returns,
    plot_regression
)

# --- Import from new utility modules ---
from .opt_core import (
    get_risk_free_rate,
    get_optimization_inputs,
    find_optimal_portfolio,
    objective_neg_sharpe,
    objective_min_variance,
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
    plot_sector_allocation_pies,  
    plot_sector_risk_contribution 
)
# --- End Imports ---


def create_combined_report(portfolio_dict, benchmark_ticker,
                           train_start, train_end,
                           filename="Combined_Report.html",
                           risk_free_rate="auto",
                           display_names=None,
                           sector_map=None,
                           sector_caps=None,
                           sector_mins=None): 
    """
    Generates a single, combined HTML report for portfolio analysis,
    optimization, and walk-forward validation.
    """
    print("--- 1. INITIATING COMBINED REPORT ---")
    
    try:
        # --- A. Date & Name Setup ---
        test_start_dt = (pd.to_datetime(train_end) + timedelta(days=1))
        test_end_dt = (datetime.now() - timedelta(days=1))
        test_start = test_start_dt.strftime('%Y-%m-%d')
        test_end = test_end_dt.strftime('%Y-%m-%d')
        full_start = train_start
        full_end = test_end
        
        print(f"Full Period:  {full_start} to {full_end}")
        print(f"Train Period: {train_start} to {train_end}")
        print(f"Test Period:  {test_start} to {test_end}")

        if isinstance(risk_free_rate, str) and risk_free_rate.lower() == 'auto':
            rfr = get_risk_free_rate()
        elif isinstance(risk_free_rate, (int, float)):
            rfr = risk_free_rate
        else: rfr = 0.02
        print(f"Using Risk-Free Rate: {rfr:.2%} ---")

        tickers = list(portfolio_dict.keys())
        user_weights_dict_raw = portfolio_dict
        
        friendly_benchmark = display_names.get(benchmark_ticker, benchmark_ticker) if display_names else benchmark_ticker
        friendly_tickers = [display_names.get(t, t) for t in tickers] if display_names else tickers
        user_friendly_weights = {display_names.get(k, k): v for k, v in user_weights_dict_raw.items()} if display_names else user_weights_dict_raw

        friendly_sector_map = None
        if display_names and sector_map:
            friendly_sector_map = {display_names.get(t, t): s for t, s in sector_map.items()}
        elif sector_map: 
             friendly_sector_map = sector_map

        # --- B. Data Fetching (All 3 periods) ---
        print("\n--- 2. FETCHING ALL DATA ---")
        all_tickers = tickers + [benchmark_ticker]
        
        data_full = get_data(all_tickers, full_start, full_end)
        data_train = get_data(all_tickers, train_start, train_end)
        data_test = get_data(all_tickers, test_start, test_end)

        if data_full is None or data_train is None or data_test is None:
            raise ValueError("Failed to fetch data for one or more periods.")

        if display_names:
            data_full.rename(columns=display_names, inplace=True)
            data_train.rename(columns=display_names, inplace=True)
            data_test.rename(columns=display_names, inplace=True)

        # --- C. Run Portfolio Report (Full Period) ---
        print("\n--- 3. RUNNING USER PORTFOLIO REPORT (Full Period) ---")
        
        pr_eval_data = (data_full[[friendly_benchmark]] / data_full[[friendly_benchmark]].iloc[0]).copy()
        pr_eval_data['My Portfolio'] = get_portfolio_price(data_full[friendly_tickers], user_friendly_weights)
        pr_metrics, pr_plot_data = calculate_metrics(pr_eval_data, 'My Portfolio', friendly_benchmark, rfr)
        pr_plots = {
            "cumulative": plot_cumulative_returns(pr_plot_data, 'My Portfolio', friendly_benchmark),
            "regression": plot_regression(pr_plot_data, pr_metrics, 'My Portfolio', friendly_benchmark)
        }
        pr_rolling_returns_html = calculate_rolling_returns(pr_eval_data).to_html(classes='metrics-table')
        
        # --- D. Run Optimization & Validation (Train/Test Split) ---
        print("\n--- 4. RUNNING OPTIMIZATION & VALIDATION ---")
        
        val_results = _run_validation_logic(
            data_train, data_test, tickers, friendly_tickers, friendly_benchmark, user_friendly_weights, 
            rfr, sector_map, sector_caps, sector_mins, friendly_sector_map # <-- Pass maps
        )
        
        # --- E. Generate Final HTML ---
        print("\n--- 5. GENERATING COMBINED HTML ---")
        
        sections = [
            {
                "title": "1. User Portfolio (Full Period)",
                "description": f"Analysis of your user-defined portfolio mix ({pr_metrics.get('CAGR (Asset)', 'N/A')} CAGR) over the full historical period.",
                "sidebar": [
                    {"title": "User Portfolio Metrics (Full)", "type": "metrics", "data": pr_metrics},
                    {"title": "User Portfolio Rolling Returns (Full)", "type": "table_html", "data": pr_rolling_returns_html}
                ],
                "main_content": [
                    {"title": "Portfolio Cumulative Returns", "type": "plot", "data": pr_plots['cumulative'], "cdn_needed": True},
                    {"title": "Portfolio Alpha/Beta Regression", "type": "plot", "data": pr_plots['regression']}
                ]
            },
            {
                "title": "2. Optimization Analysis (Train Period)",
                "description": f"Analysis performed on the Training Data ({train_start} to {train_end}) to generate the optimal weights.",
                "sidebar": [
                    {"title": "Asset-Benchmark Correlation (Train)", "type": "table_html", "data": val_results["asset_corr_html"]}
                ],
                "main_content": [
                    {"title": "Strategy Compositions (by Asset)", "type": "plot", "data": val_results['optimization_plots']['pie_plot']},
                    {"title": "Strategy Compositions (by Sector)", "type": "plot", "data": val_results['optimization_plots']['sector_pie_plot']},
                    {"title": "Portfolio Risk Contribution (by Asset)", "type": "plot", "data": val_results['optimization_plots']['risk_contribution']},
                    {"title": "Portfolio Risk Contribution (by Sector)", "type": "plot", "data": val_results['optimization_plots']['sector_risk_contribution']},
                    {"title": "Strategy Rolling Sharpe Ratio (from Train)", "type": "plot", "data": val_results['optimization_plots']['rolling_sharpe_plot']},
                    {"title": "Efficient Frontier (from Train)", "type": "plot", "data": val_results['optimization_plots']['frontier']},
                    {"title": "Asset Correlation Heatmap (from Train)", "type": "plot", "data": val_results['optimization_plots']['heatmap']}
                ]
            },
            {
                "title": "3. Walk-Forward Validation (Test Period)",
                "description": f"Tests how portfolios optimized on training data would have performed in the Test Period ({test_start} to {test_end}).",
                "sidebar": [],
                "main_content": [
                    {"title": "In-Sample vs. Out-of-Sample Performance", "type": "table_html", "data": val_results["table_html"]},
                    {"title": "Out-of-Sample Cumulative Returns", "type": "plot", "data": val_results['validation_plots']['cumulative_plot']},
                    {"title": "Out-of-Sample Drawdown", "type": "plot", "data": val_results['validation_plots']['drawdown_plot']}
                ]
            }
        ]
        
        generate_html_report(sections, title="Combined Portfolio Report", filename=filename)
        
        print(f"--- Combined Report Generated: {filename} ---")
        
    except Exception as e:
        print(f"An error occurred during combined report generation: {e}")
        traceback.print_exc()

# --- 4. Helper: Run Validation Logic ---

def _run_validation_logic(data_train, data_test, tickers, friendly_tickers, friendly_benchmark, user_friendly_weights, 
                          rfr, sector_map, sector_caps, sector_mins, friendly_sector_map): # <-- NEW
    """
    Runs the walk-forward validation and all optimization analysis on the train data.
    Returns a dictionary of all results and plot figures.
    """
    num_assets = len(friendly_tickers)
    
    # --- 1. Train Phase (Optimization) ---
    print("   Running optimization on training data...")
    train_mean_returns, train_cov_matrix, train_log_returns = get_optimization_inputs(data_train[friendly_tickers])
    
    # Define constraints
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, tickers)
    bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
    cons_bal = build_constraints(num_assets, tickers)
    cons_sector = build_constraints(num_assets, tickers, sector_map, sector_caps, sector_mins)

    min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, rfr)
    max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, rfr)
    bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_bal, cons_bal, rfr)
    equal_weights_arr = np.array([1./num_assets] * num_assets)
    
    weights = {
        "User Portfolio": user_friendly_weights,
        "Equal Wt (Baseline)": {t: w for t, w in zip(friendly_tickers, equal_weights_arr)},
        "Min Vol": {t: w for t, w in zip(friendly_tickers, min_vol_weights_arr)},
        "Balanced (40% Cap)": {t: w for t, w in zip(friendly_tickers, bal_weights_arr)},
        "Max Sharpe": {t: w for t, w in zip(friendly_tickers, max_sharpe_weights_arr)}
    }
    
    if sector_map and (sector_caps or sector_mins):
        print("   Optimizing for Sector Balanced Portfolio...")
        sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_sector, rfr)
        weights["Sector Balanced"] = {t: w for t, w in zip(friendly_tickers, sec_bal_weights_arr)}

    # --- 2. In-Sample Evaluation (on Train data) ---
    print("   Calculating In-Sample performance...")
    in_sample_results = {}
    in_sample_eval_data = (data_train[[friendly_benchmark]] / data_train[[friendly_benchmark]].iloc[0]).copy()
    for name, w_dict in weights.items():
        in_sample_eval_data[name] = get_portfolio_price(data_train[friendly_tickers], w_dict)
        metrics, _ = calculate_metrics(in_sample_eval_data, name, friendly_benchmark, rfr)
        in_sample_results[name] = metrics

    # --- 3. Out-of-Sample Evaluation (on Test data) ---
    print("   Calculating Out-of-Sample performance...")
    out_sample_results = {}
    out_sample_eval_data = (data_test[[friendly_benchmark]] / data_test[[friendly_benchmark]].iloc[0]).copy()
    for name, w_dict in weights.items():
        out_sample_eval_data[name] = get_portfolio_price(data_test[friendly_tickers], w_dict)
        metrics, _ = calculate_metrics(out_sample_eval_data, name, friendly_benchmark, rfr)
        out_sample_results[name] = metrics
        
    # --- 4. Compile Validation Results Table ---
    print("   Compiling validation table...")
    final_results = []
    metrics_to_show = {
        "CAGR (Asset)": "CAGR", "Annualized Volatility (Asset)": "Volatility",
        "Sharpe Ratio (Asset)": "Sharpe Ratio", "Max Drawdown": "Max Drawdown",
        "Alpha (Annualized)": "Alpha"
    }
    
    for name in weights.keys():
        row = {"Portfolio": name}
        for key, short_name in metrics_to_show.items():
            row[f"In-Sample {short_name}"] = in_sample_results[name].get(key)
            row[f"Out-of-Sample {short_name}"] = out_sample_results[name].get(key)
        final_results.append(row)
        
    results_df = pd.DataFrame(final_results).set_index("Portfolio")
    results_df = results_df.map(lambda x: float(str(x).replace('%', '')) / 100 if isinstance(x, str) and '%' in x else (float(x) if isinstance(x, str) else x))
    validation_table_html = results_df.to_html(classes='metrics-table', float_format='{:.2%}'.format)

    # --- 5. Generate Validation Plots (Out-of-Sample) ---
    print("   Generating validation plots...")
    cumulative_returns_df = out_sample_eval_data
    drawdown_df = pd.DataFrame()
    for col in cumulative_returns_df.columns:
        peak = cumulative_returns_df[col].cummax()
        drawdown_df[col] = (cumulative_returns_df[col] - peak) / peak
        
    validation_plots = {
        "cumulative_plot": plot_cumulative_comparison(cumulative_returns_df, friendly_benchmark),
        "drawdown_plot": plot_drawdown_comparison(drawdown_df, friendly_benchmark)
    }
    
    # --- 6. Generate Optimization Plots (In-Sample) ---
    print("   Generating optimization (training) plots...")
    
    optimal_portfolios_train = {
        "Equal Wt (Baseline)": {"weights_arr": equal_weights_arr, "weights_dict": weights["Equal Wt (Baseline)"], "metrics": in_sample_results["Equal Wt (Baseline)"], "color": "blue"},
        "Min Vol": {"weights_arr": min_vol_weights_arr, "weights_dict": weights["Min Vol"], "metrics": in_sample_results["Min Vol"], "color": "green"},
        "Balanced (40% Cap)": {"weights_arr": bal_weights_arr, "weights_dict": weights["Balanced (40% Cap)"], "metrics": in_sample_results["Balanced (40% Cap)"], "color": "orange"},
        "Max Sharpe": {"weights_arr": max_sharpe_weights_arr, "weights_dict": weights["Max Sharpe"], "metrics": in_sample_results["Max Sharpe"], "color": "red"}
    }
    if "Sector Balanced" in weights:
        sec_bal_weights_arr = np.array(list(weights["Sector Balanced"].values()))
        optimal_portfolios_train["Sector Balanced"] = {"weights_arr": sec_bal_weights_arr, "weights_dict": weights["Sector Balanced"], "metrics": in_sample_results["Sector Balanced"], "color": "purple"}

    frontier_curve = calculate_efficient_frontier_curve(train_mean_returns, train_cov_matrix)
    
    daily_returns_df_train = in_sample_eval_data.pct_change().dropna()
    excess_returns_df_train = daily_returns_df_train - (rfr / 252)
    rolling_sharpe_df_train = (excess_returns_df_train.rolling(60).mean() * 252) / (excess_returns_df_train.rolling(60).std() * np.sqrt(252))

    optimization_plots = {
        "pie_plot": plot_composition_pies(optimal_portfolios_train),
        "sector_pie_plot": plot_sector_allocation_pies(optimal_portfolios_train, friendly_sector_map),
        "risk_contribution": plot_risk_contribution(optimal_portfolios_train, train_mean_returns, train_cov_matrix, friendly_tickers, rfr),
        "sector_risk_contribution": plot_sector_risk_contribution(optimal_portfolios_train, train_mean_returns, train_cov_matrix, friendly_tickers, friendly_sector_map, rfr),
        "rolling_sharpe_plot": plot_rolling_sharpe(rolling_sharpe_df_train, friendly_benchmark),
        "frontier": plot_efficient_frontier(train_mean_returns, train_cov_matrix, optimal_portfolios_train, frontier_curve, rfr),
        "heatmap": plot_correlation_heatmap(train_log_returns)
    }
    
    # Data for Asset-Benchmark Correlation Table
    benchmark_log_return = np.log(data_train[friendly_benchmark] / data_train[friendly_benchmark].shift(1)).dropna()
    aligned_log_returns, aligned_benchmark = train_log_returns.align(benchmark_log_return, join='inner', axis=0)
    asset_corr_df = aligned_log_returns.corrwith(aligned_benchmark).to_frame(name='Correlation')
    asset_corr_html = asset_corr_df.map(lambda x: f"{x:.2f}").to_html(classes='metrics-table')

    return {
        "table_html": validation_table_html,
        "validation_plots": validation_plots,
        "optimization_plots": optimization_plots,
        "asset_corr_html": asset_corr_html
    }