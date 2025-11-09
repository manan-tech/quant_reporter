import pandas as pd
import numpy as np
import traceback
from .data import get_data
from .metrics import calculate_metrics
from .html_builder import generate_html_report

# Import from new utility modules
from .opt_core import (
    get_optimization_inputs,
    find_optimal_portfolio,
    objective_neg_sharpe,
    objective_min_variance,
    get_portfolio_price,
    build_constraints
)
from .opt_plotting import (
    plot_cumulative_comparison,
    plot_drawdown_comparison
)

def create_validation_report(portfolio_dict, benchmark_ticker, 
                             train_start, train_end, 
                             test_start, test_end, 
                             filename="validation_report.html", 
                             risk_free_rate=0.02,
                             display_names=None,
                             sector_map=None,
                             sector_caps=None,
                             sector_mins=None): 
    """
    Runs a full walk-forward validation (Out-of-Sample test)
    """
    print("--- Starting Walk-Forward Validation Report ---")
    
    try:
        # --- 0. Set up names ---
        tickers = list(portfolio_dict.keys())
        num_assets = len(tickers)
        
        friendly_benchmark = benchmark_ticker
        friendly_tickers = tickers
        
        if display_names:
            print("Applying display names...")
            friendly_benchmark = display_names.get(benchmark_ticker, benchmark_ticker)
            friendly_tickers = [display_names.get(t, t) for t in tickers]

        # --- 1. IN-SAMPLE (TRAIN) PERIOD ---
        print(f"--- 1. Training Phase ({train_start} to {train_end}) ---")
        
        all_train_tickers = tickers + [benchmark_ticker]
        train_price_data = get_data(all_train_tickers, train_start, train_end)
        if train_price_data is None: raise ValueError("Could not fetch training data.")
        
        if display_names:
            train_price_data.rename(columns=display_names, inplace=True)
        
        train_mean_returns, train_cov_matrix, _ = get_optimization_inputs(train_price_data[friendly_tickers])
        
        # --- Find optimal weights ---
        bounds_uncon = tuple((0, 1) for _ in range(num_assets))
        cons_uncon = build_constraints(num_assets, tickers)
        bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
        cons_bal = build_constraints(num_assets, tickers)
        cons_sector = build_constraints(num_assets, tickers, sector_map, sector_caps, sector_mins) # <-- Pass mins

        min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, risk_free_rate)
        max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, risk_free_rate)
        bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_bal, cons_bal, risk_free_rate)
        equal_weights_arr = np.array([1./num_assets] * num_assets)
        
        weights = {
            "Equal Wt": {t: w for t, w in zip(friendly_tickers, equal_weights_arr)},
            "Min Vol": {t: w for t, w in zip(friendly_tickers, min_vol_weights_arr)},
            "Balanced (40% Cap)": {t: w for t, w in zip(friendly_tickers, bal_weights_arr)},
            "Max Sharpe": {t: w for t, w in zip(friendly_tickers, max_sharpe_weights_arr)}
        }
        
        if sector_map and (sector_caps or sector_mins):
            print("   Optimizing for Sector Balanced Portfolio...")
            sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_sector, risk_free_rate)
            weights["Sector Balanced"] = {t: w for t, w in zip(friendly_tickers, sec_bal_weights_arr)}

        # --- Calculate IN-SAMPLE performance ---
        print("Calculating In-Sample performance...")
        in_sample_results = {}
        in_sample_eval_data = (train_price_data[[friendly_benchmark]] / train_price_data[[friendly_benchmark]].iloc[0]).copy()
        for name, w_dict in weights.items():
            in_sample_eval_data[name] = get_portfolio_price(train_price_data[friendly_tickers], w_dict)
            metrics, _ = calculate_metrics(in_sample_eval_data, name, friendly_benchmark, risk_free_rate)
            in_sample_results[name] = metrics

        # --- 2. OUT-OF-SAMPLE (TEST) PERIOD ---
        print(f"--- 2. Testing Phase ({test_start} to {test_end}) ---")
        
        all_test_tickers = tickers + [benchmark_ticker]
        test_price_data = get_data(all_test_tickers, test_start, test_end)
        if test_price_data is None: raise ValueError("Could not fetch testing data.")
            
        if display_names:
            test_price_data.rename(columns=display_names, inplace=True)
            
        print("Calculating Out-of-Sample performance...")
        out_sample_results = {}
        out_sample_eval_data = (test_price_data[[friendly_benchmark]] / test_price_data[[friendly_benchmark]].iloc[0]).copy()
        for name, w_dict in weights.items():
            out_sample_eval_data[name] = get_portfolio_price(test_price_data[friendly_tickers], w_dict)
            metrics, _ = calculate_metrics(out_sample_eval_data, name, friendly_benchmark, risk_free_rate)
            out_sample_results[name] = metrics

        # --- 3. Compile Final Results Table ---
        print("Compiling final report...")
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

        # --- 4. Generate Plots for Test Period ---
        cumulative_returns_df = out_sample_eval_data
        drawdown_df = pd.DataFrame()
        for col in cumulative_returns_df.columns:
            peak = cumulative_returns_df[col].cummax()
            drawdown_df[col] = (cumulative_returns_df[col] - peak) / peak
            
        # --- 5. Build Report Sections ---
        sections = [{
            "title": "Walk-Forward Validation",
            "description": f"Training on {train_start} to {train_end}, then testing on {test_start} to {test_end}.",
            "sidebar": [],
            "main_content": [
                {"title": "In-Sample vs. Out-of-Sample Performance", "type": "table_html", "data": validation_table_html},
                {"title": "Out-of-Sample Cumulative Returns", "type": "plot", "data": plot_cumulative_comparison(cumulative_returns_df, friendly_benchmark)},
                {"title": "Out-of-Sample Drawdown", "type": "plot", "data": plot_drawdown_comparison(drawdown_df, friendly_benchmark)}
            ]
        }]
        
        # --- 6. Generate Report ---
        generate_html_report(sections, title="Walk-Forward Validation Report", filename=filename)
        
        print(f"--- Validation Report Generated: {filename} ---")
        
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        traceback.print_exc()