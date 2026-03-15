import logging
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)
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
    plot_sector_risk_contribution,
    plot_rebalancing_history,
    plot_bl_return_comparison,
    plot_bl_view_impact,
    plot_bl_weights_comparison
)
from .monte_carlo import (
    simulate_portfolio_paths,
    calculate_simulation_metrics,
    calculate_success_probabilities,
    plot_simulation_paths,
    plot_simulation_distribution,
    plot_probability_curve
)
from .black_litterman import (
    get_market_caps,
    calculate_market_weights,
    calculate_implied_equilibrium_returns,
    calculate_black_litterman_posterior
)
from .rebalancing import simulate_rebalanced_portfolio
# --- End Imports ---


def create_combined_report(portfolio_dict, benchmark_ticker,
                           train_start, train_end,
                           filename="Combined_Report.html",
                           risk_free_rate="auto",
                           display_names=None,
                           sector_map=None,
                           sector_caps=None,
                           sector_mins=None,
                           bl_views=None,
                           bl_view_confidences=None,
                           bl_relative_views=None,
                           bl_relative_view_confidences=None,
                           rebalance_freq=None, desc=False,
                           denoise_cov=False, n_components=3): 
    """
    Generates a single, combined HTML report for portfolio analysis,
    optimization, and walk-forward validation.
    """
    logger.info("Initiating Combined Report")
    
    try:
        # --- A. Date & Name Setup ---
        test_start_dt = (pd.to_datetime(train_end) + timedelta(days=1))
        test_end_dt = (datetime.now() - timedelta(days=1))
        test_start = test_start_dt.strftime('%Y-%m-%d')
        test_end = test_end_dt.strftime('%Y-%m-%d')
        full_start = train_start
        full_end = test_end
        
        logger.info("Full Period:  %s to %s", full_start, full_end)
        logger.info("Train Period: %s to %s", train_start, train_end)
        logger.info("Test Period:  %s to %s", test_start, test_end)

        if isinstance(risk_free_rate, str) and risk_free_rate.lower() == 'auto':
            rfr = get_risk_free_rate()
        elif isinstance(risk_free_rate, (int, float)):
            rfr = risk_free_rate
        else: rfr = 0.02
        logger.info("Using Risk-Free Rate: %.2f%%", rfr * 100)

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
        logger.info("Fetching all data...")
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
        logger.info("Running user portfolio report (Full Period)...")
        
        pr_eval_data = (data_full[[friendly_benchmark]] / data_full[[friendly_benchmark]].iloc[0]).copy()
        pr_eval_data['My Portfolio'] = get_portfolio_price(data_full[friendly_tickers], user_friendly_weights)
        pr_metrics, pr_plot_data = calculate_metrics(pr_eval_data, 'My Portfolio', friendly_benchmark, rfr)
        pr_plots = {
            "cumulative": plot_cumulative_returns(pr_plot_data, 'My Portfolio', friendly_benchmark),
            "regression": plot_regression(pr_plot_data, pr_metrics, 'My Portfolio', friendly_benchmark)
        }
        
        pr_cumulative_desc = "Shows the growth of $1 invested at the start of the Full Period, comparing your portfolio directly against the benchmark. Look for consistent outperformance." if desc else None
        pr_regression_desc = "Scatter plot of your portfolio's daily returns versus the benchmark's returns. The slope of the line (Beta) indicates market risk, while the y-intercept (Alpha) indicates outperformance independent of the market." if desc else None

        pr_rolling_returns_html = calculate_rolling_returns(pr_eval_data).to_html(classes='metrics-table')
        
        # --- D. Run Optimization & Validation (Train/Test Split) ---
        logger.info("Running optimization & validation...")
        
        # Translate BL view keys from raw tickers to display names 
        # (the covariance matrix uses display names at this point)
        friendly_bl_views = None
        friendly_bl_confidences = None
        if bl_views and display_names:
            friendly_bl_views = {display_names.get(t, t): v for t, v in bl_views.items()}
        elif bl_views:
            friendly_bl_views = bl_views
        if bl_view_confidences and display_names:
            friendly_bl_confidences = {display_names.get(t, t): v for t, v in bl_view_confidences.items()}
        elif bl_view_confidences:
            friendly_bl_confidences = bl_view_confidences

        # Translate BL relative view keys from raw tickers to display names
        friendly_bl_relative_views = None
        friendly_bl_relative_confidences = bl_relative_view_confidences
        if bl_relative_views and display_names:
            friendly_bl_relative_views = [
                (display_names.get(o, o), display_names.get(u, u), s)
                for o, u, s in bl_relative_views
            ]
        elif bl_relative_views:
            friendly_bl_relative_views = bl_relative_views

        val_results = _run_validation_logic(
            data_train, data_test, tickers, friendly_tickers, friendly_benchmark, user_friendly_weights, 
            rfr, sector_map, sector_caps, sector_mins, friendly_sector_map,
            bl_views=friendly_bl_views, bl_view_confidences=friendly_bl_confidences,
            bl_relative_views=friendly_bl_relative_views,
            bl_relative_view_confidences=friendly_bl_relative_confidences,
            rebalance_freq=rebalance_freq, desc=desc,
            denoise_cov=denoise_cov, n_components=n_components
        )
        
        # --- E. Prepare Descriptions ---
        # 1. Optimization 
        opt_pie_desc = "Breakdown of the specific underlying assets allocated to each optimization strategy based on the Training Data." if desc else None
        opt_sec_pie_desc = "Broader view of the asset allocations grouped by their market sector." if desc else None
        opt_risk_desc = "Decomposition of portfolio volatility. Shows which individual assets contribute the most to the portfolio's overall price swings." if desc else None
        opt_sec_risk_desc = "Decomposition of portfolio volatility grouped by sector. Helps identify unintended macro risk concentrations." if desc else None
        opt_roll_desc = "60-day rolling Sharpe ratio over the training period. Tracks how risk-adjusted performance changed dynamically over time. Values > 1 indicate excellent historical risk-adjusted returns." if desc else None
        opt_front_desc = "The Efficient Frontier plots the optimal tradeoff curve between Risk (Volatility) and Reward (Expected Return). Portfolios lying on the dotted line offer the best theoretical return for their level of risk." if desc else None
        opt_heat_desc = "Correlation matrix of the assets. Green implies assets move together, red implies they move inversely. Lower correlations generally improve diversification benefits." if desc else None

        # 2. Validation
        val_cum_desc = "Out-of-sample performance: How the optimized portfolios would have performed if deployed during the Test Period. This is the true test of robustness." if desc else None
        val_draw_desc = "Measures peak-to-trough declines during the Test Period. Flattish lines at 0 indicate all-time highs, while deep dips quantify maximum historical pain." if desc else None

        # 3. Monte Carlo
        mc_paths_desc = "Randomly generated potential future trajectories for the User Portfolio based on its historical volatility and returns. Gives a sense of the 'cone of uncertainty'." if desc else None
        mc_dist_desc = "Histogram showing the distribution of final portfolio values across all 1000 simulations. Focus on the 5th and 95th percentiles to understand the tail risks and upside potential." if desc else None
        mc_prob_desc = "The chance of achieving at least a target cumulative return over the horizon. Use this to gauge the likelihood of hitting specific financial goals." if desc else None

        # 4. Black-Litterman
        bl_comp_desc = "Compares the purely market-implied (Equilibrium) returns against the Posterior returns generated after blending in the investor views." if desc else None
        bl_imp_desc = "Shows exactly how much the investor views shifted the expected returns upwards or downwards from the baseline equilibrium." if desc else None
        bl_wgt_desc = "How the resulting optimal Black-Litterman portfolio weights differ from the standard market-cap weighted allocation." if desc else None

        # --- F. Generate Final HTML ---
        logger.info("Generating combined HTML...")
        
        sections = [
            {
                "title": "1. User Portfolio (Full Period)",
                "description": f"Analysis of your user-defined portfolio mix ({pr_metrics.get('CAGR (Asset)', 'N/A')} CAGR) over the full historical period.",
                "sidebar": [
                    {"title": "Performance Metrics", "type": "metrics", "data": pr_metrics},
                    {"title": "Rolling Returns Summary", "type": "table_html", "data": pr_rolling_returns_html}
                ],
                "main_content": [
                    {"title": "Cumulative Returns", "type": "plot", "data": pr_plots["cumulative"], "description": pr_cumulative_desc},
                    {"title": "Regression Analysis", "type": "plot", "data": pr_plots["regression"], "description": pr_regression_desc}
                ]
            },
            {
                "title": "2. Optimization Analysis (Train Period)",
                "description": f"Analysis performed on the Training Data ({train_start} to {train_end}) to generate the optimal weights.",
                "sidebar": [
                    {"title": "Asset-Benchmark Correlation (Train)", "type": "table_html", "data": val_results["asset_corr_html"]}
                ],
                "main_content": [
                    {"title": "Strategy Compositions (by Asset)", "type": "plot", "data": val_results['optimization_plots']['pie_plot'], "description": opt_pie_desc},
                    {"title": "Strategy Compositions (by Sector)", "type": "plot", "data": val_results['optimization_plots']['sector_pie_plot'], "description": opt_sec_pie_desc},
                    {"title": "Portfolio Risk Contribution (by Asset)", "type": "plot", "data": val_results['optimization_plots']['risk_contribution'], "description": opt_risk_desc},
                    {"title": "Portfolio Risk Contribution (by Sector)", "type": "plot", "data": val_results['optimization_plots']['sector_risk_contribution'], "description": opt_sec_risk_desc},
                    {"title": "Strategy Rolling Sharpe Ratio (from Train)", "type": "plot", "data": val_results['optimization_plots']['rolling_sharpe_plot'], "description": opt_roll_desc},
                    {"title": "Efficient Frontier (from Train)", "type": "plot", "data": val_results['optimization_plots']['frontier'], "description": opt_front_desc},
                    {"title": "Asset Correlation Heatmap (from Train)", "type": "plot", "data": val_results['optimization_plots']['heatmap'], "description": opt_heat_desc}
                ]
            },
            {
                "title": "3. Walk-Forward Validation (Test Period)",
                "description": f"Tests how portfolios optimized on training data would have performed in the Test Period ({test_start} to {test_end}).",
                "sidebar": [],
                "main_content": [
                    {"title": "In-Sample vs. Out-of-Sample Performance", "type": "table_html", "data": val_results["table_html"]},
                    {"title": "Out-of-Sample Cumulative Returns", "type": "plot", "data": val_results['validation_plots']['cumulative_plot'], "description": val_cum_desc},
                    {"title": "Out-of-Sample Drawdown", "type": "plot", "data": val_results['validation_plots']['drawdown_plot'], "description": val_draw_desc}
                ]
            },
        ]

        # Add Rebalancing History plot if it exists
        if 'rebalance_history' in val_results['validation_plots']:
            val_reb_desc = "Shows the evolution of portfolio weights during the test period. The 'hills' and 'valleys' represent asset weights drifting as prices change, while vertical resets indicate rebalancing back to target." if desc else None
            sections[2]["main_content"].append({
                "title": "Portfolio Rebalancing History (Weight Evolution)",
                "type": "plot",
                "data": val_results['validation_plots']['rebalance_history'],
                "description": val_reb_desc
            })
        
        sections.append(
            {
                "title": "4. Monte Carlo Simulation (User Portfolio)",
                "description": f"Projection of future portfolio value using 1000 simulations over the Test Period ({val_results['test_days']} trading days).",
                "sidebar": [
                    {"title": "Simulation Risk Metrics", "type": "metrics", "data": val_results['mc_metrics']},
                    {"title": "Success Probabilities", "type": "metrics", "data": val_results['mc_probs']}
                ],
                "main_content": [
                    {"title": "Projected Future Paths", "type": "plot", "data": val_results['mc_plots']['paths'], "description": mc_paths_desc},
                    {"title": "Distribution of Final Returns", "type": "plot", "data": val_results['mc_plots']['dist'], "description": mc_dist_desc},
                    {"title": "Probability of Exceeding Return", "type": "plot", "data": val_results['mc_plots']['prob_curve'], "description": mc_prob_desc}
                ]
            }
        )
        
        # --- Conditionally add BL Deep Dive section ---
        if val_results.get('bl_plots'):
            bl_main_content = []
            if 'return_comparison' in val_results['bl_plots']:
                bl_main_content.append({
                    "title": "Equilibrium vs Posterior Expected Returns",
                    "type": "plot", "data": val_results['bl_plots']['return_comparison'],
                    "description": bl_comp_desc
                })
            if 'view_impact' in val_results['bl_plots']:
                bl_main_content.append({
                    "title": "View Impact: Shift from Market Equilibrium",
                    "type": "plot", "data": val_results['bl_plots']['view_impact'],
                    "description": bl_imp_desc
                })
            if 'weights_comparison' in val_results['bl_plots']:
                bl_main_content.append({
                    "title": "Market-Cap Weights vs BL Optimized Weights",
                    "type": "plot", "data": val_results['bl_plots']['weights_comparison'],
                    "description": bl_wgt_desc
                })
            
            if bl_main_content:
                sections.append({
                    "title": "5. Black-Litterman Deep Dive",
                    "description": "How investor views shifted expected returns from market equilibrium, and the resulting portfolio weight differences.",
                    "sidebar": [],
                    "main_content": bl_main_content
                })
        
        generate_html_report(sections, title="Combined Portfolio Report", filename=filename)
        
        logger.info("Combined Report Generated: %s", filename)
        
    except Exception as e:
        logger.error("An error occurred during combined report generation: %s", e)
        traceback.print_exc()

# --- 4. Helper: Run Validation Logic ---

def _run_validation_logic(data_train, data_test, tickers, friendly_tickers, friendly_benchmark, user_friendly_weights, 
                          rfr, sector_map, sector_caps, sector_mins, friendly_sector_map,
                          bl_views=None, bl_view_confidences=None,
                          bl_relative_views=None, bl_relative_view_confidences=None,
                          rebalance_freq=None, desc=False,
                          denoise_cov=False, n_components=3):
    """
    Runs the walk-forward validation and all optimization analysis on the train data.
    Returns a dictionary of all results and plot figures.
    """
    num_assets = len(friendly_tickers)
    
    # --- 1. Train Phase (Optimization) ---
    logger.info("Calculating optimal weights based on Training Data...")
    
    # Use denoising if requested
    train_mean_returns, train_cov_matrix, train_log_returns = get_optimization_inputs(
        data_train[friendly_tickers], denoise_cov=denoise_cov, n_components=n_components
    )
    
    # 1a. Standard Constraints & Bounds
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, tickers)
    bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
    cons_bal = build_constraints(num_assets, tickers)
    cons_sector = build_constraints(num_assets, tickers, sector_map, sector_caps, sector_mins)

    min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, rfr)
    max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_uncon, rfr)
    bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_bal, cons_bal, rfr)
    equal_weights_arr = np.array([1./num_assets] * num_assets)

    # --- 1b. Black-Litterman Optimization ---
    logger.info("Running Black-Litterman optimization...")
    bl_equilibrium_returns = None
    bl_market_weights = None
    try:
        # Fetch market caps for the training period assets
        market_caps = get_market_caps(tickers)
        
        # Calculate market-cap weights and equilibrium returns (for visualization)
        bl_market_weights = calculate_market_weights(market_caps)
        bl_market_weights = bl_market_weights.reindex(friendly_tickers).fillna(0)
        bl_market_weights = bl_market_weights / bl_market_weights.sum()
        bl_equilibrium_returns = calculate_implied_equilibrium_returns(
            train_cov_matrix, bl_market_weights, risk_aversion=2.5
        )
        
        # Calculate Posterior returns with views
        bl_means, bl_cov = calculate_black_litterman_posterior(
            train_mean_returns, 
            train_cov_matrix, 
            market_caps=market_caps,
            risk_aversion=2.5,
            view_dict=bl_views,
            view_confidences=bl_view_confidences,
            relative_views=bl_relative_views,
            relative_view_confidences=bl_relative_view_confidences
        )
        
        # Optimize using BL estimates (Max Sharpe on BL posterior)
        bl_weights_arr = find_optimal_portfolio(objective_neg_sharpe, bl_means, bl_cov, bounds_uncon, cons_uncon, rfr)
        has_bl = True
    except Exception as e:
        logger.warning("Black-Litterman optimization failed: %s", e)
        traceback.print_exc()
        bl_weights_arr = None
        has_bl = False
    
    weights = {
        "User Portfolio": user_friendly_weights,
        "Equal Wt (Baseline)": {t: w for t, w in zip(friendly_tickers, equal_weights_arr)},
        "Min Vol": {t: w for t, w in zip(friendly_tickers, min_vol_weights_arr)},
        "Balanced (40% Cap)": {t: w for t, w in zip(friendly_tickers, bal_weights_arr)},
        "Max Sharpe": {t: w for t, w in zip(friendly_tickers, max_sharpe_weights_arr)}
    }
    
    if has_bl and bl_weights_arr is not None:
        weights["Black-Litterman (Mkt Caps)"] = {t: w for t, w in zip(friendly_tickers, bl_weights_arr)}
    
    if sector_map and (sector_caps or sector_mins):
        logger.info("Optimizing for Sector Balanced Portfolio...")
        sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, train_mean_returns, train_cov_matrix, bounds_uncon, cons_sector, rfr)
        weights["Sector Balanced"] = {t: w for t, w in zip(friendly_tickers, sec_bal_weights_arr)}

    # --- 2. In-Sample Evaluation (on Train data) ---
    logger.info("Calculating In-Sample performance...")
    if rebalance_freq:
        logger.info("Using rebalanced portfolios (freq=%s) for in-sample.", rebalance_freq)
    in_sample_results = {}
    in_sample_eval_data = (data_train[[friendly_benchmark]] / data_train[[friendly_benchmark]].iloc[0]).copy()
    for name, w_dict in weights.items():
        if rebalance_freq is not None:
            w_series = pd.Series(w_dict)
            w_series.index = [t for t in tickers if friendly_tickers[tickers.index(t)] in w_dict.keys() or t in w_dict.keys()]
            # Create a localized copy of train data with raw tickers for simulation
            train_raw = data_train[friendly_tickers].copy()
            train_raw.columns = tickers
            in_sample_eval_data[name], _ = simulate_rebalanced_portfolio(train_raw, w_series, rebalance_freq)
        else:
            in_sample_eval_data[name] = get_portfolio_price(data_train[friendly_tickers], w_dict)
        metrics, _ = calculate_metrics(in_sample_eval_data, name, friendly_benchmark, rfr)
        in_sample_results[name] = metrics

    # --- 3. Out-of-Sample Evaluation (on Test data) ---
    logger.info("Calculating Out-of-Sample performance...")
    if rebalance_freq:
        logger.info("Using rebalanced portfolios (freq=%s) for out-of-sample.", rebalance_freq)
    out_sample_results = {}
    out_sample_eval_data = (data_test[[friendly_benchmark]] / data_test[[friendly_benchmark]].iloc[0]).copy()
    
    rebalance_history_df = None
    
    for name, w_dict in weights.items():
        if rebalance_freq is not None:
            p_series, w_history_df = simulate_rebalanced_portfolio(
                data_test[friendly_tickers], w_dict, rebalance_freq=rebalance_freq
            )
            out_sample_eval_data[name] = p_series
            
            # Save the weight history for the User Portfolio or Max Sharpe (as primary examples)
            if rebalance_history_df is None or name == "User Portfolio":
                rebalance_history_df = w_history_df
        else:
            out_sample_eval_data[name] = get_portfolio_price(data_test[friendly_tickers], w_dict)
            
        metrics, _ = calculate_metrics(out_sample_eval_data, name, friendly_benchmark, rfr)
        out_sample_results[name] = metrics
        
    # --- 4. Compile Validation Results Table ---
    logger.info("Compiling validation table...")
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
    logger.info("Generating validation plots...")
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
    logger.info("Generating optimization (training) plots...")
    
    optimal_portfolios_train = {
        "Equal Wt (Baseline)": {"weights_arr": equal_weights_arr, "weights_dict": weights["Equal Wt (Baseline)"], "metrics": in_sample_results["Equal Wt (Baseline)"], "color": "blue"},
        "Min Vol": {"weights_arr": min_vol_weights_arr, "weights_dict": weights["Min Vol"], "metrics": in_sample_results["Min Vol"], "color": "green"},
        "Balanced (40% Cap)": {"weights_arr": bal_weights_arr, "weights_dict": weights["Balanced (40% Cap)"], "metrics": in_sample_results["Balanced (40% Cap)"], "color": "orange"},
        "Max Sharpe": {"weights_arr": max_sharpe_weights_arr, "weights_dict": weights["Max Sharpe"], "metrics": in_sample_results["Max Sharpe"], "color": "red"}
    }
    if "Black-Litterman (Mkt Caps)" in weights:
        bl_w = np.array(list(weights["Black-Litterman (Mkt Caps)"].values()))
        optimal_portfolios_train["Black-Litterman"] = {
            "weights_arr": bl_w, 
            "weights_dict": weights["Black-Litterman (Mkt Caps)"], 
            "metrics": in_sample_results["Black-Litterman (Mkt Caps)"], 
            "color": "teal"
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

    # --- 7. Run Monte Carlo Simulation (on User Portfolio) ---
    logger.info("Running Monte Carlo simulation...")
    # We use the full period mean returns and cov matrix for the simulation to be most representative
    # Or we can use the train period. Let's use the train period to be consistent with the "validation" theme,
    # effectively asking "Based on what we knew then, what did we project?"
    # We set the time horizon to match the actual Test period length for a fair comparison.
    
    test_days = len(data_test)
    logger.info("Simulation Horizon: %d trading days (matching Test period)", test_days)
    
    user_weights_arr = np.array(list(user_friendly_weights.values()))
    mc_sim_df = simulate_portfolio_paths(user_weights_arr, train_mean_returns, train_cov_matrix, num_simulations=1000, time_horizon=test_days)
    
    # Get Actuals
    actual_path = out_sample_eval_data["User Portfolio"]
    actual_return = (actual_path.iloc[-1] / actual_path.iloc[0]) - 1
    
    mc_metrics, mc_total_returns = calculate_simulation_metrics(mc_sim_df, actual_return=actual_return)
    mc_probs = calculate_success_probabilities(mc_total_returns)
    
    mc_plots = {
        "paths": plot_simulation_paths(mc_sim_df, actual_path=actual_path),
        "dist": plot_simulation_distribution(mc_total_returns, actual_return=actual_return),
        "prob_curve": plot_probability_curve(mc_total_returns, actual_return=actual_return)
    }

    # --- 8. Generate BL-specific Plots ---
    bl_plots = {}
    if has_bl and bl_equilibrium_returns is not None:
        logger.info("Generating Black-Litterman deep-dive plots...")
        try:
            bl_plots['return_comparison'] = plot_bl_return_comparison(
                bl_equilibrium_returns, bl_means, view_dict=bl_views
            )
            if bl_views:
                bl_plots['view_impact'] = plot_bl_view_impact(
                    bl_equilibrium_returns, bl_means, bl_views
                )
            bl_plots['weights_comparison'] = plot_bl_weights_comparison(
                bl_market_weights,
                weights.get("Black-Litterman (Mkt Caps)", {}),
                view_dict=bl_views
            )
        except Exception as e:
            logger.warning("BL plot generation failed: %s", e)
            traceback.print_exc()

    return {
        "table_html": validation_table_html,
        "validation_plots": validation_plots,
        "optimization_plots": optimization_plots,
        "asset_corr_html": asset_corr_html,
        "mc_metrics": mc_metrics,
        "mc_probs": mc_probs,
        "mc_plots": mc_plots,
        "test_days": test_days,
        "bl_plots": bl_plots
    }