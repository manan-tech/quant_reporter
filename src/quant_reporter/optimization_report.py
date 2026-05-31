import logging
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .metrics import calculate_metrics
from .html_builder import generate_html_report

from .opt_core import (
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
    plot_monthly_heatmaps,
    plot_sector_allocation_pies,  
    plot_sector_risk_contribution,
    plot_bl_return_comparison,
    plot_bl_view_impact,
    plot_bl_weights_comparison
)
from .advanced_optimizers import (
    optimize_risk_parity,
    optimize_hrp,
    optimize_min_correlation,
    optimize_max_diversification
)
from .black_litterman import (
    get_market_caps,
    calculate_market_weights,
    calculate_implied_equilibrium_returns,
    calculate_black_litterman_posterior
)

def calculate_transition_costs(current_weights, target_weights_dict, transaction_cost_bps=10):
    """
    Estimates the turnover and transition cost to move from the current
    portfolio to a target portfolio.
    """
    cost_data = []
    
    # Ensure current weights dict is complete for all target keys
    all_assets = set(current_weights.keys())
    for target_name, t_weights in target_weights_dict.items():
        all_assets.update(t_weights.keys())
    
    current_aligned = np.array([current_weights.get(a, 0.0) for a in all_assets])
    
    for target_name, t_weights in target_weights_dict.items():
        target_aligned = np.array([t_weights.get(a, 0.0) for a in all_assets])
        # Turnover is half the sum of absolute weight changes
        turnover = 0.5 * np.sum(np.abs(target_aligned - current_aligned))
        est_cost_pct = turnover * (transaction_cost_bps / 10000)
        
        cost_data.append({
            "Target Strategy": target_name,
            "Turnover": turnover,
            "Est. Transition Cost": est_cost_pct
        })
        
    df = pd.DataFrame(cost_data).set_index("Target Strategy")
    return df

def calculate_market_regime_capture(eval_data, benchmark_ticker):
    """
    Calculates Up-Market and Down-Market Capture ratios for each strategy.
    """
    daily_returns = eval_data.pct_change().dropna()
    bench_returns = daily_returns[benchmark_ticker]
    
    up_days = bench_returns > 0
    down_days = bench_returns <= 0
    
    capture_data = {}
    for col in daily_returns.columns:
        if col == benchmark_ticker:
            continue
            
        up_return = (1 + daily_returns.loc[up_days, col]).prod() - 1
        bench_up = (1 + bench_returns[up_days]).prod() - 1
        up_capture = (up_return / bench_up) if bench_up > 0 else 0
        
        down_return = (1 + daily_returns.loc[down_days, col]).prod() - 1
        bench_down = (1 + bench_returns[down_days]).prod() - 1
        down_capture = (down_return / bench_down) if bench_down < 0 else 0
        
        capture_data[col] = {
            "Up-Market Capture": up_capture,
            "Down-Market Capture": down_capture
        }
        
    return pd.DataFrame.from_dict(capture_data, orient='index')

def compute_optimization_analysis(ctx: ReportContext):
    """
    Computes all optimizations and comparative analysis.
    """
    logger.info("Computing Optimization Analysis...")
    
    num_assets = len(ctx.tickers)
    
    # 1. Define Constraints
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, ctx.tickers)
    bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
    cons_bal = build_constraints(num_assets, ctx.tickers)
    cons_sector = build_constraints(num_assets, ctx.tickers, ctx.sector_map, ctx.sector_caps, ctx.sector_mins)
    
    # 2. Run Standard Optimizers
    logger.info("Running Standard Optimizers...")
    min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
    min_vol_weights_dict = {t: w for t, w in zip(ctx.friendly_tickers, min_vol_weights_arr)}

    max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
    max_sharpe_weights_dict = {t: w for t, w in zip(ctx.friendly_tickers, max_sharpe_weights_arr)}

    bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_bal, cons_bal, ctx.risk_free_rate)
    bal_weights_dict = {t: w for t, w in zip(ctx.friendly_tickers, bal_weights_arr)}
    
    equal_weights_arr = np.array([1./num_assets] * num_assets)
    equal_weights_dict = {t: w for t, w in zip(ctx.friendly_tickers, equal_weights_arr)}

    weights_collection = {
        "User Portfolio": ctx.user_friendly_weights,
        "Equal Wt (Baseline)": equal_weights_dict,
        "Minimum Volatility": min_vol_weights_dict,
        "Max Sharpe (Unconstrained)": max_sharpe_weights_dict,
        "Balanced (40% Cap)": bal_weights_dict
    }

    if ctx.sector_map and (ctx.sector_caps or ctx.sector_mins):
        logger.info("Optimizing for Sector Balanced Portfolio...")
        sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_sector, ctx.risk_free_rate)
        weights_collection["Sector Balanced"] = {t: w for t, w in zip(ctx.friendly_tickers, sec_bal_weights_arr)}

    # 3. Run Advanced Optimizers (Silent fail if error)
    advanced_optims = [
        ("Risk Parity", optimize_risk_parity),
        ("Min Correlation", optimize_min_correlation),
        ("Max Diversification", optimize_max_diversification)
    ]
    
    for name, func in advanced_optims:
        try:
            arr = func(ctx.cov_matrix)
            weights_collection[name] = {t: w for t, w in zip(ctx.friendly_tickers, arr)}
        except Exception as e:
            logger.debug(f"{name} optimization failed: {e}")

    try:
        hrp_arr, _ = optimize_hrp(ctx.cov_matrix)
        weights_collection["HRP"] = {t: w for t, w in zip(ctx.friendly_tickers, hrp_arr)}
    except Exception:
        pass

    # 4. Black-Litterman (if applicable)
    bl_plots = []
    if ctx.bl_views or ctx.bl_relative_views:
        logger.info("Running Black-Litterman optimization...")
        try:
            market_caps = get_market_caps(ctx.tickers)
            bl_market_weights = calculate_market_weights(market_caps)
            bl_market_weights = bl_market_weights.reindex(ctx.friendly_tickers).fillna(0)
            bl_market_weights = bl_market_weights / bl_market_weights.sum()
            
            bl_equilibrium_returns = calculate_implied_equilibrium_returns(
                ctx.cov_matrix, bl_market_weights, risk_aversion=2.5
            )
            
            bl_means, bl_cov = calculate_black_litterman_posterior(
                ctx.mean_returns, 
                ctx.cov_matrix, 
                market_caps=market_caps,
                risk_aversion=2.5,
                view_dict=ctx.bl_views,
                view_confidences=ctx.bl_view_confidences,
                relative_views=ctx.bl_relative_views,
                relative_view_confidences=ctx.bl_relative_view_confidences
            )
            
            bl_weights_arr = find_optimal_portfolio(objective_neg_sharpe, bl_means, bl_cov, bounds_uncon, cons_uncon, ctx.risk_free_rate)
            weights_collection["Black-Litterman"] = {t: w for t, w in zip(ctx.friendly_tickers, bl_weights_arr)}
            
            # Save BL specific plots
            if ctx.bl_views:
                mapped_views = {ctx.display_names.get(k, k) if ctx.display_names else k: v for k, v in ctx.bl_views.items()}
            else:
                mapped_views = {}
            bl_plots.extend([
                {"title": "BL: Equilibrium vs Posterior Returns", "type": "plot", "data": plot_bl_return_comparison(bl_equilibrium_returns, bl_means, mapped_views)},
                {"title": "BL: View Impact", "type": "plot", "data": plot_bl_view_impact(bl_equilibrium_returns, bl_means, mapped_views)},
                {"title": "BL: Market vs Optimized Weights", "type": "plot", "data": plot_bl_weights_comparison(bl_market_weights, weights_collection["Black-Litterman"], mapped_views)}
            ])
        except Exception as e:
            logger.warning(f"Black-Litterman skipped: {e}")

    # 5. Evaluate Performance (In-Sample)
    eval_data = (ctx.price_data_train[[ctx.friendly_benchmark]] / ctx.price_data_train[[ctx.friendly_benchmark]].iloc[0]).copy()
    optimal_portfolios = {}
    
    target_weights_dict = {}

    for name, w_dict in weights_collection.items():
        eval_data[name] = get_portfolio_price(ctx.price_data_train[ctx.friendly_tickers], w_dict)
        metrics, _ = calculate_metrics(eval_data, name, ctx.friendly_benchmark, ctx.risk_free_rate)
        # Store for pie charts and rich info
        aligned_arr = np.array([w_dict.get(t, 0) for t in ctx.friendly_tickers])
        color = "blue" # We can make color dynamic if needed
        optimal_portfolios[name] = {
            "weights_arr": aligned_arr,
            "weights_dict": w_dict,
            "metrics": metrics,
            "color": "rgb(55, 128, 191)" # Default color
        }
        if name != "User Portfolio":
            target_weights_dict[name] = w_dict

    # 6. Calculate Transition Costs & Regime Capture
    transition_df = calculate_transition_costs(ctx.user_friendly_weights, target_weights_dict)
    capture_df = calculate_market_regime_capture(eval_data, ctx.friendly_benchmark)

    # Convert to HTML for sidebar
    transition_html = transition_df.map(lambda x: f"{x:.2%}").to_html(classes='metrics-table')
    capture_html = capture_df.map(lambda x: f"{x:.2f}x").to_html(classes='metrics-table')

    # 7. Precompute Chart Data
    daily_returns_df = eval_data.pct_change().dropna()
    excess_returns_df = daily_returns_df - (ctx.risk_free_rate / 252)
    rolling_sharpe_df = (excess_returns_df.rolling(60).mean() * 252) / (excess_returns_df.rolling(60).std() * np.sqrt(252))
    
    cumulative_returns_df = eval_data
    drawdown_df = pd.DataFrame()
    for col in cumulative_returns_df.columns:
        peak = cumulative_returns_df[col].cummax()
        drawdown_df[col] = (cumulative_returns_df[col] - peak) / peak
        
    frontier_curve = calculate_efficient_frontier_curve(ctx.mean_returns, ctx.cov_matrix)

    # 8. Assemble Content
    sidebar_items = [
        {"title": "Estimated Transition Costs (from User Portfolio)", "type": "table_html", "data": transition_html},
        {"title": "Market Regime Capture", "type": "table_html", "data": capture_html}
    ]
    
    main_content = [
        {"title": "Strategy Compositions (by Asset)", "type": "plot", "data": plot_composition_pies(optimal_portfolios)},
        {"title": "Strategy Compositions (by Sector)", "type": "plot", "data": plot_sector_allocation_pies(optimal_portfolios, ctx.friendly_sector_map)},
        {"title": "Portfolio Risk Contribution (by Asset)", "type": "plot", "data": plot_risk_contribution(optimal_portfolios, ctx.mean_returns, ctx.cov_matrix, ctx.friendly_tickers, ctx.risk_free_rate)},
        {"title": "Portfolio Risk Contribution (by Sector)", "type": "plot", "data": plot_sector_risk_contribution(optimal_portfolios, ctx.mean_returns, ctx.cov_matrix, ctx.friendly_tickers, ctx.friendly_sector_map, ctx.risk_free_rate)},
        {"title": "Strategy Cumulative Returns", "type": "plot", "data": plot_cumulative_comparison(cumulative_returns_df, ctx.friendly_benchmark)},
        {"title": "Strategy Drawdown", "type": "plot", "data": plot_drawdown_comparison(drawdown_df, ctx.friendly_benchmark)},
        {"title": "Strategy Rolling Sharpe Ratio", "type": "plot", "data": plot_rolling_sharpe(rolling_sharpe_df, ctx.friendly_benchmark)},
        {"title": "Strategy Monthly Returns Heatmap", "type": "plot", "data": plot_monthly_heatmaps(eval_data, ctx.friendly_benchmark)},
        {"title": "Efficient Frontier", "type": "plot", "data": plot_efficient_frontier(ctx.mean_returns, ctx.cov_matrix, optimal_portfolios, frontier_curve, ctx.risk_free_rate)},
        {"title": "Asset Correlation Heatmap", "type": "plot", "data": plot_correlation_heatmap(ctx.log_returns)},
        {"title": "Strategy Metrics", "type": "metrics_grid", "data": optimal_portfolios}
    ]
    
    sections = [
        {
            "title": "Strategy Discovery & Optimization",
            "description": f"Analysis of multiple optimization algorithms utilizing training data from {ctx.train_start} to {ctx.train_end}.",
            "sidebar": sidebar_items,
            "main_content": main_content
        }
    ]
    
    if bl_plots:
        sections.append({
            "title": "Black-Litterman Detail",
            "description": "Granular breakdown of the subjective market views injected into the BL optimization prior.",
            "main_content": bl_plots
        })

    return sections

def create_optimization_report(portfolio_dict, benchmark_ticker, train_start, train_end, 
                               filename="Optimization_Report.html", **kwargs):
    """
    Standalone entry point to generate the Optimization Report.
    """
    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_optimization_analysis(ctx)
    generate_html_report(sections, title="Optimization Report", filename=filename)
    logger.info("Optimization report complete.")
