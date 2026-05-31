import logging
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .metrics import compute_drawdown
from .html_builder import generate_html_report
from .analytics import ReturnsBundle, compute_metrics

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


def _bundle_from_growth(strategy_growth, benchmark_growth):
    """Build a ReturnsBundle from two Growth-of-$1 series (portfolio and benchmark)."""
    growth = pd.concat({"Portfolio": strategy_growth, "Benchmark": benchmark_growth}, axis=1).dropna()
    return ReturnsBundle(daily=growth.pct_change().dropna(), growth=growth, weights_history=None)


def calculate_overfitting_score(is_metrics, oos_metrics):
    """
    Calculates Overfitting Score and Strategy Degradation.
    Overfitting Score = (IS_Sharpe - OOS_Sharpe) / IS_Sharpe
    Strategy Degradation = (IS_CAGR - OOS_CAGR) / IS_CAGR

    Expects numeric metric dicts (from compute_metrics) keyed by
    "Realized Sharpe" and "Realized CAGR".
    """
    scores = {}
    for portfolio_name in is_metrics.keys():
        is_sharpe = is_metrics[portfolio_name].get("Realized Sharpe")
        oos_sharpe = oos_metrics[portfolio_name].get("Realized Sharpe")
        is_cagr = is_metrics[portfolio_name].get("Realized CAGR", 0.0)
        oos_cagr = oos_metrics[portfolio_name].get("Realized CAGR", 0.0)

        if is_sharpe and is_sharpe > 0 and oos_sharpe is not None:
            overfitting_score = max(0, (is_sharpe - oos_sharpe) / is_sharpe)
        else:
            overfitting_score = 0

        if is_cagr and is_cagr > 0 and oos_cagr is not None:
            degradation = max(0, (is_cagr - oos_cagr) / is_cagr)
        else:
            degradation = 0
            
        scores[portfolio_name] = {
            "Overfitting Score": overfitting_score,
            "Strategy Degradation": degradation
        }
    return pd.DataFrame.from_dict(scores, orient='index')

def run_rolling_windows(ctx: ReportContext, window_years=1, step_months=3,
                        denoise_cov: bool = False, n_components: int = 3,
                        return_schedule: bool = False):
    """
    Performs Rolling Window Walk-Forward Validation using a fixed window size.
    Returns a dataframe of out-of-sample Sharpe ratios across all periods.

    When ``return_schedule=True``, additionally returns a per-strategy weight
    schedule as ``(rolling_df, target_weight_schedule)`` where
    ``target_weight_schedule`` is ``dict[strategy_name -> DataFrame]`` (index = each
    window's OOS start date, columns = ``ctx.friendly_tickers``, values = that
    window's weights) for the strategies built inside the loop: ``Equal Wt``,
    ``Min Vol``, ``Max Sharpe``, ``User Portfolio``.
    """
    results = []
    schedule_rows = {"Equal Wt": {}, "Min Vol": {}, "Max Sharpe": {}, "User Portfolio": {}}

    start_date = ctx.price_data_full.index.min()
    end_date = ctx.price_data_full.index.max()
    
    current_train_start = start_date
    window_days = int(window_years * 365)
    step_days = int(step_months * 30)
    
    num_assets = len(ctx.friendly_tickers)
    
    while current_train_start + pd.Timedelta(days=window_days+step_days) <= end_date:
        train_end = current_train_start + pd.Timedelta(days=window_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=step_days)
        
        train_data = ctx.price_data_full.loc[current_train_start:train_end]
        test_data = ctx.price_data_full.loc[test_start:test_end]
        
        if train_data.empty or test_data.empty:
            break
            
        mean_returns, cov_matrix, _ = get_optimization_inputs(
                train_data[ctx.friendly_tickers],
                denoise_cov=denoise_cov,
                n_components=n_components,
            )
        
        bounds_uncon = tuple((0, 1) for _ in range(num_assets))
        cons_uncon = build_constraints(num_assets, ctx.tickers)
        
        try:
            min_vol_arr = find_optimal_portfolio(objective_min_variance, mean_returns, cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
            max_sharpe_arr = find_optimal_portfolio(objective_neg_sharpe, mean_returns, cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
            equal_arr = np.array([1./num_assets] * num_assets)
            
            w_dicts = {
                "Equal Wt": {t: w for t, w in zip(ctx.friendly_tickers, equal_arr)},
                "Min Vol": {t: w for t, w in zip(ctx.friendly_tickers, min_vol_arr)},
                "Max Sharpe": {t: w for t, w in zip(ctx.friendly_tickers, max_sharpe_arr)},
                "User Portfolio": ctx.user_friendly_weights
            }

            for _name, _w in w_dicts.items():
                schedule_rows[_name][test_start] = _w

            window_metrics = {"Test Period": f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"}
            
            bench_growth = (test_data[ctx.friendly_benchmark] / test_data[ctx.friendly_benchmark].iloc[0])
            for name, w_dict in w_dicts.items():
                strat_growth = get_portfolio_price(test_data[ctx.friendly_tickers], w_dict)
                bundle = _bundle_from_growth(strat_growth, bench_growth)
                m = compute_metrics(bundle, ctx.risk_free_rate)
                window_metrics[f"{name} Sharpe"] = m.get("Realized Sharpe", np.nan)
                
            results.append(window_metrics)
        except Exception as e:
            logger.debug(f"Rolling optimization failed for window {current_train_start}: {e}")
            
        current_train_start += pd.Timedelta(days=step_days)
    
    rolling_df = pd.DataFrame(results).set_index("Test Period") if results else pd.DataFrame()
    if return_schedule:
        schedule = {
            name: pd.DataFrame.from_dict(rows, orient="index").reindex(
                columns=ctx.friendly_tickers).sort_index()
            for name, rows in schedule_rows.items()
        }
        return rolling_df, schedule
    return rolling_df


def compute_validation_analysis(ctx: ReportContext):
    """
    Computes analysis explicitly for the validation and walk-forward report.
    """
    logger.info("Computing Validation Analysis...")
    
    num_assets = len(ctx.tickers)
    
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, ctx.tickers)
    bounds_bal = tuple((0, 0.40) for _ in range(num_assets))
    cons_bal = build_constraints(num_assets, ctx.tickers)
    cons_sector = build_constraints(num_assets, ctx.tickers, ctx.sector_map, ctx.sector_caps, ctx.sector_mins)
    
    # --- IS Optimization using ctx inputs ---
    min_vol_weights_arr = find_optimal_portfolio(objective_min_variance, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
    max_sharpe_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_uncon, ctx.risk_free_rate)
    bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_bal, cons_bal, ctx.risk_free_rate)
    equal_weights_arr = np.array([1./num_assets] * num_assets)
    
    weights = {
        "Equal Wt": {t: w for t, w in zip(ctx.friendly_tickers, equal_weights_arr)},
        "Min Vol": {t: w for t, w in zip(ctx.friendly_tickers, min_vol_weights_arr)},
        "Balanced (40% Cap)": {t: w for t, w in zip(ctx.friendly_tickers, bal_weights_arr)},
        "Max Sharpe": {t: w for t, w in zip(ctx.friendly_tickers, max_sharpe_weights_arr)},
        "User Portfolio": ctx.user_friendly_weights
    }

    if ctx.sector_map and (ctx.sector_caps or ctx.sector_mins):
        sec_bal_weights_arr = find_optimal_portfolio(objective_neg_sharpe, ctx.mean_returns, ctx.cov_matrix, bounds_uncon, cons_sector, ctx.risk_free_rate)
        weights["Sector Balanced"] = {t: w for t, w in zip(ctx.friendly_tickers, sec_bal_weights_arr)}

    # --- IS Performance (numeric metrics via analytics core) ---
    in_sample_results = {}
    is_bench_growth = ctx.price_data_train[ctx.friendly_benchmark] / ctx.price_data_train[ctx.friendly_benchmark].iloc[0]
    # Build cumulative-return frame for plotting (benchmark + all strategies)
    in_sample_eval_data = pd.DataFrame({ctx.friendly_benchmark: is_bench_growth})

    for name, w_dict in weights.items():
        strat_growth = get_portfolio_price(ctx.price_data_train[ctx.friendly_tickers], w_dict)
        in_sample_eval_data[name] = strat_growth
        bundle = _bundle_from_growth(strat_growth, is_bench_growth)
        in_sample_results[name] = compute_metrics(bundle, ctx.risk_free_rate)

    # --- OOS Performance (numeric metrics via analytics core) ---
    out_sample_results = {}
    oos_bench_growth = ctx.price_data_test[ctx.friendly_benchmark] / ctx.price_data_test[ctx.friendly_benchmark].iloc[0]
    # Build cumulative-return frame for plotting (benchmark + all strategies)
    out_sample_eval_data = pd.DataFrame({ctx.friendly_benchmark: oos_bench_growth})

    for name, w_dict in weights.items():
        strat_growth = get_portfolio_price(ctx.price_data_test[ctx.friendly_tickers], w_dict)
        out_sample_eval_data[name] = strat_growth
        bundle = _bundle_from_growth(strat_growth, oos_bench_growth)
        out_sample_results[name] = compute_metrics(bundle, ctx.risk_free_rate)

    # --- Combine and Score ---
    final_results = []
    # Keys match compute_metrics numeric dict; values are display column names
    metrics_to_show = {
        "Realized CAGR": "CAGR",
        "Realized Volatility": "Volatility",
        "Realized Sharpe": "Sharpe Ratio",
        "Max Drawdown": "Max Drawdown",
    }

    for name in weights.keys():
        row = {"Portfolio": name}
        for key, short_name in metrics_to_show.items():
            row[f"In-Sample {short_name}"] = in_sample_results[name].get(key)
            row[f"Out-of-Sample {short_name}"] = out_sample_results[name].get(key)
        final_results.append(row)

    results_df = pd.DataFrame(final_results).set_index("Portfolio")
    # Values are already numeric floats — no string round-trip needed
    validation_table_html = results_df.to_html(classes='metrics-table', float_format='{:.2%}'.format)

    # Overfitting
    overfitting_df = calculate_overfitting_score(in_sample_results, out_sample_results)
    overfitting_html = overfitting_df.to_html(classes='metrics-table', float_format='{:.2%}'.format)

    # Rolling Windows (use same denoise settings as in-sample optimization)
    _denoise_cov = getattr(ctx, "denoise_cov", False)
    _n_components = getattr(ctx, "n_components", 3)
    rolling_df = run_rolling_windows(ctx, denoise_cov=_denoise_cov, n_components=_n_components)
    rolling_html = "<p><i>Insufficient data for rolling windows</i></p>"
    if not rolling_df.empty:
        rolling_html = rolling_df.to_html(classes='metrics-table', float_format='{:.2f}'.format)

    # --- Plotting ---
    cumulative_returns_df = out_sample_eval_data
    drawdown_df = pd.DataFrame()
    for col in cumulative_returns_df.columns:
        drawdown_df[col] = compute_drawdown(cumulative_returns_df[col]).curve
        
    sections = [{
        "title": "Walk-Forward Validation Dashboard",
        "description": f"Model robusted with IS window {ctx.train_start} to {ctx.train_end}, testing on strictly OOS data {ctx.test_start} to {ctx.test_end}.",
        "sidebar": [
            {"title": "Overfitting & Degradation", "type": "table_html", "data": overfitting_html},
            {"title": "Rolling Test OOS Sharpe", "type": "table_html", "data": rolling_html}
        ],
        "main_content": [
            {"title": "In-Sample vs. Out-of-Sample Results", "type": "table_html", "data": validation_table_html},
            {"title": "Out-of-Sample Cumulative Returns", "type": "plot", "data": plot_cumulative_comparison(cumulative_returns_df, ctx.friendly_benchmark)},
            {"title": "Out-of-Sample Drawdown", "type": "plot", "data": plot_drawdown_comparison(drawdown_df, ctx.friendly_benchmark)}
        ]
    }]
    
    return sections


def create_validation_report(portfolio_dict, benchmark_ticker, train_start, train_end, 
                             filename="Validation_Report.html", **kwargs):
    """
    Standalone entry point to generate the Validation Report.
    """
    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_validation_analysis(ctx)
    generate_html_report(sections, title="Validation Report", filename=filename)
    logger.info("Validation report complete.")
