import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .html_builder import generate_html_report
from .analytics import compute_metrics
from .report_helpers import (
    bundle_from_growth,
    compute_drawdown_frame,
    run_standard_optimizers,
)

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

def _rolling_oos_sharpe(price_data, asset_cols, strategies, *, window_years=1,
                        step_months=3, risk_free_rate=0.02, benchmark_col=None,
                        denoise_cov=False, n_components=3, return_schedule=False):
    """Shared walk-forward core.

    `strategies` is an ordered dict ``{name -> fn(train_df, mean, cov) -> weights_dict}``.
    For each rolling train/test window: compute optimization inputs once on the
    train slice, build each strategy's weights, apply them to the test slice, and
    record each strategy's realized OOS Sharpe. Returns a DataFrame indexed by
    "Test Period" with columns "{name} Sharpe" (plus the per-strategy weight
    schedule when ``return_schedule=True``). When ``benchmark_col`` is None, each
    strategy's own growth is used as the bundle benchmark (Realized Sharpe is a
    portfolio-only metric, so this does not affect it).
    """
    results = []
    schedule_rows = {name: {} for name in strategies}

    start_date = price_data.index.min()
    end_date = price_data.index.max()
    current_train_start = start_date
    window_days = int(window_years * 365)
    step_days = int(step_months * 30)

    while current_train_start + pd.Timedelta(days=window_days + step_days) <= end_date:
        train_end = current_train_start + pd.Timedelta(days=window_days)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=step_days)

        train_data = price_data.loc[current_train_start:train_end]
        test_data = price_data.loc[test_start:test_end]
        if train_data.empty or test_data.empty:
            break

        mean_returns, cov_matrix, _ = get_optimization_inputs(
            train_data[asset_cols], denoise_cov=denoise_cov, n_components=n_components)

        try:
            w_dicts = {name: fn(train_data, mean_returns, cov_matrix)
                       for name, fn in strategies.items()}
            for _name, _w in w_dicts.items():
                schedule_rows[_name][test_start] = _w

            window_metrics = {"Test Period":
                              f"{test_start.strftime('%Y-%m')} to {test_end.strftime('%Y-%m')}"}
            bench_growth = None
            if benchmark_col is not None:
                bench_growth = test_data[benchmark_col] / test_data[benchmark_col].iloc[0]

            for name, w_dict in w_dicts.items():
                strat_growth = get_portfolio_price(test_data[asset_cols], w_dict)
                bench = bench_growth if bench_growth is not None else strat_growth
                bundle = bundle_from_growth(strat_growth, bench)
                m = compute_metrics(bundle, risk_free_rate)
                window_metrics[f"{name} Sharpe"] = m.get("Realized Sharpe", np.nan)

            results.append(window_metrics)
        except Exception as e:
            logger.debug(f"Rolling optimization failed for window {current_train_start}: {e}")

        current_train_start += pd.Timedelta(days=step_days)

    rolling_df = pd.DataFrame(results).set_index("Test Period") if results else pd.DataFrame()
    if return_schedule:
        schedule = {
            name: pd.DataFrame.from_dict(rows, orient="index").reindex(
                columns=asset_cols).sort_index()
            for name, rows in schedule_rows.items()
        }
        return rolling_df, schedule
    return rolling_df


def run_rolling_windows(ctx: ReportContext, window_years=1, step_months=3,
                        denoise_cov: bool = False, n_components: int = 3,
                        return_schedule: bool = False,
                        objective=None, profile=None):
    """
    Performs Rolling Window Walk-Forward Validation using a fixed window size.
    Returns a dataframe of out-of-sample Sharpe ratios across all periods.

    When ``return_schedule=True``, additionally returns a per-strategy weight
    schedule as ``(rolling_df, target_weight_schedule)`` for the strategies built
    inside the loop: ``Equal Wt``, ``Min Vol``, ``Max Sharpe``, ``User Portfolio``
    (and ``Recommended`` when ``objective`` or ``profile`` is supplied).

    When ``objective`` or ``profile`` is given, an extra ``Recommended`` strategy
    is validated using the profile-constrained optimize (honoring
    ``max_position_weight``, sector caps, exclusions) and the chosen objective.
    With both None, the strategy set is exactly the historical four (unchanged).
    """
    asset_cols = ctx.friendly_tickers
    num_assets = len(asset_cols)
    bounds_uncon = tuple((0, 1) for _ in range(num_assets))
    cons_uncon = build_constraints(num_assets, ctx.tickers)
    rf = ctx.risk_free_rate

    def _equal(train, mean, cov):
        return {t: 1.0 / num_assets for t in asset_cols}

    def _minvol(train, mean, cov):
        arr = find_optimal_portfolio(objective_min_variance, mean, cov,
                                     bounds_uncon, cons_uncon, rf)
        return {t: w for t, w in zip(asset_cols, arr)}

    def _maxsharpe(train, mean, cov):
        arr = find_optimal_portfolio(objective_neg_sharpe, mean, cov,
                                     bounds_uncon, cons_uncon, rf)
        return {t: w for t, w in zip(asset_cols, arr)}

    def _user(train, mean, cov):
        return ctx.user_friendly_weights

    strategies = {"Equal Wt": _equal, "Min Vol": _minvol,
                  "Max Sharpe": _maxsharpe, "User Portfolio": _user}

    if objective is not None or profile is not None:
        obj = objective or objective_neg_sharpe
        if profile is not None:
            from .planning import apply_constraints
            r_bounds, r_cons = apply_constraints(
                profile, ctx.tickers, sector_map=getattr(ctx, "sector_map", None))
        else:
            r_bounds, r_cons = bounds_uncon, cons_uncon

        def _recommended(train, mean, cov, _obj=obj, _b=r_bounds, _c=r_cons):
            arr = find_optimal_portfolio(_obj, mean, cov, _b, _c, rf)
            return {t: w for t, w in zip(asset_cols, arr)}

        strategies["Recommended"] = _recommended

    return _rolling_oos_sharpe(
        ctx.price_data_full, asset_cols, strategies,
        window_years=window_years, step_months=step_months, risk_free_rate=rf,
        benchmark_col=ctx.friendly_benchmark, denoise_cov=denoise_cov,
        n_components=n_components, return_schedule=return_schedule)


def compute_validation_analysis(ctx: ReportContext):
    """
    Computes analysis explicitly for the validation and walk-forward report.
    """
    logger.info("Computing Validation Analysis...")
    
    # --- IS Optimization using ctx inputs (shared with the optimization report) ---
    arrays = run_standard_optimizers(ctx)

    def _to_dict(arr):
        return {t: w for t, w in zip(ctx.friendly_tickers, arr)}

    weights = {
        "Equal Wt": _to_dict(arrays["equal"]),
        "Min Vol": _to_dict(arrays["min_vol"]),
        "Balanced (40% Cap)": _to_dict(arrays["balanced"]),
        "Max Sharpe": _to_dict(arrays["max_sharpe"]),
        "User Portfolio": ctx.user_friendly_weights
    }

    if "sector" in arrays:
        weights["Sector Balanced"] = _to_dict(arrays["sector"])

    # --- IS Performance (numeric metrics via analytics core) ---
    in_sample_results = {}
    is_bench_growth = ctx.price_data_train[ctx.friendly_benchmark] / ctx.price_data_train[ctx.friendly_benchmark].iloc[0]
    # Build cumulative-return frame for plotting (benchmark + all strategies)
    in_sample_eval_data = pd.DataFrame({ctx.friendly_benchmark: is_bench_growth})

    for name, w_dict in weights.items():
        strat_growth = get_portfolio_price(ctx.price_data_train[ctx.friendly_tickers], w_dict)
        in_sample_eval_data[name] = strat_growth
        bundle = bundle_from_growth(strat_growth, is_bench_growth)
        in_sample_results[name] = compute_metrics(bundle, ctx.risk_free_rate)

    # --- OOS Performance (numeric metrics via analytics core) ---
    out_sample_results = {}
    oos_bench_growth = ctx.price_data_test[ctx.friendly_benchmark] / ctx.price_data_test[ctx.friendly_benchmark].iloc[0]
    # Build cumulative-return frame for plotting (benchmark + all strategies)
    out_sample_eval_data = pd.DataFrame({ctx.friendly_benchmark: oos_bench_growth})

    for name, w_dict in weights.items():
        strat_growth = get_portfolio_price(ctx.price_data_test[ctx.friendly_tickers], w_dict)
        out_sample_eval_data[name] = strat_growth
        bundle = bundle_from_growth(strat_growth, oos_bench_growth)
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
    drawdown_df = compute_drawdown_frame(cumulative_returns_df)


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
