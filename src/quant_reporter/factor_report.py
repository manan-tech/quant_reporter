import logging
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)

from .report_context import ReportContext, build_context
from .html_builder import generate_html_report
from .opt_core import get_portfolio_price

from .factor_models import (
    fetch_fama_french_factors,
    run_factor_regression,
    compute_factor_attribution
)
from .attribution import compute_brinson_attribution
from .factor_plotting import (
    plot_factor_loadings,
    plot_cumulative_contributions,
    plot_brinson_attribution,
    plot_rolling_exposures
)

def run_rolling_factor_regression(portfolio_returns: pd.Series, factors: pd.DataFrame, window_days=126):
    """
    Computes rolling factor betas over a specified window.
    Default window is 126 days (~6 months).
    """
    # Align data
    aligned = pd.concat([portfolio_returns, factors], axis=1, join='inner').dropna()
    if len(aligned) < window_days + 10:
        return pd.DataFrame() # Not enough data
        
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]
    
    # We want to regress y on X in a rolling window
    # To do this efficiently in pandas without a loop for OLS:
    # Actually, statsmodels RollingOLS is best, but we'll use a simple loop over numpy indexing
    # to avoid the heavy statsmodels dependency if it's not strictly necessary.
    
    betas = []
    dates = []
    
    Y_arr = y.values
    X_arr = X.values
    
    # Add constant
    X_arr = np.hstack((np.ones((X_arr.shape[0], 1)), X_arr))
    
    for i in range(window_days, len(Y_arr)):
        Y_win = Y_arr[i-window_days:i]
        X_win = X_arr[i-window_days:i]
        
        # OLS: (X^T X)^-1 X^T Y
        try:
            beta = np.linalg.inv(X_win.T @ X_win) @ X_win.T @ Y_win
            betas.append(beta)
            dates.append(aligned.index[i])
        except np.linalg.LinAlgError:
            pass # Skip if singular
            
    if not betas:
        return pd.DataFrame()
        
    col_names = ['Alpha'] + list(X.columns)
    rolling_betas = pd.DataFrame(betas, index=dates, columns=col_names)
    
    # Drop Alpha for pure exposure tracking
    return rolling_betas.drop(columns=['Alpha'])

def analyze_style_drift(rolling_betas: pd.DataFrame):
    """
    Measures how much the factor exposures have drifted from the start of the period to the end.
    """
    if rolling_betas.empty or len(rolling_betas) < 2:
        return pd.DataFrame()
        
    start_exposures = rolling_betas.iloc[0]
    end_exposures = rolling_betas.iloc[-1]
    
    drift = end_exposures - start_exposures
    drift_pct = (drift / np.abs(start_exposures)).replace([np.inf, -np.inf], 0).fillna(0)
    
    df = pd.DataFrame({
        "Initial Exposure": start_exposures,
        "Current Exposure": end_exposures,
        "Absolute Drift": drift,
        "Relative Drift (%)": drift_pct
    })
    return df

def identify_factor_regimes(factors: pd.DataFrame, window_days=63):
    """
    Identifies the recent market regime by looking at trailing factor returns.
    E.g. Is the market rewarding Value or Growth? Small Cap or Large Cap?
    """
    if factors.empty:
         return pd.DataFrame()
    # Looking at the last quarter
    recent_factors = factors.tail(window_days)
    cumulative_returns = (1 + recent_factors).prod() - 1
    
    # Simple logic
    regimes = []
    for factor, ret in cumulative_returns.items():
        if factor == 'Mkt-RF':
            regimes.append({'Factor': 'Market', 'Regime': 'Bull' if ret > 0 else 'Bear', 'Return': ret})
        elif factor == 'SMB':
            regimes.append({'Factor': 'Size', 'Regime': 'Small-Cap Lead' if ret > 0 else 'Large-Cap Lead', 'Return': ret})
        elif factor == 'HML':
            regimes.append({'Factor': 'Value/Growth', 'Regime': 'Value Lead' if ret > 0 else 'Growth Lead', 'Return': ret})
        elif factor == 'RMW':
            regimes.append({'Factor': 'Profitability', 'Regime': 'Robust Lead' if ret > 0 else 'Weak Lead', 'Return': ret})
        elif factor == 'CMA':
            regimes.append({'Factor': 'Investment', 'Regime': 'Conservative Lead' if ret > 0 else 'Aggressive Lead', 'Return': ret})

    return pd.DataFrame(regimes)

def compute_factor_analysis(ctx: ReportContext):
    """
    Computes factor regression and Brinson attribution.
    """
    logger.info("Computing Factor & Attribution Analysis...")
    
    # Provide the portfolio price history based on weights
    user_weights = ctx.user_friendly_weights
    portfolio_prices = get_portfolio_price(ctx.price_data_train[ctx.friendly_tickers], user_weights)
    portfolio_returns = portfolio_prices.pct_change().dropna()
    benchmark_returns = ctx.price_data_train[ctx.friendly_benchmark].pct_change().dropna()

    # 1. Fetch Fama-French
    factor_start = portfolio_returns.index.min().strftime('%Y-%m-%d')
    factor_end = portfolio_returns.index.max().strftime('%Y-%m-%d')
    try:
        factors = fetch_fama_french_factors(
            dataset="F-F_Research_Data_5_Factors_2x3_daily",
            start_date=factor_start,
            end_date=factor_end,
            cache=True
        )
    except Exception as e:
        logger.warning(f"Failed to fetch 5-Factor dataset: {e}. Falling back to 3-Factor.")
        try:
            factors = fetch_fama_french_factors(
                dataset="F-F_Research_Data_Factors_daily",
                start_date=factor_start,
                end_date=factor_end,
                cache=True
            )
        except Exception:
            logger.error("Could not fetch any Fama-French factors.")
            factors = pd.DataFrame()

    # Base sections
    sections = []

    has_ff = not factors.empty
    if has_ff:
        # 2. Factor Regression
        reg_results = run_factor_regression(portfolio_returns, factors)
        attr_results = compute_factor_attribution(portfolio_returns, factors, reg_results['betas'], reg_results['alpha'])
        
        # Plotting uses the imported factor_plotting functions
        plot_exposures = plot_factor_loadings(reg_results)
        plot_attr = plot_cumulative_contributions(attr_results)
        # plot_fit is removed
        plot_fit = None
        
        # New features
        rolling_betas = run_rolling_factor_regression(portfolio_returns, factors)
        style_drift_df = analyze_style_drift(rolling_betas)
        regimes_df = identify_factor_regimes(factors)
        
        rolling_plot = None
        drift_html = ""
        regimes_html = ""
        
        if not rolling_betas.empty:
            rolling_plot = plot_rolling_exposures(rolling_betas)
            drift_html = style_drift_df.map(lambda x: f"{x:.2f}" if isinstance(x, float) else x).to_html(classes='metrics-table')
        if not regimes_df.empty:
            regimes_html = regimes_df.to_html(classes='metrics-table', index=False)
            
        metrics_dict = {
            "Factor Metrics": {
                "R-Squared": f"{reg_results['r_squared']:.1%}",
                "Annualized Alpha": f"{reg_results['alpha'] * 252:.2%}", # Simple annualized
                "Market Beta": f"{reg_results['betas'].get('Mkt-RF', 0):.2f}"
            }
        }
        
        sidebar = [
            {"title": "Regression Summary", "type": "metrics", "data": metrics_dict},
            {"title": "Recent Macro Regimes (Trailing Quarter)", "type": "table_html", "data": regimes_html},
            {"title": "Style Drift (Start to End)", "type": "table_html", "data": drift_html}
        ]
        
        main_content = [
            {"title": "Factor Exposures (Betas)", "type": "plot", "data": plot_exposures},
            {"title": "Rolling Exposures (6-Month)", "type": "plot", "data": rolling_plot} if rolling_plot else {"title": "Rolling Exposures", "type": "text", "data": "Insufficient data."},
            {"title": "Performance Attribution", "type": "plot", "data": plot_attr}
        ]
        
        if plot_fit:
            main_content.append({"title": "Model Fit (Actual vs Predicted)", "type": "plot", "data": plot_fit})
        
        sections.append({
            "title": "Fama-French Factor Analysis",
            "description": "Factor attribution and exposure analysis mapping portfolio returns to systemic market risk premiums.",
            "sidebar": sidebar,
            "main_content": main_content
        })

    # 3. Brinson Attribution
    # We require sector map and benchmark weights
    # Assuming equal benchmark weights for Brinson analysis as fallback
    bench_weights = {t: 1.0/len(ctx.friendly_tickers) for t in ctx.friendly_tickers}
    
    if ctx.friendly_sector_map:
        try:
            asset_returns_df = ctx.price_data_train[ctx.friendly_tickers].pct_change().dropna()
            br_results = compute_brinson_attribution(
                portfolio_weights=user_weights,
                benchmark_weights=bench_weights,
                asset_returns=asset_returns_df,
                sector_map=ctx.friendly_sector_map,
            )
            plot_brinson = plot_brinson_attribution(br_results)
            
            # Metrics from the Total row of the returned DataFrame
            total_row = br_results.loc['Total'] if 'Total' in br_results.index else br_results.iloc[-1]
            alloc_effect = total_row.get('Allocation_Effect', 0)
            sel_effect = total_row.get('Selection_Effect', 0)
            total_active = total_row.get('Excess_Return', alloc_effect + sel_effect)
            
            metrics = {
                "Total Active Return": f"{total_active:.2%}",
                "Allocation Effect": f"{alloc_effect:.2%}",
                "Selection Effect": f"{sel_effect:.2%}"
            }
            
            sections.append({
                "title": "Brinson-Fachler Performance Attribution",
                "description": "Decomposes active portfolio returns into sector allocation and asset selection decisions.",
                "sidebar": [{"title": "Attribution Summary", "type": "metrics", "data": {"Summary": metrics}}],
                "main_content": [
                    {"title": "Attribution by Sector", "type": "plot", "data": plot_brinson}
                ]
            })
        except Exception as e:
            logger.warning(f"Failed to generate Brinson attribution: {e}")
            traceback.print_exc()

    if not sections:
         sections = [{"title": "Factor Analysis", "description": "Insufficient data to perform factor analysis (Fama-French unreachable and Brinson missing inputs).", "main_content": []}]
         
    return sections


def create_factor_report(portfolio_dict, benchmark_ticker, train_start, train_end, 
                         filename="Factor_Report.html", **kwargs):
    ctx = build_context(portfolio_dict, benchmark_ticker, train_start, train_end, **kwargs)
    sections = compute_factor_analysis(ctx)
    generate_html_report(sections, title="Factor Attribution Insight", filename=filename)
    logger.info("Factor report complete.")