import logging

logger = logging.getLogger(__name__)

from .report_context import build_context
from .html_builder import generate_html_report

# Import component-level compute functions
from .portfolio_report import compute_portfolio_analysis
from .optimization_report import compute_optimization_analysis
from .validation_report import compute_validation_analysis
from .monte_carlo_report import compute_monte_carlo_analysis
from .factor_report import compute_factor_analysis

def _build_executive_summary(ctx):
    """
    Builds a high-level executive summary summarizing the portfolio and its parameters.
    """
    weights_html_rows = []
    for ticker, weight in ctx.user_friendly_weights.items():
        weights_html_rows.append(f"<tr><td>{ticker}</td><td>{weight:.1%}</td></tr>")
        
    weights_html = f"""
    <table class="metrics-table">
        <tr><th>Asset</th><th>Weight</th></tr>
        {"".join(weights_html_rows)}
    </table>
    """
    
    summary_html = f"""
    <div style="font-size: 1.1em; line-height: 1.6;">
        <p><strong>Benchmark:</strong> {ctx.friendly_benchmark}</p>
        <p><strong>Analysis Period:</strong> {ctx.full_start} to {ctx.full_end}</p>
        <p><strong>Risk-Free Rate Assumption:</strong> {ctx.risk_free_rate:.2%}</p>
    </div>
    """
    
    return {
        "title": "Executive Summary",
        "description": "High-level overview of the constructed portfolio and foundational analysis parameters.",
        "sidebar": [
            {"title": "Initial Portfolio Composition", "type": "table_html", "data": weights_html}
        ],
        "main_content": [
            {"title": "Report Configuration", "type": "text_html", "data": summary_html}
        ]
    }

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
    optimization, walk-forward validation, monte carlo, and factor attribution.
    """
    logger.info("Initiating Combined Report generation...")
    
    # 1. Build a single ReportContext to fetch data and prepare inputs exactly once
    try:
        ctx = build_context(
            portfolio_dict=portfolio_dict,
            benchmark_ticker=benchmark_ticker,
            train_start=train_start,
            train_end=train_end,
            risk_free_rate=risk_free_rate,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
            sector_mins=sector_mins,
            bl_views=bl_views,
            bl_view_confidences=bl_view_confidences,
            bl_relative_views=bl_relative_views,
            bl_relative_view_confidences=bl_relative_view_confidences,
            rebalance_freq=rebalance_freq,
            denoise_cov=denoise_cov,
            n_components=n_components
        )
    except Exception as e:
        logger.error(f"Failed to build analysis context: {e}")
        raise

    all_sections = []
    
    # 2. Executive Summary
    all_sections.append(_build_executive_summary(ctx))
    
    # 3. Portfolio Analysis Module
    try:
        all_sections.extend(compute_portfolio_analysis(ctx))
    except Exception as e:
        logger.error(f"Portfolio Analysis failed: {e}")
        
    # 4. Factor Attribution Module
    try:
        all_sections.extend(compute_factor_analysis(ctx))
    except Exception as e:
        logger.error(f"Factor Analysis failed: {e}")

    # 5. Optimization Module
    try:
        all_sections.extend(compute_optimization_analysis(ctx))
    except Exception as e:
        logger.error(f"Optimization Analysis failed: {e}")

    # 6. Walk-Forward Validation Module
    try:
        all_sections.extend(compute_validation_analysis(ctx))
    except Exception as e:
        logger.error(f"Validation Analysis failed: {e}")

    # 7. Monte Carlo Module
    try:
        all_sections.extend(compute_monte_carlo_analysis(ctx))
    except Exception as e:
        logger.error(f"Monte Carlo Analysis failed: {e}")

    # 8. Generate HTML
    generate_html_report(all_sections, title="Comprehensive Quant Report", filename=filename)
    logger.info(f"Combined Report Generated Successfully: {filename}")