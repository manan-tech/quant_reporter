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


def _error_section(name, exc):
    """Return a visible error section dict for a failed delegate module."""
    return {
        "title": f"⚠ {name} — module failed",
        "description": f"This module raised an error and was skipped: {type(exc).__name__}: {exc}",
        "sidebar": [],
        "main_content": [
            {
                "title": "Error",
                "type": "text_html",
                "data": f"<pre>{type(exc).__name__}: {exc}</pre>",
            }
        ],
    }


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
        ],
    }


def _dedupe_heatmap_sections(sections):
    """
    Remove duplicate correlation-heatmap plot items from the combined section list.
    The first heatmap (by title containing 'Correlation Heatmap' or 'Correlation') encountered
    is kept; subsequent identical-title plot items are dropped.
    Only the main_content plot items are deduped; sidebar items are left alone.
    """
    seen_plot_titles = set()
    result = []
    for section in sections:
        new_main = []
        for item in section.get("main_content", []):
            title = item.get("title", "")
            # Identify heatmap items by a keyword in the title
            if "Correlation" in title and "Heatmap" in title:
                if title in seen_plot_titles:
                    # Skip duplicate
                    continue
                seen_plot_titles.add(title)
            new_main.append(item)
        # Rebuild section with deduplicated main_content
        deduped = dict(section)
        if "main_content" in section:
            deduped["main_content"] = new_main
        result.append(deduped)
    return result


def _assemble_combined_sections(
    ctx,
    *,
    strict=False,
    num_simulations=5000,
    time_horizon=252,
    initial_investment=10000,
    seed=42,
):
    """
    Assemble and return the list of report sections from all delegate modules.

    This helper is factored out so it can be tested without triggering HTML file I/O.

    Parameters
    ----------
    ctx : ReportContext
        Pre-built analysis context.
    strict : bool
        When True, any delegate exception re-raises immediately instead of producing
        a visible error section.
    num_simulations, time_horizon, initial_investment, seed
        Forwarded to compute_monte_carlo_analysis.
    """
    all_sections = []

    # Executive Summary
    all_sections.append(_build_executive_summary(ctx))

    # Portfolio Analysis Module
    try:
        all_sections.extend(compute_portfolio_analysis(ctx))
    except Exception as e:
        logger.error(f"Portfolio Analysis failed: {e}")
        if strict:
            raise
        all_sections.append(_error_section("Portfolio Analysis", e))

    # Factor Attribution Module
    try:
        all_sections.extend(compute_factor_analysis(ctx))
    except Exception as e:
        logger.error(f"Factor Analysis failed: {e}")
        if strict:
            raise
        all_sections.append(_error_section("Factor Analysis", e))

    # Optimization Module
    try:
        all_sections.extend(compute_optimization_analysis(ctx))
    except Exception as e:
        logger.error(f"Optimization Analysis failed: {e}")
        if strict:
            raise
        all_sections.append(_error_section("Optimization Analysis", e))

    # Walk-Forward Validation Module
    try:
        all_sections.extend(compute_validation_analysis(ctx))
    except Exception as e:
        logger.error(f"Validation Analysis failed: {e}")
        if strict:
            raise
        all_sections.append(_error_section("Validation Analysis", e))

    # Monte Carlo Module
    try:
        all_sections.extend(
            compute_monte_carlo_analysis(
                ctx,
                num_simulations=num_simulations,
                time_horizon=time_horizon,
                initial_investment=initial_investment,
                seed=seed,
            )
        )
    except Exception as e:
        logger.error(f"Monte Carlo Analysis failed: {e}")
        if strict:
            raise
        all_sections.append(_error_section("Monte Carlo Analysis", e))

    # De-duplicate correlation heatmap sections (combined report only)
    all_sections = _dedupe_heatmap_sections(all_sections)

    return all_sections


def create_combined_report(
    portfolio_dict,
    benchmark_ticker,
    train_start,
    train_end,
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
    rebalance_freq=None,
    denoise_cov=False,
    n_components=3,
    strict=False,
    num_simulations=5000,
    time_horizon=252,
    initial_investment=10000,
    seed=42,
):
    """
    Generates a single, combined HTML report for portfolio analysis,
    optimization, walk-forward validation, monte carlo, and factor attribution.

    Parameters
    ----------
    strict : bool
        When True, any delegate module failure re-raises immediately instead of
        producing a visible error section and continuing.
    num_simulations, time_horizon, initial_investment, seed
        Monte Carlo parameters forwarded to compute_monte_carlo_analysis.
    """
    logger.info("Initiating Combined Report generation...")

    # Build a single ReportContext to fetch data and prepare inputs exactly once
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
            n_components=n_components,
        )
    except Exception as e:
        logger.error(f"Failed to build analysis context: {e}")
        raise

    all_sections = _assemble_combined_sections(
        ctx,
        strict=strict,
        num_simulations=num_simulations,
        time_horizon=time_horizon,
        initial_investment=initial_investment,
        seed=seed,
    )

    # Generate HTML
    generate_html_report(all_sections, title="Comprehensive Quant Report", filename=filename)
    logger.info(f"Combined Report Generated Successfully: {filename}")
