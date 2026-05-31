from conftest import make_synthetic_prices
from quant_reporter.report_context import build_context_from_prices
from quant_reporter.portfolio_report import compute_portfolio_analysis
from quant_reporter.optimization_report import compute_optimization_analysis
from quant_reporter.validation_report import compute_validation_analysis


def _ctx():
    return build_context_from_prices(
        make_synthetic_prices(), {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK",
        train_start="2021-01-01", train_end="2022-12-31",
    )


def test_portfolio_analysis_runs_offline_from_core():
    ctx = _ctx()
    sections = compute_portfolio_analysis(ctx)
    assert isinstance(sections, list) and len(sections) > 0
    # the report must not recompute drawdown independently of the core:
    assert ctx.analytics.drawdown.max_dd == ctx.analytics.drawdown.curve.min()


def test_optimization_analysis_runs_offline_from_core():
    ctx = _ctx()
    sections = compute_optimization_analysis(ctx)
    assert isinstance(sections, list) and len(sections) > 0


def test_validation_analysis_runs_offline_numeric():
    ctx = _ctx()
    sections = compute_validation_analysis(ctx)
    assert isinstance(sections, list) and len(sections) > 0
