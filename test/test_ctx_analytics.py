import numpy as np
import pandas as pd
import pytest
from conftest import make_synthetic_prices
from quant_reporter.report_context import ReportContext
from quant_reporter.opt_core import get_optimization_inputs
from quant_reporter.analytics import PortfolioAnalytics


def _ctx_from_prices(prices, weights, benchmark="BMK", rfr=0.02, rebalance_freq=None):
    tickers = list(weights)
    mean_returns, cov_matrix, log_returns = get_optimization_inputs(prices[tickers])
    ctx = ReportContext(
        full_start="2021-01-01", full_end="2023-12-31",
        train_start="2021-01-01", train_end="2022-12-31",
        test_start="2023-01-01", test_end="2023-12-31",
        portfolio_dict=weights, benchmark_ticker=benchmark,
        display_names=None, sector_map=None, risk_free_rate=rfr,
        sector_caps=None, sector_mins=None,
        bl_views=None, bl_view_confidences=None,
        bl_relative_views=None, bl_relative_view_confidences=None,
        rebalance_freq=rebalance_freq,
        tickers=tickers, friendly_tickers=tickers, friendly_benchmark=benchmark,
        friendly_sector_map=None, user_friendly_weights=weights,
        price_data_full=prices, price_data_train=prices, price_data_test=prices,
        mean_returns=mean_returns, cov_matrix=cov_matrix, log_returns=log_returns,
    )
    ctx.analytics = PortfolioAnalytics(ctx)
    return ctx


def test_ctx_analytics_consistency_and_caching():
    prices = make_synthetic_prices()
    ctx = _ctx_from_prices(prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    # anti-divergence guard: scalar == curve.min()
    assert ctx.analytics.drawdown.max_dd == ctx.analytics.drawdown.curve.min()
    # metrics drawdown == accessor drawdown (same series, computed once)
    assert ctx.analytics.metrics["Max Drawdown"] == pytest.approx(ctx.analytics.drawdown.max_dd)
    # model_stats present and numeric
    assert set(ctx.analytics.model_stats) == {"Expected Return", "Expected Volatility", "Expected Sharpe"}
    # cached: same object on re-access
    assert ctx.analytics.returns is ctx.analytics.returns
