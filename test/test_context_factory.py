import pandas as pd
from conftest import make_synthetic_prices
from quant_reporter.report_context import build_context_from_prices


def test_build_context_from_prices_offline():
    prices = make_synthetic_prices()
    ctx = build_context_from_prices(
        prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK",
        train_start="2021-01-01", train_end="2022-12-31",
    )
    assert ctx.friendly_benchmark == "BMK"
    assert list(ctx.mean_returns.index) == ["AAA", "BBB", "CCC"]
    assert ctx.analytics.metrics["Max Drawdown"] == ctx.analytics.drawdown.max_dd
    assert ctx.price_data_train.index[-1] <= pd.to_datetime("2022-12-31")


def test_build_context_benchmark_is_also_a_holding():
    """Regression: benchmarking a portfolio against one of its own holdings
    (the 60/40 case — hold SPY+AGG, benchmark against SPY) must NOT duplicate the
    benchmark column or inflate the optimization-input asset count.

    Previously ordered_cols = friendly_tickers + [benchmark] produced
    ['AAA','BBB','AAA'], so price_data_train[friendly_tickers] matched the
    duplicated 'AAA' twice -> mean_returns had N=3 for a 2-asset portfolio and
    every weights-vs-assets op raised 'operands could not be broadcast (2,)(3,)'.
    """
    prices = make_synthetic_prices()
    ctx = build_context_from_prices(
        prices, {"AAA": 0.6, "BBB": 0.4}, "AAA",   # AAA is held AND the benchmark
        train_start="2021-01-01", train_end="2022-12-31",
        risk_free_rate=0.03,                        # numeric -> fully offline
    )
    assert not ctx.price_data_full.columns.duplicated().any()
    assert list(ctx.mean_returns.index) == ["AAA", "BBB"]
    assert ctx.cov_matrix.shape == (2, 2)
    assert ctx.friendly_benchmark == "AAA"
    # analytics must compute without a shape mismatch
    assert ctx.analytics.metrics["Max Drawdown"] == ctx.analytics.drawdown.max_dd
