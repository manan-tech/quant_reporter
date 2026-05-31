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
