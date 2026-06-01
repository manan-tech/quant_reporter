# test/test_backtest_report.py
import os
import pandas as pd
import plotly.graph_objects as go
import pytest

from quant_reporter.strategy import backtest, backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from quant_reporter.backtest_report import (
    plot_wealth, plot_drawdown, plot_weights, plot_turnover, plot_rolling,
    build_sections, create_backtest_report,
)
from conftest import make_synthetic_prices


def _res():
    prices = make_synthetic_prices(n_days=600)
    return backtest(equal_weight, prices, benchmark="BMK", rebalance="M")


def test_plot_functions_return_figures_with_data():
    res = _res()
    for fn in (plot_wealth, plot_drawdown, plot_weights, plot_turnover, plot_rolling):
        fig = fn(res)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1


def test_plot_wealth_includes_benchmark_trace():
    res = _res()
    names = [t.name for t in plot_wealth(res).data]
    assert any("Benchmark" in (n or "") for n in names)


def test_build_sections_has_all_panels():
    res = _res()
    sections = build_sections(res)
    titles = [item.get("title") for s in sections for item in s["main_content"]]
    for expected in ("Growth of $1", "Underwater Drawdown", "Weights Over Time",
                     "Turnover per Rebalance", "Rolling Sharpe & Vol",
                     "Performance Metrics", "Trade Blotter"):
        assert expected in titles


def test_create_report_writes_file(tmp_path):
    res = _res()
    path = str(tmp_path / "bt.html")
    out = create_backtest_report(res, path=path)
    assert out == path
    assert os.path.exists(path)
    html = open(path, encoding="utf-8").read()
    assert "Growth of $1" in html and len(html) > 1000


def test_multi_strategy_report_has_comparison(tmp_path):
    prices = make_synthetic_prices(n_days=600)
    results = backtest_many({"EW": equal_weight, "RP": risk_parity}, prices, benchmark="BMK")
    path = str(tmp_path / "cmp.html")
    create_backtest_report(results, path=path)
    html = open(path, encoding="utf-8").read()
    assert "OOS Strategy Comparison" in html


def test_open_browser_uses_absolute_file_url(tmp_path, monkeypatch):
    opened = {}
    monkeypatch.setattr("webbrowser.open", lambda url: opened.setdefault("url", url) or True)
    res = _res()
    path = str(tmp_path / "bt.html")
    create_backtest_report(res, path=path, open_browser=True)
    # Must be an absolute file URL (file:///...), not a relative one.
    assert opened["url"].startswith("file:///")
