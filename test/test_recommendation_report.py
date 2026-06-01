# test/test_recommendation_report.py
import os
import numpy as np
import pandas as pd
import pytest

from quant_reporter.recommendation import recommend
from quant_reporter.recommendation_report import (
    build_recommendation_section, create_recommendation_report,
)
from quant_reporter.strategy import backtest_many
from quant_reporter.strategies import equal_weight, risk_parity
from conftest import make_synthetic_prices

_OFF = dict(vol_target=99, max_drawdown_limit=99, max_weight=0.99, max_risk_contribution=0.99)


def _full_rec():
    prices = make_synthetic_prices(n_days=700)[["AAA", "BBB", "CCC"]]
    results = backtest_many({"EW": equal_weight, "RP": risk_parity},
                            make_synthetic_prices(n_days=700), benchmark="BMK")
    return recommend(prices, current_weights={"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                     results=results, **_OFF)


def test_build_section_has_four_panels():
    section = build_recommendation_section(_full_rec())
    titles = [item["title"] for item in section["main_content"]]
    for expected in ("Recommended Target Weights", "Rebalance Plan",
                     "Risk Alerts", "Strategy Verdict"):
        assert expected in titles


def test_create_report_writes_file(tmp_path):
    path = str(tmp_path / "reco.html")
    out = create_recommendation_report(_full_rec(), path=path)
    assert out == path and os.path.exists(path)
    html = open(path, encoding="utf-8").read()
    assert "Recommended Target Weights" in html and "Risk Alerts" in html
    assert len(html) > 1000


def test_to_html_delegates(tmp_path):
    path = str(tmp_path / "reco2.html")
    _full_rec().to_html(path)
    assert os.path.exists(path)


def test_report_handles_missing_panels(tmp_path):
    # no current_weights -> no trades; no results -> no verdict
    rec = recommend(make_synthetic_prices(n_days=700)[["AAA", "BBB", "CCC"]], **_OFF)
    path = str(tmp_path / "reco3.html")
    create_recommendation_report(rec, path=path)
    html = open(path, encoding="utf-8").read()
    assert "No trades" in html and "No strategy comparison" in html
