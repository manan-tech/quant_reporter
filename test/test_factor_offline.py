import numpy as np
import pandas as pd
import quant_reporter.factor_report as factor_report
from conftest import make_synthetic_prices
from quant_reporter.report_context import build_context_from_prices
from quant_reporter.factor_report import compute_factor_analysis


def _fake_ff_factors(*args, **kwargs):
    # Matches the schema produced by factor_models._download_fama_french:
    # columns 'Mkt-RF','SMB','HML','RF' in DECIMALS (not percent), daily DatetimeIndex.
    # run_factor_regression requires 'Mkt-RF' (+ optional SMB/HML) and reads 'RF'.
    idx = pd.bdate_range("2021-01-01", periods=756)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0004, 0.01, 756),
            "SMB": rng.normal(0, 0.005, 756),
            "HML": rng.normal(0, 0.005, 756),
            "RF": np.full(756, 0.02 / 252),
        },
        index=idx,
    )


def test_factor_analysis_offline(monkeypatch):
    # compute_factor_analysis calls the name imported INTO factor_report, so we patch
    # factor_report.fetch_fama_french_factors (not factor_models.*).
    monkeypatch.setattr(factor_report, "fetch_fama_french_factors", _fake_ff_factors)
    ctx = build_context_from_prices(
        make_synthetic_prices(),
        {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
        "BMK",
        train_start="2021-01-01",
        train_end="2022-12-31",
    )
    sections = compute_factor_analysis(ctx)
    assert isinstance(sections, list) and len(sections) > 0


def test_factor_analysis_offline_with_sectors(monkeypatch):
    # With a sector map but no benchmark_weights, the Brinson section must be present
    # and HONESTLY labeled as "vs Equal-Weight Baseline".
    monkeypatch.setattr(factor_report, "fetch_fama_french_factors", _fake_ff_factors)
    ctx = build_context_from_prices(
        make_synthetic_prices(),
        {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
        "BMK",
        train_start="2021-01-01",
        train_end="2022-12-31",
        sector_map={"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"},
    )
    sections = compute_factor_analysis(ctx)
    titles = [s["title"] for s in sections]
    attr_titles = [t for t in titles if "Attribution" in t]
    assert attr_titles, f"expected an attribution section, got {titles}"
    # No benchmark weights supplied -> must not claim benchmark-relative; must say baseline.
    assert any("Equal-Weight" in t for t in attr_titles), attr_titles


def test_factor_analysis_offline_with_benchmark_weights(monkeypatch):
    # When real benchmark weights ARE supplied, run the true benchmark-relative Brinson.
    monkeypatch.setattr(factor_report, "fetch_fama_french_factors", _fake_ff_factors)
    ctx = build_context_from_prices(
        make_synthetic_prices(),
        {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
        "BMK",
        train_start="2021-01-01",
        train_end="2022-12-31",
        sector_map={"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"},
    )
    ctx.benchmark_weights = {"AAA": 0.33, "BBB": 0.33, "CCC": 0.34}
    sections = compute_factor_analysis(ctx)
    attr_titles = [s["title"] for s in sections if "Attribution" in s["title"]]
    assert attr_titles
    assert any("Equal-Weight" not in t for t in attr_titles), attr_titles
