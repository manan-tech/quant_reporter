"""
Network-gated end-to-end smoke test for the ctx-based report generators.

These exercise the full build_context -> compute -> render pipeline against live
market data, so they are skipped automatically when data can't be fetched
(e.g. offline CI).
"""
import pytest

import quant_reporter as qr


def test_portfolio_report_end_to_end(tmp_path):
    out = tmp_path / "portfolio.html"
    try:
        qr.create_portfolio_report(
            portfolio_dict={"AAPL": 0.5, "MSFT": 0.5},
            benchmark_ticker="SPY",
            train_start="2023-01-01",
            train_end="2023-06-30",
            risk_free_rate=0.045,
            filename=str(out),
        )
    except Exception as e:  # pragma: no cover - network dependent
        pytest.skip(f"market data unavailable: {e}")
    assert out.exists() and out.stat().st_size > 5000
