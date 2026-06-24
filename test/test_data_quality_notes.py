"""Offline tests for the Data Quality Notes feature (GH #4).

Reports silently apply fallbacks that change what the user asked for (dropping a
ticker that failed to download and renormalizing the remaining weights, or
falling back to the 2% default risk-free rate). These tests assert that those
fallbacks are surfaced in the report instead of only in logs.
"""

from conftest import make_synthetic_prices
from quant_reporter.providers import DEFAULT_RISK_FREE_RATE
from quant_reporter.report_context import (
    build_context,
    build_context_from_prices,
    data_quality_section,
)
from quant_reporter.portfolio_report import create_portfolio_report
import quant_reporter.combined_report as cr


PORTFOLIO = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}


class _DroppingProvider:
    """Offline fixture provider that omits one holding's price column.

    Exercises the drop-ticker + renormalize-weights fallback without any
    network access. ``rfr`` controls the risk-free rate returned for "auto".
    """

    def __init__(self, drop="CCC", rfr=DEFAULT_RISK_FREE_RATE):
        self._drop = drop
        self._rfr = rfr

    def get_prices(self, tickers, start, end):
        prices = make_synthetic_prices()  # AAA, BBB, CCC, BMK
        return prices[[c for c in prices.columns if c != self._drop]]

    def get_risk_free_rate(self):
        return self._rfr


def _dropping_ctx(**kw):
    return build_context(
        PORTFOLIO, "BMK", train_start="2021-01-01", train_end="2022-12-31",
        data_provider=_DroppingProvider(**kw),
    )


# ---------------------------------------------------------------------------
# Context-level: the fallbacks are recorded on ctx.data_quality
# ---------------------------------------------------------------------------

def test_dropped_ticker_recorded_and_weights_renormalized():
    ctx = _dropping_ctx()
    dq = ctx.data_quality

    assert dq.dropped_tickers == ["CCC"]
    # AAA/BBB rescaled from 50/30 (of the original 100%) to sum to 100%.
    assert set(dq.weight_adjustments) == {"AAA", "BBB"}
    old_aaa, new_aaa = dq.weight_adjustments["AAA"]
    assert old_aaa == 0.5
    assert new_aaa == 0.5 / 0.8  # 0.625
    assert abs(sum(new for _, new in dq.weight_adjustments.values()) - 1.0) < 1e-9
    assert dq.has_notes()


def test_rfr_fallback_flagged_only_on_default():
    # Provider returns the 2% default -> flagged as a fallback under "auto".
    assert _dropping_ctx(rfr=DEFAULT_RISK_FREE_RATE).data_quality.rfr_fallback is True
    # A live-looking rate is not flagged.
    assert _dropping_ctx(rfr=0.045).data_quality.rfr_fallback is False


def test_clean_inputs_produce_no_section():
    # Full data + explicit numeric rfr => no fallbacks => no banner.
    ctx = build_context_from_prices(
        make_synthetic_prices(), PORTFOLIO, "BMK",
        train_start="2021-01-01", train_end="2022-12-31", risk_free_rate=0.03,
    )
    assert ctx.data_quality.has_notes() is False
    assert data_quality_section(ctx) is None


# ---------------------------------------------------------------------------
# Rendering: the section is built and carries the right content
# ---------------------------------------------------------------------------

def test_data_quality_section_content():
    section = data_quality_section(_dropping_ctx())
    assert section is not None
    assert section["title"] == "Data Quality Notes"
    html = section["main_content"][0]["data"]
    assert "Dropped tickers" in html and "CCC" in html
    assert "Weights renormalized" in html
    assert "62.5%" in html  # 0.5 / 0.8 rendered
    assert "Risk-free rate" in html


# ---------------------------------------------------------------------------
# Integration: the note appears in the generated report HTML
# ---------------------------------------------------------------------------

def test_portfolio_report_html_shows_notes(tmp_path):
    out = tmp_path / "Portfolio_Report.html"
    create_portfolio_report(
        PORTFOLIO, "BMK", "2021-01-01", "2022-12-31",
        filename=str(out), data_provider=_DroppingProvider(),
    )
    html = out.read_text(encoding="utf-8")
    assert "Data Quality Notes" in html
    assert "Dropped tickers" in html and "CCC" in html
    assert "Weights renormalized" in html
    assert "Risk-free rate" in html


def test_combined_report_includes_data_quality_section(monkeypatch):
    # Patch the factor delegate to avoid network Fama-French fetches.
    monkeypatch.setattr(
        cr, "compute_factor_analysis",
        lambda ctx: (_ for _ in ()).throw(RuntimeError("no FF")),
    )
    sections = cr._assemble_combined_sections(_dropping_ctx(), strict=False)
    titles = [s["title"] for s in sections]
    assert "Data Quality Notes" in titles
    # It should lead the report (before the Executive Summary).
    assert titles.index("Data Quality Notes") < titles.index("Executive Summary")
