import pytest
import quant_reporter.combined_report as cr
from conftest import make_synthetic_prices
from quant_reporter.report_context import build_context_from_prices


def _ctx():
    return build_context_from_prices(
        make_synthetic_prices(), {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}, "BMK",
        train_start="2021-01-01", train_end="2022-12-31",
    )


def _sections_with_failing_factor(monkeypatch, **kw):
    """
    Patch factor delegate to raise so we don't need network FF data.
    The other four modules run offline. Returns assembled sections list.
    """
    monkeypatch.setattr(
        cr, "compute_factor_analysis",
        lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    return cr._assemble_combined_sections(_ctx(), **kw)


# ---------------------------------------------------------------------------
# Test 1: non-strict — failing factor yields a visible error section, others present
# ---------------------------------------------------------------------------

def test_combined_failloud_section(monkeypatch):
    sections = _sections_with_failing_factor(monkeypatch, strict=False)

    # Find titles of all sections
    titles = [s["title"] for s in sections]

    # An error section for Factor Analysis must be present
    error_titles = [t for t in titles if "Factor Analysis" in t and "module failed" in t]
    assert error_titles, f"Expected an error section for Factor Analysis, got: {titles}"

    # Executive Summary must still be present
    assert any("Executive Summary" in t for t in titles), "Executive Summary missing"

    # The error section must contain useful content
    error_section = next(s for s in sections if "Factor Analysis" in s["title"] and "module failed" in s["title"])
    assert "RuntimeError" in error_section["description"]
    assert "boom" in error_section["description"]
    assert error_section["main_content"][0]["type"] == "text_html"
    assert "RuntimeError" in error_section["main_content"][0]["data"]

    # Other modules (portfolio, optimization, validation, monte carlo) should still contribute
    # At least one section from portfolio (2 sections) and one from optimization / validation / MC
    non_error_non_summary = [
        s for s in sections
        if "module failed" not in s["title"] and s["title"] != "Executive Summary"
    ]
    assert len(non_error_non_summary) >= 4, (
        f"Expected at least 4 successful module sections, got {len(non_error_non_summary)}: "
        f"{[s['title'] for s in non_error_non_summary]}"
    )


# ---------------------------------------------------------------------------
# Test 2: strict=True — the factor failure propagates
# ---------------------------------------------------------------------------

def test_combined_strict_reraises(monkeypatch):
    with pytest.raises(RuntimeError, match="boom"):
        _sections_with_failing_factor(monkeypatch, strict=True)


# ---------------------------------------------------------------------------
# Test 3: heatmap deduplication in combined assembly
# ---------------------------------------------------------------------------

def test_combined_heatmap_deduplication(monkeypatch):
    """Combined assembly must not contain two correlation-heatmap items with the same title."""
    # Patch factor so we don't need network
    monkeypatch.setattr(
        cr, "compute_factor_analysis",
        lambda ctx: (_ for _ in ()).throw(RuntimeError("no FF")),
    )
    sections = cr._assemble_combined_sections(_ctx(), strict=False)

    # Collect all main_content plot titles that look like heatmaps
    heatmap_titles = []
    for section in sections:
        for item in section.get("main_content", []):
            title = item.get("title", "")
            if "Correlation" in title and "Heatmap" in title:
                heatmap_titles.append(title)

    # There should be no duplicate heatmap titles
    assert len(heatmap_titles) == len(set(heatmap_titles)), (
        f"Duplicate heatmap titles found: {heatmap_titles}"
    )


# ---------------------------------------------------------------------------
# Test 4: MC params are forwarded
# ---------------------------------------------------------------------------

def test_combined_mc_params_forwarded(monkeypatch):
    """num_simulations / time_horizon / seed are forwarded to compute_monte_carlo_analysis."""
    captured = {}

    original_mc = cr.compute_monte_carlo_analysis

    def mock_mc(ctx, num_simulations=5000, time_horizon=252, initial_investment=10000, seed=42):
        captured["num_simulations"] = num_simulations
        captured["time_horizon"] = time_horizon
        captured["initial_investment"] = initial_investment
        captured["seed"] = seed
        return original_mc(ctx, num_simulations=num_simulations,
                           time_horizon=time_horizon,
                           initial_investment=initial_investment,
                           seed=seed)

    monkeypatch.setattr(cr, "compute_monte_carlo_analysis", mock_mc)
    # Patch factor to avoid network
    monkeypatch.setattr(
        cr, "compute_factor_analysis",
        lambda ctx: (_ for _ in ()).throw(RuntimeError("no FF")),
    )

    cr._assemble_combined_sections(
        _ctx(),
        strict=False,
        num_simulations=100,
        time_horizon=21,
        initial_investment=50000,
        seed=7,
    )

    assert captured["num_simulations"] == 100
    assert captured["time_horizon"] == 21
    assert captured["initial_investment"] == 50000
    assert captured["seed"] == 7
