import tempfile
from pathlib import Path

from quant_reporter.html_builder import generate_html_report


def test_text_html_main_content_renders_body():
    """
    Regression test for GitHub issue #21.

    text_html items in main_content were silently dropping item["data"].
    Verify the body content appears in the generated output.
    """
    sections = [
        {
            "title": "Executive Summary",
            "sidebar": [],
            "main_content": [
                {
                    "title": "Report Configuration",
                    "type": "text_html",
                    "data": (
                        "<p><strong>Risk-Free Rate Assumption:</strong> 5.00%</p>"
                    ),
                }
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.html"
        generate_html_report(sections, title="Test", filename=str(out))
        html = out.read_text(encoding="utf-8")

    assert "Risk-Free Rate Assumption" in html


def test_text_html_sidebar_renders_body():
    """
    text_html items in sidebar should also render item["data"].
    """
    sections = [
        {
            "title": "Info",
            "sidebar": [
                {"title": "Note", "type": "text_html", "data": "<p>Sidebar body</p>"}
            ],
            "main_content": [],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.html"
        generate_html_report(sections, title="Test", filename=str(out))
        html = out.read_text(encoding="utf-8")

    assert "Sidebar body" in html


def test_text_html_error_section_renders():
    """
    Error sections (which produce text_html items) must render
    the traceback content inside <pre> tags.
    """
    sections = [
        {
            "title": "⚠ Factor Analysis — module failed",
            "description": "RuntimeError: boom",
            "sidebar": [],
            "main_content": [
                {
                    "title": "Error",
                    "type": "text_html",
                    "data": "<pre>RuntimeError: boom</pre>",
                }
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.html"
        generate_html_report(sections, title="Test", filename=str(out))
        html = out.read_text(encoding="utf-8")

    assert "RuntimeError: boom" in html
    assert "<pre>RuntimeError: boom</pre>" in html


def test_existing_types_still_work():
    """
    Ensure plot, table_html, metrics_grid, and metrics still render
    correctly after adding text_html support.
    """
    import pandas as pd

    sections = [
        {
            "title": "Mixed Section",
            "sidebar": [
                {
                    "title": "Sidebar Metrics",
                    "type": "metrics",
                    "data": {"Alpha": 0.05, "Beta": 1.2},
                },
                {
                    "title": "Sidebar Table",
                    "type": "table_html",
                    "data": "<table><tr><td>data</td></tr></table>",
                },
            ],
            "main_content": [
                {
                    "title": "Table",
                    "type": "table_html",
                    "data": "<table class='test-table'><tr><td>cell</td></tr></table>",
                },
                {
                    "title": "Metrics Grid",
                    "type": "metrics_grid",
                    "data": {
                        "Test": {
                            "weights_dict": {"A": 0.6, "B": 0.4},
                            "metrics": {"Return": 0.10, "Vol": 0.15},
                        }
                    },
                },
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.html"
        generate_html_report(sections, title="Test", filename=str(out))
        html = out.read_text(encoding="utf-8")

    assert "test-table" in html or "class='test-table'" in html
    assert "Optimal Weights" in html
    assert "Sidebar Metrics" in html or "SidebarTable" in html
