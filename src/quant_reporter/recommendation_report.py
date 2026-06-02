# src/quant_reporter/recommendation_report.py
"""Recommendation report rendering (SP4).

Renders a Recommendation as a transparent HTML section via html_builder.
Lazy-imported by Recommendation.to_html(). Numbers come only from the
recommendation objects; nothing is recomputed.
"""
import os

import pandas as pd

from .html_builder import generate_html_report

_SEV_COLOR = {"breach": "#c0392b", "warning": "#e67e22", "ok": "#27ae60"}


def _weights_table_html(rw):
    rows = "".join(f"<tr><td>{tk}</td><td>{wt:.2%}</td></tr>"
                   for tk, wt in rw.weights.items())
    return (f'<table class="metrics-table"><tr><th>Asset</th><th>Weight</th></tr>'
            f'{rows}</table>')


def _trades_table_html(plan):
    if plan is None or not plan.orders:
        return "<p>No trades.</p>"
    rows = "".join(
        f"<tr><td>{o.side.upper()}</td><td>{o.ticker}</td><td>{o.current_weight:.2%}</td>"
        f"<td>{o.target_weight:.2%}</td><td>{o.delta:+.2%}</td></tr>" for o in plan.orders)
    return ('<table class="metrics-table"><tr><th>Side</th><th>Asset</th><th>Current</th>'
            f'<th>Target</th><th>Delta</th></tr>{rows}</table>')


def _alerts_html(alerts):
    if not alerts:
        return "<p>No risk alerts &mdash; all checks within limits.</p>"
    items = "".join(
        f'<li style="color:{_SEV_COLOR.get(a.severity, "#333")}">'
        f'<b>[{a.severity.upper()}] {a.kind}</b>: {a.rationale}</li>' for a in alerts)
    return f"<ul>{items}</ul>"


def _verdict_table_html(verdict):
    if verdict is None or not verdict.ranking:
        return "<p>No strategy comparison.</p>"
    df = pd.DataFrame(verdict.ranking).set_index("name")
    return df.to_html(classes="metrics-table", border=0, float_format=lambda x: f"{x:.3f}")


_SUIT_COLOR = {True: "#27ae60", False: "#c0392b"}


def _suitability_html(suitability):
    if suitability is None:
        return "<p>No investor profile provided &mdash; suitability not assessed.</p>"
    overall_color = _SUIT_COLOR[suitability.suitable]
    overall_label = "SUITABLE" if suitability.suitable else "NOT SUITABLE"
    rows = "".join(
        f'<tr>'
        f'<td style="color:{_SUIT_COLOR[c.passed]};font-weight:bold">{"PASS" if c.passed else "FAIL"}</td>'
        f'<td>{c.name}</td>'
        f'<td>{c.detail}</td>'
        f'</tr>'
        for c in suitability.checks
    )
    return (
        f'<p style="color:{overall_color};font-weight:bold">{overall_label}: {suitability.rationale}</p>'
        f'<table class="metrics-table">'
        f'<tr><th>Result</th><th>Check</th><th>Detail</th></tr>'
        f'{rows}'
        f'</table>'
    )


_VERDICT_COLOR = {"holds up": "#27ae60", "fragile (overfit)": "#c0392b",
                  "inconclusive": "#7f8c8d"}


def _validation_html(validation):
    if validation is None:
        return "<p>Walk-forward validation not run (pass <code>validate=True</code>).</p>"
    color = _VERDICT_COLOR.get(validation.verdict, "#7f8c8d")
    baseline = ("" if validation.baseline_oos_sharpe is None
                else f" | current portfolio OOS Sharpe {validation.baseline_oos_sharpe:.2f}")
    row_html = []
    for w in validation.per_window:
        cur = w.get("current_sharpe")
        cur_txt = "&mdash;" if cur is None else f"{cur:.2f}"
        row_html.append(
            f'<tr><td>{w["period"]}</td>'
            f'<td>{w["recommended_sharpe"]:.2f}</td>'
            f'<td>{cur_txt}</td></tr>'
        )
    rows = "".join(row_html)
    return (
        f'<p style="color:{color};font-weight:bold">{validation.verdict.upper()}: '
        f'{validation.rationale}</p>'
        f'<p>In-sample Sharpe {validation.in_sample_sharpe:.2f} | '
        f'OOS Sharpe {validation.oos_sharpe:.2f} | {validation.n_windows} windows{baseline}</p>'
        f'<table class="metrics-table">'
        f'<tr><th>Test Period</th><th>Recommended Sharpe</th><th>Current Sharpe</th></tr>'
        f'{rows}'
        f'</table>'
    )


def build_recommendation_section(rec):
    return {
        "title": "Recommendations",
        "main_content": [
            {"type": "table_html", "title": "Recommended Target Weights",
             "data": _weights_table_html(rec.target_weights),
             "description": rec.target_weights.rationale},
            {"type": "table_html", "title": "Rebalance Plan",
             "data": _trades_table_html(rec.trades),
             "description": (rec.trades.rationale if rec.trades is not None
                             else "No current weights provided.")},
            {"type": "table_html", "title": "Risk Alerts",
             "data": _alerts_html(rec.alerts)},
            {"type": "table_html", "title": "Strategy Verdict",
             "data": _verdict_table_html(rec.verdict),
             "description": (rec.verdict.rationale if rec.verdict is not None
                             else "No strategies compared.")},
            {"type": "table_html", "title": "Suitability Assessment",
             "data": _suitability_html(rec.suitability)},
            {"type": "table_html", "title": "Walk-Forward Validation",
             "data": _validation_html(rec.validation),
             "description": (rec.validation.rationale if rec.validation is not None
                             else "Pass validate=True to run out-of-sample walk-forward.")},
        ],
    }


def create_recommendation_report(rec, path=None, open_browser=False):
    path = path or "recommendation_report.html"
    generate_html_report([build_recommendation_section(rec)],
                          title="Recommendation Report", filename=path)
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(path)}")
    return path
