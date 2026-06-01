"""
Generate the quant_reporter example-report gallery.

Runs the flagship **combined report** on a set of well-known portfolios and
writes standalone interactive HTML into ``site/`` (repo root), plus a
``site/index.html`` landing page. Publish ``site/`` via GitHub Pages — see
``.github/workflows/pages.yml``. (A dedicated ``site/`` dir keeps the published
gallery separate from the internal planning docs under ``docs/``.)

Usage
-----
    python examples/gallery/generate_gallery.py        # live Yahoo Finance data

The portfolios are recognizable on purpose — they double as discovery surface
("All-Weather portfolio analysis", "60/40 backtest", "Magnificent 7 risk", ...).
"""
from __future__ import annotations

import os
import traceback

import quant_reporter as qr

# Long in-sample window; the out-of-sample test window is derived automatically
# (train_end + 1 day .. yesterday) and used by the validation section.
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
BENCHMARK = "SPY"

# For ETF baskets the "sector" is really an asset class — the sector_map just
# groups holdings for the allocation pie and sector-aware attribution. For
# Magnificent 7 these are true GICS sectors.
PORTFOLIOS = [
    {
        "slug": "60-40",
        "name": "Classic 60/40",
        "blurb": "The textbook balanced portfolio: 60% US equities, 40% US bonds.",
        "weights": {"SPY": 0.60, "AGG": 0.40},
        "sector_map": {"SPY": "US Equity", "AGG": "US Bonds"},
    },
    {
        "slug": "all-weather",
        "name": "All-Weather (Ray Dalio)",
        "blurb": "Dalio's risk-balanced mix across stocks, long & intermediate bonds, gold, and commodities.",
        "weights": {"VTI": 0.30, "TLT": 0.40, "IEI": 0.15, "GLD": 0.075, "DBC": 0.075},
        "sector_map": {
            "VTI": "Equity", "TLT": "Long-Term Bonds", "IEI": "Intermediate Bonds",
            "GLD": "Gold", "DBC": "Commodities",
        },
    },
    {
        "slug": "magnificent-7",
        "name": "Magnificent 7",
        "blurb": "The mega-cap tech basket — a vivid demo of concentration risk and factor exposure.",
        "weights": {t: 1 / 7 for t in ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")},
        "sector_map": {
            "AAPL": "Information Technology", "MSFT": "Information Technology",
            "NVDA": "Information Technology", "GOOGL": "Communication Services",
            "META": "Communication Services", "AMZN": "Consumer Discretionary",
            "TSLA": "Consumer Discretionary",
        },
    },
    {
        "slug": "bogleheads-3-fund",
        "name": "Bogleheads 3-Fund",
        "blurb": "The classic low-cost index trio: total US, total international, total bond.",
        "weights": {"VTI": 0.60, "VXUS": 0.20, "BND": 0.20},
        "sector_map": {"VTI": "US Equity", "VXUS": "Intl Equity", "BND": "US Bonds"},
    },
]

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.normpath(os.path.join(HERE, "..", "..", "site"))


def build_gallery(out_dir: str = SITE, portfolios=PORTFOLIOS, provider=None):
    """Build every portfolio's combined report + the index page into *out_dir*."""
    os.makedirs(out_dir, exist_ok=True)
    built = []
    for p in portfolios:
        path = os.path.join(out_dir, f"{p['slug']}.html")
        print(f"[gallery] building {p['name']} -> {path}")
        try:
            qr.create_combined_report(
                p["weights"], BENCHMARK, TRAIN_START, TRAIN_END,
                filename=path, sector_map=p.get("sector_map"), data_provider=provider,
            )
            built.append({**p, "ok": True})
        except Exception as exc:  # keep building the rest of the gallery
            print(f"[gallery] FAILED {p['name']}: {exc}")
            traceback.print_exc()
            built.append({**p, "ok": False, "error": str(exc)})
    _write_index(out_dir, built)
    ok = sum(1 for b in built if b["ok"])
    print(f"[gallery] done: {ok}/{len(built)} reports built -> {os.path.join(out_dir, 'index.html')}")
    return built


def _write_index(out_dir: str, built: list[dict]) -> None:
    cards = []
    for b in built:
        tickers = ", ".join(b["weights"].keys())
        if b["ok"]:
            open_tag, close_tag, cta = f'<a class="card" href="{b["slug"]}.html">', "</a>", '<span class="cta">View report &rarr;</span>'
        else:
            open_tag, close_tag, cta = '<div class="card disabled">', "</div>", '<span class="cta err">build failed</span>'
        cards.append(
            f'{open_tag}<h2>{b["name"]}</h2>'
            f'<p class="blurb">{b["blurb"]}</p>'
            f'<p class="tickers">{tickers}</p>{cta}{close_tag}'
        )
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>quant_reporter — example report gallery</title>
<style>
 body {{ font-family:-apple-system,system-ui,sans-serif; background:#0f172a; color:#e2e8f0; margin:0; }}
 header {{ padding:3rem 1.5rem 2rem; max-width:900px; margin:0 auto; }}
 header h1 {{ font-size:1.8rem; margin:0 0 .5rem; }}
 header p {{ color:#94a3b8; max-width:62ch; }}
 header code {{ background:#1e293b; padding:.15rem .45rem; border-radius:4px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:1rem; max-width:900px; margin:0 auto; padding:0 1.5rem 3rem; }}
 .card {{ display:block; background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.3rem; text-decoration:none; color:inherit; transition:.15s; }}
 .card:hover {{ border-color:#38bdf8; transform:translateY(-2px); }}
 .card.disabled {{ opacity:.5; }}
 .card h2 {{ font-size:1.1rem; margin:0 0 .4rem; }}
 .blurb {{ color:#94a3b8; font-size:.9rem; margin:0 0 .7rem; }}
 .tickers {{ font-family:monospace; font-size:.78rem; color:#64748b; margin:0 0 1rem; }}
 .cta {{ color:#38bdf8; font-weight:600; font-size:.9rem; }}
 .cta.err {{ color:#f87171; }}
 footer {{ text-align:center; color:#64748b; font-size:.8rem; padding:2rem; }}
 footer a {{ color:#38bdf8; }}
</style></head>
<body>
<header>
  <h1>quant_reporter — example reports</h1>
  <p>Interactive, multi-page portfolio analytics generated with a single function call.
  Install with <code>pip install quant-reporter</code> and reproduce any of these in a few lines.</p>
</header>
<div class="grid">{"".join(cards)}</div>
<footer>Generated with quant_reporter &middot; <a href="https://github.com/manan-tech/quant_reporter">GitHub</a></footer>
</body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)


if __name__ == "__main__":
    build_gallery()
