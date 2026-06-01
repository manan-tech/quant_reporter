# Example report gallery

Recognizable portfolios analyzed end-to-end with `quant_reporter` — they double
as a demo, documentation, and discovery surface ("All-Weather portfolio
analysis", "60/40 backtest", ...).

| Portfolio | Tickers |
|---|---|
| Classic 60/40 | SPY / AGG |
| All-Weather (Ray Dalio) | VTI / TLT / IEI / GLD / DBC |
| Magnificent 7 | AAPL MSFT GOOGL AMZN NVDA META TSLA |
| Bogleheads 3-Fund | VTI / VXUS / BND |

## Regenerate the gallery

Requires network (live Yahoo Finance data):

```bash
pip install -e ".[test]"
python examples/gallery/generate_gallery.py
```

This writes one interactive report per portfolio plus an `index.html` landing
page into the repo-root `site/` folder.

> `site/` is **git-ignored** and intentionally NOT committed to `main` — the
> combined reports are ~5 MB each, and committing them would bloat `main`'s
> history on every regeneration.

## Publish to GitHub Pages (via the `gh-pages` branch)

The gallery is served from a dedicated `gh-pages` branch so `main` stays lean.

1. Regenerate: `python examples/gallery/generate_gallery.py`
2. Push `site/`'s contents to the `gh-pages` branch. Easiest:
   `npx gh-pages -d site` (or push the files to the root of an orphan
   `gh-pages` branch by hand / via a worktree).
3. One-time: **Settings → Pages → Build and deployment → Deploy from a branch →
   `gh-pages` / (root)**.

The live gallery will be at `https://manan-tech.github.io/quant_reporter/`.

## Zero-setup notebook

[`quant_reporter_gallery.ipynb`](quant_reporter_gallery.ipynb) runs the whole
thing in Colab — pick a portfolio, Run all, see the report inline. Pre-release it
installs from GitHub `main`; switch to `pip install quant-reporter` after the
PyPI release.
