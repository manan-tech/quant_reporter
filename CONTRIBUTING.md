# Contributing to quant_reporter

Thanks for taking the time to contribute! `quant_reporter` is a portfolio-analytics
library for systematic traders and quant researchers, and it's maintained by a single
author on a best-effort basis. Issues and PRs are genuinely welcome — this guide exists
to make your contribution land smoothly.

> **First time here?** Look for issues labelled
> [`good first issue`](https://github.com/manan-tech/quant_reporter/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
> and [`help wanted`](https://github.com/manan-tech/quant_reporter/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

## Ground rules for a quant library

This is financial software. Two rules are non-negotiable because getting them wrong
produces *confidently wrong* numbers:

1. **No look-ahead bias.** Any signal, weight schedule, or backtest decision at day *d*
   may only use data available up to day *d−1*. Volatility/momentum estimates are
   lagged (`.shift(1)`). If you add a new signal or strategy, add a property test that
   shuffles future data and asserts the present-day decision is unchanged (see
   `test/test_signals.py` for the pattern).
2. **Honest out-of-sample stats.** Selection and validation use separate splits. When a
   change touches the recommendation/validation layer, never tune against the split you
   report. Multiple-testing deflation (DSR) is there for a reason — keep it.

Everything user-facing should ship with a plain-language *why / assumption / what breaks
it*, and the "this is not investment advice" honesty layer stays loud.

## Development setup

Requires Python ≥ 3.9.

```bash
git clone https://github.com/manan-tech/quant_reporter.git
cd quant_reporter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pip install ruff
```

## Before you open a PR

Run the same checks CI runs:

```bash
ruff check src/          # lint (real bugs only — F + E9)
pytest test/ -q          # full suite; the network smoke test is auto-skipped offline
```

- **Tests must pass and coverage must stay ≥ 80%** (CI gate). New behaviour needs new
  tests. Prefer offline tests — use the `DataProvider` protocol with a CSV/fixture
  provider rather than hitting Yahoo (see `test/conftest.py`).
- **Keep it offline-testable.** All data access goes through `DataProvider`; don't add a
  hard dependency on a live network call inside core logic.
- **Match the surrounding style.** Ruff is configured for high-signal rules only; we are
  not gating line-length/whitespace, so mirror the nearby code.

## Architecture in one paragraph

Data flows through a single `DataProvider` (yfinance by default, swappable for
Bloomberg/CSV/fixtures). `ReportContext` fetches **once** and every report/analysis reads
from it ("fetch once, render many"). Layers, lowest to highest: `analytics`/`metrics`
(measurement) → `opt_core`/`advanced_optimizers` (optimization) → `backtest`/`strategy`
(cost-aware walk-forward engine) → `recommendation`/`planning` (the only *opinionated*
layer, always carrying `rationale` + `evidence`) → `*_report` (HTML rendering). Bigger
features follow a brainstorm → spec → plan → execute cycle; specs and plans live in
`docs/superpowers/`.

## Commit & PR conventions

- Branch from `main`; use conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`,
  `test:`, `refactor:`, `chore:`).
- One logical change per PR. Describe *what* and *why*, and note any change to reported
  numbers (e.g. a metric definition shift).
- The public API follows [SemVer](https://semver.org/) — flag anything breaking.

## Reporting bugs & security issues

- Bugs / features: open a [GitHub Issue](https://github.com/manan-tech/quant_reporter/issues)
  using the templates.
- Security vulnerabilities: **do not** open a public issue — see
  [SECURITY.md](SECURITY.md).

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
