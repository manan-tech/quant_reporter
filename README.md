# Quant Reporter

[![CI](https://github.com/manan-tech/quant_reporter/actions/workflows/ci.yml/badge.svg)](https://github.com/manan-tech/quant_reporter/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/quant-reporter.svg)](https://pypi.org/project/quant-reporter/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/quant-reporter?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/quant-reporter)
[![Python versions](https://img.shields.io/pypi/pyversions/quant-reporter.svg)](https://pypi.org/project/quant-reporter/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Portfolio analytics, MPT optimization, cost-aware backtesting, and walk-forward validation for systematic traders and quant researchers.

> **Not financial advice — see [Disclaimer](#disclaimer).** Pulls market data from Yahoo Finance via `yfinance` (network access — see [Data sources & offline use](#data-sources--offline-use)).

`quant_reporter` turns a plain `{ticker: weight}` portfolio into rich, interactive, multi-page HTML reports. It is built on `pandas`, `numpy`, `scipy`, `statsmodels`, `yfinance`, and `plotly`, and covers performance & risk analytics, modern portfolio optimization, Monte Carlo forecasting, walk-forward validation, and Fama-French / Brinson attribution.

### 30-second start

```bash
pip install quant-reporter
```

```python
import quant_reporter as qr

# Build a full multi-page HTML report for a portfolio vs. a benchmark.
qr.create_portfolio_report(
    {"AAPL": 0.6, "MSFT": 0.4}, "SPY",
    "2020-01-01", "2023-12-31",
    filename="report.html",
)
# → open report.html
```

That's the whole loop: a `{ticker: weight}` dict in, an interactive report out. Everything below (optimization, backtesting, Black-Litterman, custom data sources) is optional depth.

**See it without installing:** browse the [example report gallery](https://manan-tech.github.io/quant_reporter/) (60/40, All-Weather, Magnificent 7, Bogleheads 3-fund) or run the [zero-setup Colab notebook](https://colab.research.google.com/github/manan-tech/quant_reporter/blob/main/examples/gallery/quant_reporter_gallery.ipynb).

> **Version note:** the last release on PyPI was `1.1.1`. `2.0.0` was an internal milestone and was **never published**, so `2.1.0` is the first 2.x release you can `pip install`. The 1.x → 2.x change is breaking (see [Migrating from 1.x](#migrating-from-1x)).
>
> **2.0** introduces a unified `ReportContext` architecture: every report takes the same inputs — a portfolio, a benchmark, and a training window — fetches data **once**, and renders. This is a breaking change from 1.x (see [Migrating from 1.x](#migrating-from-1x)).
>
> **2.1** adds a primitives-first **strategy → backtest → report** loop — a cost-aware, walk-forward backtest engine with honest out-of-sample statistics (PSR/DSR) and an interactive backtest report — plus an opt-in **recommendation layer** (target weights, rebalance trade lists, risk-limit alerts, strategy verdicts, each carrying its rationale & evidence). All additive; the 2.0 API is unchanged. See [Strategy backtesting & recommendations](#strategy-backtesting--recommendations-21).
>
> **2.2** adds a **decision-support planning layer**: a CFA-grounded investor `Profile` (risk-tolerance presets + return objective + TTLU constraints) that constrains the optimizer **and** sets the risk-alert thresholds via `recommend(profile=)`, plus **walk-forward validation of recommendations** (`recommend(validate=True)` → out-of-sample vs in-sample Sharpe, degradation, and a holds-up / fragile / inconclusive verdict). All additive; the 2.1 API is unchanged.

---

## Who it's for

| Question a trader/PM asks | What the package answers |
|---|---|
| *Is my portfolio good, risk-adjusted?* | Portfolio report: Sharpe, Sortino, Calmar, max drawdown, VaR/CVaR, alpha/beta vs a benchmark |
| *How should I weight these?* | Optimization report: efficient frontier, min-vol, max-Sharpe, sector caps/mins, Risk Parity, HRP, Min-Correlation, Max-Diversification |
| *Am I just overfitting the past?* | Validation report: train/test out-of-sample split, overfitting score, walk-forward windows |
| *What could happen next?* | Monte Carlo report: GBM path simulation, success probabilities, time-to-target, day-1 stress shocks |
| *Was it skill or just market beta?* | Factor report: Fama-French regression (alpha vs factor exposure) + Brinson allocation/selection attribution |
| *How do I fold in my own views?* | Black-Litterman: blend market equilibrium with absolute & relative views |
| *Which strategy actually holds up out-of-sample?* | **Strategy backtesting (2.1)**: cost-aware walk-forward `backtest`/`backtest_many`, honest PSR/DSR out-of-sample stats, interactive backtest report |
| *What should I do about it?* (opt-in) | **Recommendation layer (2.1)**: recommended target weights, a rebalance trade list, risk-limit alerts, and a strategy verdict — each with its rationale & evidence |
| *Does this suit my goals, and does it hold up out-of-sample?* (opt-in) | **Decision-support layer (2.2)**: a CFA-grounded investor `Profile` that constrains the optimizer and sets alert thresholds, plus walk-forward validation of the recommendation (in-sample vs OOS Sharpe + a holds-up / fragile verdict) |
| *Is this backtest result real, or did I overfit picking it?* | **Backtest honesty (2.3)**: Probability of Backtest Overfitting (PBO via CSCV), Minimum Track Record / Backtest Length, and a one-call overfitting verdict that deflates the Sharpe for the number of trials you ran |

The 2.0 report generators are descriptive analytics on daily historical data — a decision-support and communication tool. **2.1 adds a cost-aware, walk-forward backtest engine, a composable strategy layer, and an opt-in recommendation layer** (see [Strategy backtesting & recommendations](#strategy-backtesting--recommendations-21)). Monte Carlo assumes Geometric Brownian Motion (thin tails — it understates crash risk), and reports depend on live `yfinance` data.

---

## Installation

```bash
pip install quant-reporter          # from PyPI
```

For local development (editable install + test tooling):

```bash
git clone https://github.com/manan-tech/quant_reporter.git
cd quant_reporter
pip install -e ".[test]"
```

Requires Python ≥ 3.9.

---

## Quick start

Every report shares the same call shape:

```python
create_<kind>_report(portfolio_dict, benchmark_ticker, train_start, train_end, filename=..., **options)
```

- `portfolio_dict` — `{ticker: weight}` (weights need not sum to 1; they are used as given).
- `benchmark_ticker` — e.g. `"SPY"`.
- `train_start` / `train_end` — the in-sample window used to fit optimizers and metrics. The **out-of-sample** test window is derived automatically (`train_end + 1 day` … yesterday) and used by the validation report.

```python
import quant_reporter as qr

portfolio = {"AAPL": 0.3, "MSFT": 0.3, "XOM": 0.2, "GLD": 0.2}

# The flagship: one HTML covering all five analyses
qr.create_combined_report(
    portfolio_dict=portfolio,
    benchmark_ticker="SPY",
    train_start="2018-01-01",
    train_end="2023-12-31",
    filename="Combined_Report.html",
    sector_map={"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy", "GLD": "Commodities"},
    sector_caps={"Tech": 0.5, "Energy": 0.3, "Commodities": 0.3},
    risk_free_rate="auto",   # fetches the live 13-week T-bill rate; or pass a float like 0.045
)
```

---

## Data sources & offline use

By default `quant_reporter` fetches prices and the risk-free rate from Yahoo Finance via [`yfinance`](https://github.com/ranaroussi/yfinance). **This means a normal report run makes live network calls** — relevant for corporate, air-gapped, or compliance-sensitive environments.

All data access goes through a single `DataProvider` protocol, so you can swap Yahoo for Bloomberg, Refinitiv, a local CSV, or a test fixture without touching any report code. A provider only needs two methods:

```python
import pandas as pd
import quant_reporter as qr

class CSVProvider:
    """Reads prices from a local CSV — no network, fully reproducible."""
    def __init__(self, csv_path):
        self._prices = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    def get_prices(self, tickers, start, end):
        return self._prices.loc[start:end, list(tickers)]
    def get_risk_free_rate(self):
        return 0.045

provider = CSVProvider("prices.csv")

# Per call:
qr.create_combined_report(portfolio, "SPY", "2018-01-01", "2023-12-31",
                          filename="report.html", data_provider=provider)

# …or globally, for the whole session:
qr.set_default_provider(provider)
```

Already have a price DataFrame in memory? `qr.build_context_from_prices(prices, ...)` skips fetching entirely. Pass a numeric `risk_free_rate=` (instead of `"auto"`) to stay 100% offline. (Black-Litterman market caps and per-asset fundamentals are opt-in extras that still use yfinance unless your provider also implements `get_market_caps`.)

---

## The report generators

All six accept the common signature above; the options below are all keyword-only and optional. Every generator also accepts `data_provider=` (see [Data sources & offline use](#data-sources--offline-use)).

| Function | Focus |
|---|---|
| `create_portfolio_report` | Risk/return dashboard vs benchmark (also aliased as `create_full_report`) |
| `create_optimization_report` | Optimizers, sector constraints, efficient frontier, Black-Litterman |
| `create_validation_report` | In-sample vs out-of-sample, overfitting score, walk-forward |
| `create_monte_carlo_report` | GBM forecasting, success probabilities, stress scenarios |
| `create_factor_report` | Fama-French regression + Brinson attribution |
| `create_combined_report` | All of the above in a single document |

```python
import quant_reporter as qr

portfolio = {"AAPL": 0.4, "MSFT": 0.35, "GLD": 0.25}
common = dict(benchmark_ticker="SPY", train_start="2018-01-01", train_end="2023-12-31")

qr.create_portfolio_report(portfolio_dict=portfolio, filename="01_Portfolio.html", **common)
qr.create_optimization_report(portfolio_dict=portfolio, filename="02_Optimization.html", **common)
qr.create_monte_carlo_report(portfolio_dict=portfolio, filename="03_MonteCarlo.html",
                             num_simulations=5000, **common)
qr.create_validation_report(portfolio_dict=portfolio, filename="04_Validation.html", **common)
qr.create_factor_report(portfolio_dict=portfolio, filename="05_Factor.html",
                        sector_map={"AAPL": "Tech", "MSFT": "Tech", "GLD": "Commodities"},
                        **common)
```

### Common options (keyword arguments)

| Option | Type | Meaning |
|---|---|---|
| `risk_free_rate` | `float` or `"auto"` | Annual risk-free rate. `"auto"` fetches the live 13-week T-bill (`^IRX`). Default `"auto"`. |
| `display_names` | `dict` | Friendly labels, e.g. `{"AAPL": "Apple"}`. |
| `sector_map` | `dict` | `{ticker: sector}` — enables sector constraints, sector charts, and Brinson attribution. |
| `sector_caps` / `sector_mins` | `dict` | `{sector: max_weight}` / `{sector: min_weight}` for the optimizer. |
| `bl_views` | `dict` | Absolute Black-Litterman views, e.g. `{"AAPL": 0.15}` ("AAPL returns 15% p.a."). |
| `bl_view_confidences` | `dict` | Confidence (0–1) per absolute view. |
| `bl_relative_views` | `list[tuple]` | Relative views as `(outperformer, underperformer, spread)`, e.g. `[("NVDA", "AAPL", 0.03)]`. |
| `bl_relative_view_confidences` | `list[float]` | Confidence (0–1) per relative view. |
| `denoise_cov` | `bool` | Eigenvalue-clip the covariance matrix before optimizing. |

### Black-Litterman example

```python
qr.create_optimization_report(
    portfolio_dict={"AAPL": 0.25, "NVDA": 0.25, "JPM": 0.25, "XOM": 0.25},
    benchmark_ticker="SPY",
    train_start="2019-01-01",
    train_end="2023-12-31",
    filename="BL_Optimization.html",
    bl_views={"NVDA": 0.20},                       # absolute: NVDA returns 20% p.a.
    bl_view_confidences={"NVDA": 0.6},
    bl_relative_views=[("AAPL", "XOM", 0.05)],     # AAPL outperforms XOM by 5%
    bl_relative_view_confidences=[0.5],
)
```

---

## Strategy backtesting & recommendations (2.1)

2.1 adds a first-class **strategy → backtest → report** loop and an opt-in
**recommendation layer**, both additive to the 2.0 API.

### Backtest a strategy

A strategy is **any callable** `(prices, **params) -> weights` — returning a `dict` for a
static allocation or a dated `DataFrame` schedule — or a prebuilt from `qr.REGISTRY`, or a
`qr.Strategy` wrapper. `qr.backtest` runs it through the cost-aware, walk-forward engine
(reusing the tested `simulate_strategy`) and returns a rich `BacktestResult`.

```python
import quant_reporter as qr

prices = qr.get_data(["SPY", "TLT", "GLD"], "2015-01-01", "2024-12-31")

res = qr.backtest(qr.risk_parity, prices, benchmark="SPY",
                  rebalance="M", cost_model=qr.transaction_cost_model)
res.metrics      # dict: CAGR, Sharpe, Sortino, Calmar, Max Drawdown, ...
res.oos_stats    # {'psr': ..., 'dsr': ...} — honest out-of-sample stats
res.report("Backtest.html", open_browser=True)   # interactive HTML report
```

Prebuilt strategies (keys of `qr.REGISTRY`): `equal_weight`, `inverse_vol`, `min_variance`,
`risk_parity`, `max_sharpe`, `trend_following`, `cross_sectional_momentum` — plus the
higher-order `qr.vol_target_overlay(base_fn, target_vol=...)`. Schedule-producing strategies
are look-ahead-safe (signals lagged, each row decided on data up to *d−1*).

Compare several strategies (deflated for multiple testing) in one report:

```python
results = qr.backtest_many(
    {"EW": qr.equal_weight, "RP": qr.risk_parity, "Trend": qr.trend_following},
    prices, benchmark="SPY", cost_model=qr.transaction_cost_model)
qr.create_backtest_report(results, path="Compare.html")   # adds an OOS comparison panel
```

A consolidated **metrics** library (`qr.summary_metrics`, `qr.sharpe`, `qr.sortino`,
`qr.calmar`, `qr.max_drawdown`, `qr.value_at_risk`, …) and minimize-ready **objectives**
(`qr.neg_sharpe`, `qr.variance`, `qr.cvar_objective`, …) back the report and are usable on
their own.

### Recommendations (opt-in — the only opinionated layer)

Everything above is opinion-free with explicit parameters. The recommendation layer is where
opinions live — vol target, drawdown limit, concentration caps, the selection metric — all
overridable defaults. Each recommendation carries a human-readable `rationale` and a
machine-readable `evidence` dict. It **consumes** the backtest/analytics primitives; it never
re-optimizes or re-backtests.

```python
rec = qr.recommend(
    prices,                                  # asset prices (exclude any benchmark column)
    current_weights={"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
    results=results,                         # from backtest_many — drives the verdict
    vol_target=0.10, max_drawdown_limit=0.20, max_weight=0.40,
)
rec.target_weights   # RecommendedWeights — optimal target + rationale/evidence
rec.trades           # RebalancePlan — buy/sell deltas, turnover, est. cost, no-trade band
rec.alerts           # list[RiskAlert] — vol / drawdown / concentration / sector / factor breaches
rec.verdict          # StrategyVerdict — which strategy wins on deflated Sharpe, with evidence
print(rec.to_text())                  # plain-text digest
rec.to_html("Recommendation.html")    # transparent HTML section
```

The four pieces are also standalone — `qr.recommend_weights`, `qr.rebalance_trades`,
`qr.risk_alerts`, `qr.compare_verdict` — and a recommendation can be embedded directly into a
backtest report: `res.report("Backtest.html", recommendation=rec)`.

### Backtest honesty — is this result real, or overfit? (2.3)

When you try many strategy configurations and keep the best, its backtest is biased upward.
This layer measures that bias directly. Pass the return streams of the configurations you
tried (e.g. from `backtest_many` over a parameter set) and get a one-call verdict:

```python
report = qr.assess_overfitting(returns_matrix=config_returns)  # (T periods × N configs)
report.verdict   # 'robust' | 'caution' | 'likely_overfit' | 'inconclusive'
report.pbo       # Probability of Backtest Overfitting (CSCV) — how often the in-sample
                 # winner lands at/below the out-of-sample median
print(report.to_text())
sections = [qr.overfitting_section(report)]   # embeddable HTML report section
```

The pieces are also standalone:

```python
qr.probability_of_backtest_overfitting(config_returns, n_splits=16)  # PBOResult
qr.min_track_record_length(returns, prob=0.95)   # observations needed for a significant Sharpe
qr.min_backtest_length(n_trials=100, sr_target_annual=1.0)  # years before N trials fake a Sharpe
```

The thresholds behind the verdict are documented heuristics, not decision rules — read them
alongside the strategy logic. Refs: Bailey & López de Prado, *Deflated Sharpe Ratio* and
*The Probability of Backtest Overfitting*.

---

## Library (advanced) usage

Beyond the one-call reports, the building blocks are importable for notebooks and custom scripts.

### Build a context once, reuse it

```python
from quant_reporter import build_context
from quant_reporter.optimization_report import compute_optimization_analysis

ctx = build_context({"AAPL": 0.5, "MSFT": 0.5}, "SPY", "2018-01-01", "2023-12-31")
# ctx carries price_data_full/train/test, mean_returns, cov_matrix, log_returns, ...
sections = compute_optimization_analysis(ctx)
```

### Optimizers

```python
from quant_reporter import (
    get_optimization_inputs, optimize_risk_parity, optimize_hrp,
    optimize_min_correlation, optimize_max_diversification,
)

mean_returns, cov_matrix, log_returns = get_optimization_inputs(price_df)
weights_rp = optimize_risk_parity(cov_matrix)
weights_hrp, _ = optimize_hrp(cov_matrix)
```

### Fama-French factor analysis

```python
import quant_reporter as qr

factors = qr.fetch_fama_french_factors(dataset="F-F_Research_Data_Factors_daily",
                                       start_date="2020-01-01")
res = qr.run_factor_regression(portfolio_returns, factors)   # portfolio_returns: a pd.Series
print(f"Alpha (annualized): {res['alpha']:.2%}")
print(f"Market beta: {res['betas']['Mkt-RF']:.3f}  R^2: {res['r_squared']:.3f}")

attribution = qr.compute_factor_attribution(portfolio_returns, factors,
                                            res["betas"], res["alpha"])
```

### Brinson performance attribution

```python
import quant_reporter as qr

# asset_returns: a DataFrame of per-asset returns (DatetimeIndex, one column per ticker)
attribution = qr.compute_brinson_attribution(
    portfolio_weights={"AAPL": 0.4, "XOM": 0.3, "JPM": 0.3},
    benchmark_weights={"AAPL": 0.3, "XOM": 0.4, "JPM": 0.2, "GS": 0.1},
    asset_returns=asset_returns,
    sector_map={"AAPL": "Tech", "XOM": "Energy", "JPM": "Finance", "GS": "Finance"},
)
print(attribution.loc["Total"])   # Allocation_Effect, Selection_Effect, Interaction_Effect, ...
```

### Black-Litterman (low level)

```python
from quant_reporter import calculate_black_litterman_posterior

posterior_returns, posterior_cov = calculate_black_litterman_posterior(
    hist_mean_returns, cov_matrix,
    view_dict={"AAPL": 0.10},
    relative_views=[("NVDA", "AAPL", 0.03)],   # tuples of (outperformer, underperformer, spread)
)
```

### Monte Carlo (low level)

```python
from quant_reporter import simulate_portfolio_paths, calculate_success_probabilities

sim = simulate_portfolio_paths(weights, mean_returns, cov_matrix,
                               num_simulations=5000, time_horizon=252)
```

---

## Migrating from 1.x

2.0 unifies every report around `build_context`. The reports no longer take pre-computed
returns/weights — they take the portfolio, benchmark, and training window and fetch data
themselves:

```python
# 1.x
qr.create_factor_report(portfolio_returns=returns, portfolio_name="Mine", filename="f.html")

# 2.0
qr.create_factor_report(portfolio_dict={"AAPL": 0.5, "MSFT": 0.5},
                        benchmark_ticker="SPY",
                        train_start="2020-01-01", train_end="2023-12-31",
                        filename="f.html")
```

Other changes: `compute_brinson_attribution` now takes a single `asset_returns` matrix (plus
portfolio/benchmark weight dicts and a `sector_map`) instead of separate return series;
`create_full_report` is retained as an alias for `create_portfolio_report`.

---

## Examples & testing

- `examples/generate_all_5_reports.py` — generates all five individual reports for a sample portfolio.
- `examples/example_combined_report.py` — the combined flagship report.
- `examples/example_black_litterman.py` — Black-Litterman views.
- `examples/example_strategy_report.py` — (2.1) backtest several strategies → interactive backtest report (offline).
- `examples/example_recommendation.py` — (2.1) opt-in recommendation bundle + transparent report, embedded in a backtest report (offline).

```bash
pip install -e ".[test]"
pytest            # offline unit tests; the report smoke test is skipped without network
```

---

## Support & status

`quant_reporter` is maintained by a single author on a **best-effort** basis. Bug
reports and PRs are welcome via [GitHub Issues](https://github.com/manan-tech/quant_reporter/issues);
responses are not guaranteed on any timeline. For security reports, see
[SECURITY.md](SECURITY.md). The public API follows [SemVer](https://semver.org/) —
breaking changes bump the major version.

**Want to contribute?** See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the two
non-negotiable quant rules (no look-ahead bias, honest out-of-sample stats), and the test
workflow. New contributors: start with issues labelled
[`good first issue`](https://github.com/manan-tech/quant_reporter/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or [`help wanted`](https://github.com/manan-tech/quant_reporter/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).

---

## License

MIT — see [LICENSE](LICENSE).

---

## Disclaimer

**Not financial advice.** `quant_reporter` is provided for informational, research, and educational purposes only. It does not constitute financial, investment, tax, or legal advice, nor a recommendation or solicitation to buy or sell any security or financial instrument. Outputs may be inaccurate or incomplete, and all models carry assumptions and limitations (for example, the Monte Carlo engine assumes Geometric Brownian Motion, which has thin tails and **understates crash risk**). You are solely responsible for any decisions made using this software. Use at your own risk; the authors accept no liability for any losses.

**Market-data source / Yahoo Finance terms.** `quant_reporter` uses the third-party [`yfinance`](https://github.com/ranaroussi/yfinance) library (Apache-2.0) to retrieve market data from Yahoo Finance. This project is **not** affiliated with, endorsed by, or vetted by Yahoo, Inc. The data is intended for personal use and may be subject to [Yahoo's Terms of Service](https://policies.yahoo.com/us/en/yahoo/terms/index.htm). You are responsible for reviewing and complying with those terms before using any retrieved data, particularly for commercial purposes.
