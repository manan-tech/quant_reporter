# Quant Reporter

A Python library for advanced quantitative portfolio analysis, optimization, validation, and reporting.

`quant_reporter` turns a plain `{ticker: weight}` portfolio into rich, interactive, multi-page HTML reports. It is built on `pandas`, `numpy`, `scipy`, `statsmodels`, `yfinance`, and `plotly`, and covers performance & risk analytics, modern portfolio optimization, Monte Carlo forecasting, walk-forward validation, and Fama-French / Brinson attribution.

> **2.0** introduces a unified `ReportContext` architecture: every report takes the same inputs — a portfolio, a benchmark, and a training window — fetches data **once**, and renders. This is a breaking change from 1.x (see [Migrating from 1.x](#migrating-from-1x)).

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

The package is descriptive analytics on daily historical data — a decision-support and communication tool, not a cost-aware execution backtester. Monte Carlo assumes Geometric Brownian Motion (thin tails — it understates crash risk), and it depends on live `yfinance` data.

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

## The report generators

All six accept the common signature above; the options below are all keyword-only and optional.

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

```bash
pip install -e ".[test]"
pytest            # offline unit tests; the report smoke test is skipped without network
```

---

## License

MIT — see [LICENSE](LICENSE).
