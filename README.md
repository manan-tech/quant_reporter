# Quant Reporter

A Python library for advanced quantitative portfolio analysis, optimization, and validation.

`quant_reporter` moves beyond simple metrics by providing a suite of tools to analyze, optimize, and stress-test investment portfolios. It is built on `pandas`, `yfinance`, and `plotly` to create rich, interactive, and cross-browser compatible HTML reports.

This package is designed to be used in two ways:
1.  **As a Report Generator:** Use one of the two main functions (`create_full_report` or `create_combined_report`) to instantly generate a comprehensive, multi-page HTML report.
2.  **As a Core Library:** Import individual functions (e.g., `get_optimization_inputs`, `plot_efficient_frontier`) to build your own custom analysis scripts or notebooks.

## Key Features

* **Simple & Portfolio Analysis:** Analyze a single ticker or a complex, weighted portfolio.
* **Rich Metrics:** Calculates 15+ key performance and risk metrics, including **Sharpe, Sortino, Calmar, VaR (Value at Risk), CVaR (Conditional VaR),** and **Alpha/Beta**.
* **Modern Portfolio Theory (MPT):** Generates optimized portfolios based on:
    * Minimum Volatility
    * Maximum Sharpe Ratio (Unconstrained)
    * Maximum Sharpe (Asset-Capped, e.g., max 40% per asset)
    * **Sector-Based Constraints** (e.g., max 50% in 'Tech', min 5% in 'Commodities')
* **Advanced Optimization Methods** ⭐ NEW:
    * **Risk Parity:** Equalizes risk contribution across assets (not capital allocation)
    * **Hierarchical Risk Parity (HRP):** Uses machine learning clustering for robust diversification
    * **Minimum Correlation:** Minimizes average pairwise correlation for maximum diversification benefit
    * **Maximum Diversification:** Maximizes the diversification ratio (weighted volatility / portfolio volatility)
* **Walk-Forward Validation:** The gold standard of backtesting. It trains the optimizer on one period and validates its performance out-of-sample on a separate test period.
* **Advanced Visualizations:** Generates a suite of interactive Plotly charts:
    * Efficient Frontier (with CML)
    * Asset Allocation Pie Charts
    * Sector Allocation Pie Charts
    * Asset-level Risk Contribution (Stacked Bar)
    * Sector-level Risk Contribution (Stacked Bar)
    * Rolling Sharpe Ratio
    * Cumulative Returns & Drawdown Plots
    * Correlation Heatmaps
    * Monte Carlo Simulations:
        * Future Projections: Simulate 1000+ potential future paths for your portfolio using Geometric Brownian Motion.
        * Actual vs. Simulated: Overlay your portfolio's *actual* realized performance on top of the simulations for a powerful "reality check."
        * Probability Analysis: Calculate the probability of your portfolio exceeding specific return thresholds (e.g., "65% chance of >10% return").
* Flexible & Extensible: All core math and plotting functions can be imported and used individually.

## Installation

### 1. From PyPI (Recommended)

```bash
pip install quant-reporter
```

### 2. For Development (Local Install)
```bash
git clone https://github.com/manan-tech/quant_reporter.git
cd quant_reporter
# Important: Use -e for editable mode so changes are reflected immediately
pip install -e .
```

⸻

## Quickstart: The Main Report Functions

This package provides two main report generators: a simple one and an advanced one.

### 1. create_full_report

Generates a simple performance report for a single asset or your user-defined portfolio.
```python
import quant_reporter as qr
import os
from datetime import datetime, timedelta

# Can be a single ticker or a portfolio dict
my_assets = {'AAPL': 0.6, 'MSFT': 0.4}
today = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')

qr.create_full_report(
    assets=my_assets,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date=today,
    filename=os.path.join(desktop, 'My_Simple_Report.html')
)
```

### 2. create_combined_report (Recommended)

This is the most powerful, professional-grade report. It performs a full walk-forward validation by:
	1.	Analyzing your user portfolio over the full period.
	2.	Training the optimizers on your train_start to train_end data.
	3.	Testing those optimized portfolios on the out-of-sample data (train_end to today).

```python
import quant_reporter as qr
import os
from datetime import datetime, timedelta

my_portfolio = {'AAPL': 0.6, 'MSFT': 0.4}
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')

qr.create_combined_report(
    portfolio_dict=my_portfolio,
    benchmark_ticker='SPY',
    train_start='2015-01-01',
    train_end='2021-12-31',
    filename=os.path.join(desktop, 'My_Combined_Report.html'),
    risk_free_rate=0.065
)
```

### 3. Black-Litterman Portfolio Analysis

The Black-Litterman model blends market equilibrium returns with your own investor views.

```python
import quant_reporter as qr

# 1. Define your absolute views (Ticker: Expected Annual Return)
bl_views = {
    'NVDA': 0.25,  # You expect 25% return for NVDA
    'PFE': -0.05   # You expect -5% return for PFE
}

# 2. Define confidence in those views (0.0=Uncertain to 1.0=Certain)
bl_confidences = {
    'NVDA': 0.9,
    'PFE': 0.5
}

# 3. Generate the report (integrated into create_combined_report)
qr.create_combined_report(
    portfolio_dict={'AAPL': 0.5, 'MSFT': 0.5},
    benchmark_ticker='SPY',
    train_start='2015-01-01',
    train_end='2023-12-31',
    filename='Black_Litterman_Report.html',
    bl_views=bl_views,
    bl_view_confidences=bl_confidences
)
```

### 4. Advanced Portfolio Optimization ⭐ NEW

Compare 8 different portfolio strategies including 4 advanced optimization methods:

```python
import quant_reporter as qr

# Your portfolio
my_portfolio = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.25, 'AMZN': 0.25}

# Generate comprehensive optimization report
qr.create_optimization_report(
    portfolio_dict=my_portfolio,
    benchmark_ticker='SPY',
    start_date='2020-01-01',
    end_date='2024-12-31',
    filename='Advanced_Optimization_Report.html',
    risk_free_rate=0.05
)
```

**What's included in the report:**
- **8 Portfolio Strategies:**
  1. Equal Weight (Baseline)
  2. Minimum Volatility (Traditional MPT)
  3. Balanced (40% Cap)
  4. Max Sharpe (Unconstrained MPT)
  5. **Risk Parity** - Equal risk contribution
  6. **HRP** - Hierarchical clustering
  7. **Min Correlation** - Minimize pairwise correlation
  8. **Max Diversification** - Maximize diversification ratio

- **Comprehensive Comparisons:**
  - Composition pie charts (by asset and sector)
  - Risk contribution analysis
  - Cumulative returns
  - Drawdown analysis
  - Rolling Sharpe ratio
  - Monthly returns heatmap
  - Efficient frontier with all strategies

**When to use each optimizer:**
- **Risk Parity:** When assets have different volatilities and you want balanced risk exposure
- **HRP:** When correlation structures are unstable or you want robust out-of-sample performance
- **Min Correlation:** During crisis periods when correlations spike
- **Max Diversification:** For long-only portfolios seeking maximum risk reduction

See `examples/example_advanced_optimization.py` for a complete working example.

### 5. create_monte_carlo_report

Generates a dedicated Monte Carlo simulation report.

```python
import quant_reporter as qr

# ... define assets ...

qr.create_monte_carlo_report(
    weights={'AAPL': 0.6, 'MSFT': 0.4},
    mean_returns=mean_returns, # from get_optimization_inputs
    cov_matrix=cov_matrix,     # from get_optimization_inputs
    num_simulations=1000,
    time_horizon=252,
    filename='Monte_Carlo_Report.html'
)
```

⸻

## Advanced Usage: As a Library

You can import and use all the core functions individually to build custom analyses.

Example: Get data and find a Min Vol portfolio

```python
import quant_reporter as qr
import pandas as pd

# 1. Define tickers and get data
tickers = ['AAPL', 'MSFT', 'GOOG', 'GLD']
data = qr.get_data(tickers, '2020-01-01', '2023-12-31')

# 2. Get optimization inputs
mean_returns, cov_matrix, log_returns = qr.get_optimization_inputs(data)

# 3. Define constraints (e.g., must sum to 1, no shorting)
num_assets = len(tickers)
bounds = tuple((0, 1) for _ in range(num_assets))
# 'build_constraints' creates the simple sum-to-one rule
constraints = qr.build_constraints(num_assets, tickers) 

# 4. Find the optimal weights
min_vol_weights = qr.find_optimal_portfolio(
    objective_func=qr.objective_min_variance,
    mean_returns=mean_returns,
    cov_matrix=cov_matrix,
    bounds=bounds,
    constraints=constraints,
    risk_free_rate=0.05
)

weights_df = pd.Series(min_vol_weights, index=tickers, name="Weights")
print("--- Minimum Volatility Weights ---")
print(weights_df[weights_df > 0].map(lambda x: f"{x:.2%}"))

# 5. Create and show a plot
fig = qr.plot_correlation_heatmap(log_returns)
# fig.show() # Uncomment to display
```

⸻

Full Example: All Reports with Sector Constraints

Here is a complete, copy-pasteable example using the complex US portfolio from our discussion. It runs both main reports and includes display names and sector constraints.

```python
import quant_reporter as qr
import os
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- 1. Define Your New Portfolio ---
my_portfolio = {
    # --- Technology ---
    'AAPL': 0.05,   # Apple
    'MSFT': 0.07,   # Microsoft
    'NVDA': 0.02,   # Nvidia
    'TSLA': 0.03,   # Tesla

    # --- Pharma / Healthcare ---
    'JNJ': 0.04,    # Johnson & Johnson
    'PFE': 0.03,    # Pfizer

    # --- Infrastructure / Industrials ---
    'CAT': 0.03,    # Caterpillar
    'VMC': 0.02,    # Vulcan Materials

    # --- Defense / Aerospace ---
    'LMT': 0.05,    # Lockheed Martin
    'RTX': 0.04,    # Raytheon Technologies

    # --- Banking / Financials ---
    'JPM': 0.05,    # JPMorgan Chase
    'HDB': 0.03,    # HDFC Bank (ADR)

    # --- Energy / Utilities ---
    'XOM': 0.04,    # Exxon Mobil
    'NEE': 0.03,    # NextEra Energy

    # --- Logistics / Transportation ---
    'FDX': 0.04,    # FedEx
    'UNP': 0.03,    # Union Pacific

    # --- Consumer / Retail ---
    'WMT': 0.04,    # Walmart
    'PG': 0.03,     # Procter & Gamble

    # --- Metals / Commodities ---
    'GLD': 0.04,    # SPDR Gold Shares
    'SLV': 0.03,    # iShares Silver Trust

    # --- Broad Market ETFs ---
    'DIA': 0.03,    # Dow Jones ETF
    'VTI': 0.03,    # Total Market ETF

    # --- Risk-Free / T-Bills ---
    'BIL': 0.02     # 1–3 Month Treasury Bill ETF
}

# --- 2. Define Display Names ---
display_names = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'TSLA': 'Tesla',
    'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer', 'CAT': 'Caterpillar', 
    'VMC': 'Vulcan Materials', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon', 'PLTR': 'Palantir',
    'JPM': 'JPMorgan Chase', 'HDB': 'HDFC Bank (ADR)',
    'XOM': 'Exxon Mobil', 'NEE': 'NextEra Energy', 'FDX': 'FedEx', 
    'UNP': 'Union Pacific', 'WMT': 'Walmart', 'PG': 'Procter & Gamble',
    'GLD': 'SPDR Gold ETF', 'SLV': 'iShares Silver ETF', 'DIA': 'Dow Jones ETF',
    'VTI': 'Total Market ETF', 'BIL': '1–3 Month T-Bill ETF',
    'SPY': 'S&P 500 ETF' # Benchmark
}

# --- 3. Define Sector Map & Caps (using original tickers) ---
sector_map = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'TSLA': 'Tech',
    'JNJ': 'Healthcare', 'PFE': 'Healthcare',
    'CAT': 'Industrials', 'VMC': 'Industrials', 'LMT': 'Defence', 'RTX': 'Defence',
    'FDX': 'Industrials', 'UNP': 'Industrials',
    'JPM': 'Financials', 'HDB': 'Financials',
    'XOM': 'Energy', 'NEE': 'Utilities',
    'WMT': 'Consumer', 'PG': 'Consumer',
    'GLD': 'Commodities', 'SLV': 'Commodities',
    'DIA': 'Broad Market', 'VTI': 'Broad Market',
    'BIL': 'Cash'
}

sector_caps = {
    'Tech': 0.40,         # Max 40% in Technology
    'Industrials': 0.30,
    'Defence': 0.30,
    'Healthcare': 0.20,
    'Financials': 0.20,
    'Energy': 0.15,
    'Utilities': 0.15,
    'Consumer': 0.20,
    'Commodities': 0.10,
    'Broad Market': 0.10,
    'Cash': 0.10
}

sector_mins = {
    'Tech': 0.05,         # At least 5% in Technology
    'Healthcare': 0.01,   # At least 1%
    'Industrials': 0.01,
    'Defence': 0.01,
    'Defense': 0.01,
    'Financials': 0.01,
    'Energy': 0.01,
    'Utilities': 0.01,
    'Logistics': 0.01,
    'Consumer': 0.01,
    'Commodities': 0.02,  # At least 2% in Commodities
    'Broad Market': 0.01,
    'Cash': 0.05          # At least 5% in Cash
}

# --- 4. Define Benchmark & Paths ---
benchmark_ticker = 'SPY'
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')


def run_full_reports():
    """
    Runs all three major report generators.
    """
    print("--- 1. RUNNING create_full_report ---")
    report_path_full = os.path.join(desktop, 'Portfolio_Report.html')
    
    try:
        qr.create_full_report(
            assets=my_portfolio, 
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            filename=report_path_full,
            display_names=display_names,
            risk_free_rate=0.065
        )
        print(f"--- Full Report Generated: {report_path_full} ---")
    except Exception as e:
        print(f"Error in create_full_report: {e}")
        traceback.print_exc()

    print("\n--- 2. RUNNING create_optimization_report ---")
    opt_report_path = os.path.join(desktop, 'Portfolio_Optimization_Report.html')
    
    try:
        qr.create_optimization_report(
            portfolio_dict=my_portfolio,
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date='2019-12-31',
            risk_free_rate=0.065,
            filename=opt_report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
        )
        print(f"--- Optimization Report Generated: {opt_report_path} ---")
    except Exception as e:
        print(f"Error in create_optimization_report: {e}")
        traceback.print_exc()

    print("\n--- 3. RUNNING create_combined_report ---")
    comb_report_path = os.path.join(desktop, 'Combined_Report.html')
    
    try:
        qr.create_combined_report(
            portfolio_dict=my_portfolio,
            benchmark_ticker=benchmark_ticker,
            train_start='2010-01-01',
            train_end='2023-12-31',
            risk_free_rate=0.065,
            filename=comb_report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
            sector_mins=sector_mins
        )
        print(f"--- Combined Report Generated: {comb_report_path} ---")
    except Exception as e:
        print(f"Error in create_combined_report: {e}")
        traceback.print_exc()

    print("\n--- 4. RUNNING create_monte_carlo_report ---")
    mc_report_path = os.path.join(desktop, 'Monte_Carlo_Report.html')
    
    try:
        # 1. Fetch data for simulation inputs
        # We use a recent history (e.g. last 3 years) to estimate stats
        sim_start = '2020-01-01'
        sim_end = '2023-12-31'
        tickers = list(my_portfolio.keys())
        data_mc = qr.get_data(tickers, sim_start, sim_end)
        
        # 2. Get Mean Returns & Covariance Matrix
        mean_returns, cov_matrix, _ = qr.get_optimization_inputs(data_mc)
        
        # 3. Align weights with the sorted columns from yfinance
        sorted_tickers = sorted(tickers)
        weights_list = [my_portfolio[t] for t in sorted_tickers]
        
        qr.create_monte_carlo_report(
            weights=weights_list,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            num_simulations=1000,
            time_horizon=252, # 1 Year
            filename=mc_report_path
        )
        print(f"--- Monte Carlo Report Generated: {mc_report_path} ---")
    except Exception as e:
        print(f"Error in create_monte_carlo_report: {e}")
        traceback.print_exc()

def test_individual_functions():
    """
    Demonstrates using the package as a library.
    """
    print("\n--- 4. TESTING INDIVIDUAL LIBRARY FUNCTIONS ---")
    
    try:
        tickers = list(my_portfolio.keys())
        friendly_tickers = [display_names.get(t, t) for t in tickers]
        
        # --- Test get_data ---
        print("\nTesting get_data...")
        data = qr.get_data(tickers, '2022-01-01', '2022-12-31')
        print(data.tail(3))
        
        # --- Test calculate_metrics ---
        print("\nTesting calculate_metrics...")
        data_with_bench = qr.get_data(tickers + [benchmark_ticker], '2022-01-01', '2022-12-31')
        data_with_bench.rename(columns=display_names, inplace=True)
        
        metrics, plot_data = qr.calculate_metrics(
            data_with_bench, 
            asset_col='Apple',
            benchmark_col='S&P 500 ETF',
            risk_free_rate=0.065
        )
        print(f"CAGR (Apple): {metrics['CAGR (Asset)']}")
        print(f"Beta (Apple): {metrics['Beta (vs Benchmark)']}")
        
        # --- Test individual plotting function ---
        print("\nTesting individual plot function (plot_correlation_heatmap)...")
        # Get inputs using *friendly_tickers*
        mean_returns, cov_matrix, log_returns = qr.get_optimization_inputs(data_with_bench[friendly_tickers])
        fig = qr.plot_correlation_heatmap(log_returns)

        print("Plotly figure object created successfully.")

        print("\n--- Individual tests complete ---")
        
    except Exception as e:
        print(f"Error during individual tests: {e}")
        traceback.print_exc()

# --- Run the tests ---
if __name__ == "__main__":
    run_full_reports()
    test_individual_functions()
```

## Detailed Report Documentation

Quant Reporter offers four primary report types, each serving a different stage of the investment process.

### 📊 1. Full Portfolio Report (`create_full_report`)
*   **Purpose:** Simple performance audit for a fixed asset mix.
*   **Best for:** Client reporting, quarterly reviews, and tracking performance against a standard benchmark (e.g., SPY).
*   **Key Sections:**
    *   **Cumulative Returns:** Growth of $1 vs. benchmark.
    *   **Regression Analysis:** Scatter plot with Alpha (intercept) and Beta (slope) to identify market sensitivity.
    *   **Rolling Returns:** Summary table of 1Y, 3Y, and 5Y rolling performance.

### 🧪 2. Optimization Report (`create_optimization_report`)
*   **Purpose:** Forward-looking asset allocation based on historical risk/reward.
*   **Best for:** Rebalancing, initial portfolio construction, and exploring the Efficient Frontier.
*   **Key Sections:**
    *   **Efficient Frontier:** Standard MPT curve showing the risk/return tradeoff.
    *   **Strategy Comparison:** Composition mix of Min Vol, Max Sharpe, and Equal Weight strategies.
    *   **Risk Contribution:** Decomposition of portfolio volatility by asset and sector.
    *   **Correlation Heatmap:** Visualizes diversification benefits (or lack thereof).

### 🏆 3. Combined Report (`create_combined_report`)
*   **Purpose:** The flagship "Professional Grade" analysis. Integrates optimization with out-of-sample validation.
*   **Best for:** Stress testing strategies, identifying "overfitting" in backtests, and analyzing rebalancing impact.
*   **Exclusive Features:**
    *   **Walk-Forward Validation:** Distinct Training and Testing periods to simulate real-world forward performance.
    *   **Weight Evolution Plot:** New area chart showing how weights drift due to price changes and reset during rebalancing.
    *   **Black-Litterman Integration:** Blends market equilibrium with investor views (absolute or relative).

### 🎲 4. Monte Carlo Report (`create_monte_carlo_report`)
*   **Purpose:** Probabilistic risk assessment and goal planning.
*   **Best for:** Retiremet planning and quantifying "Worst Case" scenarios.
*   **Key Sections:**
    *   **Future Paths:** 1000 simulated trajectories for the next year.
    *   **Distribution of Returns:** Histogram of final outcomes (Log-Normal).
    *   **Probability Curve:** The "Goal Likelihood" chart (e.g., "90% chance of remaining above -5% drawdown").

---

## Performance & Risk Metrics

Every report automatically calculates and displays the following core metrics (Annualized where applicable):

| Metric | Category | Description |
| :--- | :--- | :--- |
| **CAGR** | Performance | Compound Annual Growth Rate over the period. |
| **Vol (Ann)** | Risk | Annualized Standard Deviation of daily returns. |
| **Sharpe Ratio** | Risk-Adj | Excess return per unit of volatility (uses T-Bill benchmark). |
| **Sortino Ratio** | Risk-Adj | Excess return per unit of *downside* volatility. |
| **Max Drawdown** | Risk | Peak-to-trough decline (the "pain" metric). |
| **Calmar Ratio** | Risk-Adj | CAGR / Max Drawdown (efficiency of recovery). |
| **Value at Risk (VaR)** | Risk | 95% confidence level daily loss potential. |
| **Conditional VaR (CVaR)** | Risk | Expected loss *if* the VaR threshold is breached. |
| **Alpha** | Benchmark | Excess return independent of the market. |
| **Beta** | Benchmark | Sensitivity to market moves (Beta > 1 is aggressive). |
| **R-Squared** | Benchmark | Percentage of returns explained by the benchmark. |

---

## API & Function Reference

### Main Report Functions
	•	create_full_report(assets, benchmark_ticker, start_date, end_date, ...)
	•	create_combined_report(portfolio_dict, benchmark_ticker, train_start, train_end, ...)

### Key Parameters:
	•	assets (dict or str): Either a portfolio dictionary (e.g., {'AAPL': 0.5}) or a single ticker string (e.g., 'AAPL').
	•	portfolio_dict (dict): A dictionary of tickers and their weights.
	•	benchmark_ticker (str): The ticker for the benchmark (e.g., 'SPY').
	•	risk_free_rate (float or str): A float (e.g., 0.05).
	•	display_names (dict): Optional. A dictionary to map tickers to friendly names (e.g., {'AAPL': 'Apple'}).
	•	sector_map (dict): Optional. Maps raw tickers to sector names (e.g., {'AAPL': 'Tech'}).
	•	sector_caps (dict): Optional. Sets maximum allocation for sectors (e.g., {'Tech': 0.4}).
	•	sector_mins (dict): Optional. Sets minimum allocation for sectors (e.g., {'Tech': 0.05}).

## Core Library Functions

### You can import these directly for custom scripts.
	•	get_data(tickers, start_date, end_date): Fetches and cleans price data.
	•	calculate_metrics(data, asset_col, benchmark_col, ...): Returns (metrics_dict, plot_data_dict).
	•	get_optimization_inputs(price_data): Returns (mean_returns, cov_matrix, log_returns).
	•	build_constraints(num_assets, raw_tickers, ...): Creates constraint objects for the optimizer.
	•	find_optimal_portfolio(objective_func, ...): The core SciPy optimizer.
	•	plot_efficient_frontier(mean_returns, ...): Returns a Plotly Figure object.
	•	plot_risk_contribution(...): Returns a Plotly Figure object.
	•	(…and all other plot_ functions in plotting.py and opt_plotting.py)

### Future Development
*   **Rebalancing Logic:** While visualization is supported, the core rebalancing function is still to be added properly (full transactional simulation, tax-loss harvesting, etc.).
*   **Advanced Attribution:** Implement Brinson Performance Attribution (Allocation vs. Selection effects).
*   **Rolling Validation:** True "walk-forward" optimization with periodic rebalancing (e.g., re-optimize every quarter).
*   **AI-Driven Insights:** Integrate LLMs to generate textual commentary and risk warnings based on the report data.
*   **More Simulation Models:** Add support for GARCH or Bootstrapping models in Monte Carlo.

## License

This project is licensed under the MIT License.

---
