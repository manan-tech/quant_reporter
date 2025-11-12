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
* **Walk-Forward Validation:** The gold standard of backtesting. It trains the optimizer on one period and validates its performance out-of-sample on a separate test period.
* **Advanced Visualizations:** Generates a suite of interactive Plotly charts:
    * Efficient Frontier (with CML)
    * Asset Allocation Pie Charts
    * Sector Allocation Pie Charts
    * Asset-level Risk Contribution (Stacked Bar)
    * **Sector-level Risk Contribution** (Stacked Bar)
    * Rolling Sharpe Ratio
    * Cumulative Returns & Drawdown Plots
    * Correlation Heatmaps
* **Flexible & Extensible:** All core math and plotting functions can be imported and used individually.

## Installation

### 1. From PyPI (Recommended)

```bash
pip install quant-reporter
```

### 2. For Development (Local Install)
```bash
git clone https://github.com/manan-tech/quant_reporter.git
cd quant_reporter
pip install -e .[test]
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
	•	Monte Carlo Simulation: Add a create_monte_carlo_report to simulate future returns.
	•	Brinson Attribution: Add performance attribution (Allocation vs. Selection).
	•	Rolling Validation: Implement a true “walk-forward” optimization with periodic rebalancing.

## License

This project is licensed under the MIT License.

---
