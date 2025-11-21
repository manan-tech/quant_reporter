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