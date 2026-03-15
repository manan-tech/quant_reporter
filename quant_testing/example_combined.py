import quant_reporter as qr
import os
import traceback
from datetime import datetime, timedelta

def run_full_reports(portfolio_dict, display_names, sector_map, sector_caps, benchmark_ticker):
    """
    Runs all three major report generators.
    """
    print("--- 1. RUNNING create_full_report ---")
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    report_path = os.path.join(desktop, 'Portfolio_Report.html')
    
    try:
        qr.create_full_report(
            assets=portfolio_dict, 
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date= (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            filename=report_path,
            display_names=display_names,
            risk_free_rate=0.0525 # Example: 5.25%
        )
        print(f"--- Full Report Generated: {report_path} ---")
    except Exception as e:
        print(f"Error generating create_full_report: {e}")
        traceback.print_exc()

    print("\n--- 2. RUNNING create_optimization_report ---")
    opt_report_path = os.path.join(desktop, 'Portfolio_Optimization_Report.html')
    
    try:
        qr.create_optimization_report(
            portfolio_dict=portfolio_dict,
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date='2022-12-31',
            risk_free_rate=0.0525,
            filename=opt_report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps
        )
        print(f"--- Optimization Report Generated: {opt_report_path} ---")
    except Exception as e:
        print(f"Error generating create_optimization_report: {e}")
        traceback.print_exc()

    print("\n--- 3. RUNNING create_combined_report ---")
    comb_report_path = os.path.join(desktop, 'Combined_Report.html')
    
    try:
        qr.create_combined_report(
            portfolio_dict=portfolio_dict,
            benchmark_ticker=benchmark_ticker,
            train_start='2010-01-01',
            train_end='2022-12-31',
            risk_free_rate=0.0525,
            filename=comb_report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps
        )
        print(f"--- Combined Report Generated: {comb_report_path} ---")
    except Exception as e:
        print(f"Error generating create_combined_report: {e}")
        traceback.print_exc()

def test_individual_functions(portfolio_dict, display_names, benchmark_ticker):
    """
    Demonstrates using the package as a library.
    """
    print("\n--- 4. TESTING INDIVIDUAL LIBRARY FUNCTIONS ---")
    
    try:
        # --- Test get_data ---
        print("\nTesting get_data...")
        tickers = list(portfolio_dict.keys())
        data = qr.get_data(tickers, '2022-01-01', '2022-12-31')
        print(data.head())

        # --- Test calculate_metrics ---
        print("\nTesting calculate_metrics...")
        # Add benchmark to data and rename
        data_with_bench = qr.get_data(tickers + [benchmark_ticker], '2022-01-01', '2022-12-31')
        data_with_bench.rename(columns=display_names, inplace=True)
        
        metrics, plot_data = qr.calculate_metrics(
            data_with_bench, 
            asset_col=display_names['AAPL'], 
            benchmark_col=display_names[benchmark_ticker],
            risk_free_rate=0.05
        )
        print(f"CAGR (Apple): {metrics['CAGR (Asset)']}")
        print(f"Beta (Apple): {metrics['Beta (vs Benchmark)']}")
        
        # --- Test individual plotting function ---
        print("\nTesting individual plot function...")
        fig = qr.plot_correlation_heatmap(plot_data['log_returns'])
        # To show the plot:
        # fig.show() 
        print("Plotly figure object created successfully.")

        print("\n--- Individual tests complete ---")
        
    except Exception as e:
        print(f"Error during individual tests: {e}")
        traceback.print_exc()

# --- Main execution ---
if __name__ == "__main__":
    
    # --- 1. Define Portfolio, Benchmark, and Display Names ---
    benchmark_ticker = 'SPY'
    
    my_portfolio = {
        # --- Technology ---
        'AAPL': 0.05, 'MSFT': 0.07, 'NVDA': 0.02, 'TSLA': 0.03, 'PLTR': 0.02,
        # --- Pharma / Healthcare ---
        'JNJ': 0.04, 'PFE': 0.03,
        # --- Infrastructure / Industrials ---
        'CAT': 0.03, 'VMC': 0.02,
        # --- Defense / Aerospace ---
        'LMT': 0.04, 'RTX': 0.03,
        # --- Banking / Financials ---
        'JPM': 0.05, 'HDB': 0.03,
        # --- Energy / Utilities ---
        'XOM': 0.04, 'NEE': 0.03,
        # --- Logistics / Transportation ---
        'FDX': 0.04, 'UNP': 0.03,
        # --- Consumer / Retail ---
        'WMT': 0.04, 'PG': 0.03,
        # --- Metals / Commodities ---
        'GLD': 0.04, 'SLV': 0.03,
        # --- Broad Market ETFs ---
        'DIA': 0.03, 'VTI': 0.03,
        # --- Risk-Free / T-Bills ---
        'BIL': 0.02
        # Note: 'SPY' (benchmark) is intentionally removed from the portfolio
    }
    
    display_names = {
        'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'TSLA': 'Tesla',
        'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer', 'CAT': 'Caterpillar', 
        'VMC': 'Vulcan Materials', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon',
        'PLTR': 'Palantir', 'JPM': 'JPMorgan Chase', 'HDB': 'HDFC Bank (ADR)',
        'XOM': 'Exxon Mobil', 'NEE': 'NextEra Energy', 'FDX': 'FedEx', 
        'UNP': 'Union Pacific', 'WMT': 'Walmart', 'PG': 'Procter & Gamble',
        'GLD': 'SPDR Gold ETF', 'SLV': 'iShares Silver ETF', 'DIA': 'Dow Jones ETF',
        'VTI': 'Total Market ETF', 'BIL': '1–3 Month T-Bill ETF',
        'SPY': 'S&P 500 ETF' # Benchmark
    }

    # --- 2. Define NEW Sector Mappings and Caps ---
    sector_map = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'TSLA': 'Tech', 'PLTR': 'Tech',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare',
        'CAT': 'Industrials', 'VMC': 'Industrials', 'LMT': 'Industrials', 'RTX': 'Industrials',
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
        'Industrials': 0.30,  # Max 30% in Industrials
        'Healthcare': 0.20,
        'Financials': 0.20,
        'Energy': 0.15,
        'Utilities': 0.15,
        'Consumer': 0.20,
        'Commodities': 0.10,  # Max 10% in Commodities
        'Broad Market': 0.10,
        'Cash': 0.10
    }

    # --- 3. Run the report generators ---
    run_full_reports(my_portfolio, display_names, sector_map, sector_caps, benchmark_ticker)
    
    # --- 4. Run the individual function tests ---
    test_individual_functions(my_portfolio, display_names, benchmark_ticker)