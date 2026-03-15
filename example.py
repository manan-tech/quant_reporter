"""
Main example: runs all 4 report types + individual function tests.
All reports are saved to the ./reports/ directory in the project root.
"""
import logging
import quant_reporter as qr
import os
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Enable library logging
qr.enable_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ── Reports directory (same folder as this script) ─────────────────────────
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


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
    'VMC': 'Vulcan Materials', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon',
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
    'Tech': 0.40,
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
    'Tech': 0.05,
    'Healthcare': 0.01,
    'Industrials': 0.01,
    'Defence': 0.01,
    'Financials': 0.01,
    'Energy': 0.01,
    'Utilities': 0.01,
    'Consumer': 0.01,
    'Commodities': 0.02,
    'Broad Market': 0.01,
    'Cash': 0.05
}

# --- 4. Define Benchmark ---
benchmark_ticker = 'SPY'


def run_full_reports():
    """Runs all four major report generators."""

    # ── Report 1: Full Portfolio Report ─────────────────────────────────
    logger.info("1. Running create_full_report")
    report_path = os.path.join(REPORTS_DIR, 'Portfolio_Report.html')
    try:
        qr.create_full_report(
            assets=my_portfolio, 
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
            filename=report_path,
            display_names=display_names,
            risk_free_rate=0.065
        )
        logger.info("Full Report Generated: %s", report_path)
    except Exception as e:
        logger.error("Error in create_full_report: %s", e)
        traceback.print_exc()

    # ── Report 2: Optimization Report ──────────────────────────────────
    logger.info("2. Running create_optimization_report")
    opt_path = os.path.join(REPORTS_DIR, 'Optimization_Report.html')
    try:
        qr.create_optimization_report(
            portfolio_dict=my_portfolio,
            benchmark_ticker=benchmark_ticker,
            start_date='2010-01-01',
            end_date='2019-12-31',
            risk_free_rate=0.065,
            filename=opt_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
        )
        logger.info("Optimization Report Generated: %s", opt_path)
    except Exception as e:
        logger.error("Error in create_optimization_report: %s", e)
        traceback.print_exc()

    # ── Report 3: Combined Report ──────────────────────────────────────
    logger.info("3. Running create_combined_report")
    comb_path = os.path.join(REPORTS_DIR, 'Combined_Report.html')
    try:
        qr.create_combined_report(
            portfolio_dict=my_portfolio,
            benchmark_ticker=benchmark_ticker,
            train_start='2010-01-01',
            train_end='2023-12-31',
            risk_free_rate='auto',
            filename=comb_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
            sector_mins=sector_mins,
            bl_views={
                'NVDA': 0.20,  'MSFT': 0.15,  'AAPL': 0.08,
                'XOM': 0.12,   'JPM': 0.11,   'PFE': -0.05,
            },
            bl_view_confidences={
                'NVDA': 0.9,  'MSFT': 0.8,  'AAPL': 0.6,
                'XOM': 0.7,   'JPM': 0.6,   'PFE': 0.5,
            },
            bl_relative_views=[
                ('NVDA', 'TSLA', 0.05),
                ('XOM', 'NEE', 0.04),
            ],
            bl_relative_view_confidences=[0.7, 0.6],
            rebalance_freq='Q',
            desc=True
        )
        logger.info("Combined Report Generated: %s", comb_path)
    except Exception as e:
        logger.error("Error in create_combined_report: %s", e)
        traceback.print_exc()

    # ── Report 4: Monte Carlo Report ───────────────────────────────────
    logger.info("4. Running create_monte_carlo_report")
    mc_path = os.path.join(REPORTS_DIR, 'Monte_Carlo_Report.html')
    try:
        tickers = list(my_portfolio.keys())
        data_mc = qr.get_data(tickers, '2020-01-01', '2023-12-31')
        mean_returns, cov_matrix, _ = qr.get_optimization_inputs(data_mc)
        
        sorted_tickers = sorted(tickers)
        weights_list = [my_portfolio[t] for t in sorted_tickers]
        
        qr.create_monte_carlo_report(
            weights=weights_list,
            mean_returns=mean_returns,
            cov_matrix=cov_matrix,
            num_simulations=1000,
            time_horizon=252,
            filename=mc_path
        )
        logger.info("Monte Carlo Report Generated: %s", mc_path)
    except Exception as e:
        logger.error("Error in create_monte_carlo_report: %s", e)
        traceback.print_exc()


def test_individual_functions():
    """Demonstrates using the package as a library."""
    logger.info("5. Testing individual library functions")
    
    try:
        tickers = list(my_portfolio.keys())
        friendly_tickers = [display_names.get(t, t) for t in tickers]
        
        data = qr.get_data(tickers, '2022-01-01', '2022-12-31')
        logger.info("get_data returned %d rows x %d cols", len(data), len(data.columns))
        
        data_with_bench = qr.get_data(tickers + [benchmark_ticker], '2022-01-01', '2022-12-31')
        data_with_bench.rename(columns=display_names, inplace=True)
        
        metrics, _ = qr.calculate_metrics(
            data_with_bench, asset_col='Apple', benchmark_col='S&P 500 ETF', risk_free_rate=0.065
        )
        logger.info("CAGR (Apple): %s", metrics['CAGR (Asset)'])
        logger.info("Beta (Apple): %s", metrics['Beta (vs Benchmark)'])
        
        # Test Black-Litterman
        bl_tickers = ['AAPL', 'MSFT', 'GOOG']
        bl_data = qr.get_data(bl_tickers, '2023-01-01', '2023-12-31')
        bl_means, bl_cov, _ = qr.get_optimization_inputs(bl_data)
        bl_caps = qr.get_market_caps(bl_tickers)
        
        post_means, _ = qr.calculate_black_litterman_posterior(bl_means, bl_cov, bl_caps)
        logger.info("BL Posterior Means: %s", post_means.to_dict())
        
        logger.info("Individual tests complete")
        
    except Exception as e:
        logger.error("Error during individual tests: %s", e)
        traceback.print_exc()


if __name__ == "__main__":
    run_full_reports()
    test_individual_functions()