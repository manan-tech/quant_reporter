"""
Example: Generates a Combined Report.
This is the flagship analysis integrating optimization with out-of-sample validation.
"""
import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import quant_reporter as qr
import traceback
import warnings
warnings.filterwarnings('ignore')

# Enable library logging
qr.enable_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Reports directory (same folder as this script)
REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Define Your Portfolio ---
my_portfolio = {
    'AAPL': 0.05, 'MSFT': 0.07, 'NVDA': 0.02, 'TSLA': 0.03, 'JNJ': 0.04, 'PFE': 0.03,
    'CAT': 0.03, 'VMC': 0.02, 'LMT': 0.05, 'RTX': 0.04, 'JPM': 0.05, 'HDB': 0.03,
    'XOM': 0.04, 'NEE': 0.03, 'FDX': 0.04, 'UNP': 0.03, 'WMT': 0.04, 'PG': 0.03,
    'GLD': 0.04, 'SLV': 0.03, 'DIA': 0.03, 'VTI': 0.03, 'BIL': 0.02
}

# --- Define Display Names ---
display_names = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'NVDA': 'Nvidia', 'TSLA': 'Tesla',
    'JNJ': 'Johnson & Johnson', 'PFE': 'Pfizer', 'CAT': 'Caterpillar', 
    'VMC': 'Vulcan Materials', 'LMT': 'Lockheed Martin', 'RTX': 'Raytheon',
    'JPM': 'JPMorgan Chase', 'HDB': 'HDFC Bank (ADR)', 'XOM': 'Exxon Mobil', 
    'NEE': 'NextEra Energy', 'FDX': 'FedEx', 'UNP': 'Union Pacific', 
    'WMT': 'Walmart', 'PG': 'Procter & Gamble', 'GLD': 'SPDR Gold ETF', 
    'SLV': 'iShares Silver ETF', 'DIA': 'Dow Jones ETF', 'VTI': 'Total Market ETF', 
    'BIL': '1–3 Month T-Bill ETF', 'SPY': 'S&P 500 ETF'
}

# --- Define Sector Map & Constraints ---
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
    'Tech': 0.40, 'Industrials': 0.30, 'Defence': 0.30, 'Healthcare': 0.20,
    'Financials': 0.20, 'Energy': 0.15, 'Utilities': 0.15, 'Consumer': 0.20,
    'Commodities': 0.10, 'Broad Market': 0.10, 'Cash': 0.10
}

sector_mins = {
    'Tech': 0.05, 'Healthcare': 0.01, 'Industrials': 0.01, 'Defence': 0.01,
    'Financials': 0.01, 'Energy': 0.01, 'Utilities': 0.01, 'Consumer': 0.01,
    'Commodities': 0.02, 'Broad Market': 0.01, 'Cash': 0.05
}

benchmark_ticker = 'SPY'

def run_report():
    logger.info("Running create_combined_report")
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
            sector_mins=sector_mins
        )
        logger.info("Combined Report Generated: %s", comb_path)
    except Exception as e:
        logger.error("Error in create_combined_report: %s", e)
        traceback.print_exc()

if __name__ == "__main__":
    run_report()
