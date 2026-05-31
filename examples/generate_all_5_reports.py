import logging
import os
import sys
import pandas as pd
import traceback
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import quant_reporter as qr

# Enable library logging
qr.enable_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Reports directory
REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Inherit the big portfolio ---
my_portfolio = {
    'AAPL': 0.05, 'MSFT': 0.07, 'NVDA': 0.02, 'TSLA': 0.03, 'JNJ': 0.04, 'PFE': 0.03,
    'CAT': 0.03, 'VMC': 0.02, 'LMT': 0.05, 'RTX': 0.04, 'JPM': 0.05, 'HDB': 0.03,
    'XOM': 0.04, 'NEE': 0.03, 'FDX': 0.04, 'UNP': 0.03, 'WMT': 0.04, 'PG': 0.03,
    'GLD': 0.04, 'SLV': 0.03, 'DIA': 0.03, 'VTI': 0.03, 'BIL': 0.02
}

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
start = '2015-01-01'
end = '2023-12-31'
rf_rate = 0.05

def main():
    logger.info("=========================================")
    logger.info("   GENERATING ALL 5 INDIVIDUAL REPORTS   ")
    logger.info("=========================================")

    # Each generator builds its own ReportContext from the same inputs
    # (portfolio_dict, benchmark, train window) and fetches data as needed.

    # 1. Portfolio Report
    p_path = os.path.join(REPORTS_DIR, '01_Portfolio_Report.html')
    qr.create_portfolio_report(
        portfolio_dict=my_portfolio,
        benchmark_ticker=benchmark_ticker,
        train_start=start,
        train_end=end,
        filename=p_path,
        display_names=display_names,
        risk_free_rate=rf_rate
    )

    # 2. Optimization Report
    o_path = os.path.join(REPORTS_DIR, '02_Optimization_Report.html')
    qr.create_optimization_report(
        portfolio_dict=my_portfolio,
        benchmark_ticker=benchmark_ticker,
        train_start=start,
        train_end=end,
        filename=o_path,
        display_names=display_names,
        sector_map=sector_map,
        sector_caps=sector_caps,
        sector_mins=sector_mins,
        risk_free_rate=rf_rate
    )
    
    # 3. Monte Carlo Report
    m_path = os.path.join(REPORTS_DIR, '03_Monte_Carlo_Report.html')
    qr.create_monte_carlo_report(
        portfolio_dict=my_portfolio,
        benchmark_ticker=benchmark_ticker,
        train_start=start,
        train_end=end,
        filename=m_path,
        display_names=display_names,
        risk_free_rate=rf_rate
    )
    
    # 4. Validation Report
    v_path = os.path.join(REPORTS_DIR, '04_Validation_Report.html')
    qr.create_validation_report(
        portfolio_dict=my_portfolio,
        benchmark_ticker=benchmark_ticker,
        train_start=start,
        train_end=end,
        filename=v_path,
        display_names=display_names,
        risk_free_rate=rf_rate
    )
    
    # 5. Factor Report
    f_path = os.path.join(REPORTS_DIR, '05_Factor_Report.html')
    qr.create_factor_report(
        portfolio_dict=my_portfolio,
        benchmark_ticker=benchmark_ticker,
        train_start=start,
        train_end=end,
        filename=f_path,
        display_names=display_names,
        sector_map=sector_map,
        risk_free_rate=rf_rate
    )

    logger.info("✅ All 5 reports generated successfully in the examples folder.")


if __name__ == "__main__":
    main()
