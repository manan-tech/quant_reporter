"""
Example: Generates a dedicated Monte Carlo Simulation Report.
Provides probabilistic risk assessment and goal planning based on historical data.
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

def run_report():
    logger.info("Running create_monte_carlo_report")
    mc_report_path = os.path.join(REPORTS_DIR, 'Monte_Carlo_Report.html')
    
    try:
        # 1. Fetch data for simulation inputs (use recent history to estimate stats)
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
        logger.info("Monte Carlo Report Generated: %s", mc_report_path)
    except Exception as e:
        logger.error("Error in create_monte_carlo_report: %s", e)
        traceback.print_exc()

if __name__ == "__main__":
    run_report()
