"""
Advanced Portfolio Optimization — Sample Report
================================================
This script demonstrates the new advanced optimization methods in quant_reporter:
Risk Parity, Hierarchical Risk Parity (HRP), Minimum Correlation, and Maximum Diversification.

WHAT'S INCLUDED IN THE REPORT:
-------------------------------
1. **8 Portfolio Strategies Compared**:
   - Equal Weight (Baseline)
   - Minimum Volatility (Traditional MPT)
   - Balanced (40% Cap per asset)
   - Max Sharpe (Unconstrained MPT)
   - **Risk Parity** ⭐ NEW - Equalizes risk contribution across assets
   - **HRP** ⭐ NEW - Uses hierarchical clustering for diversification
   - **Min Correlation** ⭐ NEW - Minimizes average pairwise correlation
   - **Max Diversification** ⭐ NEW - Maximizes diversification ratio

2. **Comprehensive Visualizations**:
   - Strategy composition pie charts (by asset and sector)
   - Risk contribution analysis (by asset and sector)
   - Cumulative returns comparison
   - Drawdown analysis
   - Rolling Sharpe ratio
   - Monthly returns heatmap
   - Efficient frontier with all strategies plotted
   - Asset correlation heatmap

3. **Performance Metrics** for each strategy:
   - Total Return, Annualized Return, Volatility
   - Sharpe Ratio, Sortino Ratio, Calmar Ratio
   - Max Drawdown, Win Rate, Value at Risk (VaR)

WHY USE THESE OPTIMIZERS:
-------------------------
- **Risk Parity**: When you want balanced risk exposure (not capital exposure)
  → Good for: Portfolios with assets of varying volatility
  
- **HRP**: When you want diversification without relying on return forecasts
  → Good for: Unstable correlation structures, out-of-sample performance
  
- **Min Correlation**: When you want maximum diversification benefit
  → Good for: Crisis periods, when correlations spike
  
- **Max Diversification**: When you want to maximize the diversification ratio
  → Good for: Long-only portfolios seeking maximum risk reduction

Configuration:
    - 30 diversified stocks + ETFs across 11 sectors
    - Period: 2020-01-01 → 2024-12-31
    - Benchmark: SPY (S&P 500)
    - Risk-free rate: 5% (current T-bill rate)
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


# ── 1. Portfolio Definition (30 holdings) ──────────────────────────────────

portfolio = {
    # ── Technology (17%) ──
    'AAPL': 0.05,    # Apple
    'MSFT': 0.05,    # Microsoft
    'NVDA': 0.04,    # Nvidia
    'TSLA': 0.03,    # Tesla

    # ── Healthcare / Pharma (7%) ──
    'JNJ':  0.04,    # Johnson & Johnson
    'PFE':  0.03,    # Pfizer

    # ── Industrials (6%) ──
    'CAT':  0.03,    # Caterpillar
    'HON':  0.03,    # Honeywell

    # ── Defence / Aerospace (7%) ──
    'LMT':  0.04,    # Lockheed Martin
    'RTX':  0.03,    # Raytheon Technologies

    # ── Financials (7%) ──
    'JPM':  0.04,    # JPMorgan Chase
    'HDB':  0.03,    # HDFC Bank (ADR)

    # ── Energy (5%) ──
    'XOM':  0.03,    # Exxon Mobil
    'CVX':  0.02,    # Chevron

    # ── Utilities (4%) ──
    'NEE':  0.02,    # NextEra Energy
    'DUK':  0.02,    # Duke Energy

    # ── Consumer Staples (6%) ──
    'WMT':  0.03,    # Walmart
    'PG':   0.03,    # Procter & Gamble

    # ── Consumer Discretionary (5%) ──
    'HD':   0.03,    # Home Depot
    'NKE':  0.02,    # Nike

    # ── Transportation / Logistics (5%) ──
    'FDX':  0.03,    # FedEx
    'UNP':  0.02,    # Union Pacific

    # ── Real Estate (3%) ──
    'AMT':  0.02,    # American Tower (REIT)
    'O':    0.01,    # Realty Income

    # ── Commodities (6%) ──
    'GLD':  0.03,    # SPDR Gold Shares
    'SLV':  0.03,    # iShares Silver Trust

    # ── Broad Market ETFs (5%) ──
    'DIA':  0.02,    # Dow Jones ETF
    'VTI':  0.03,    # Total Stock Market ETF

    # ── Cash / T-Bills (4%) ──
    'BIL':  0.02,    # 1-3 Month T-Bill ETF
    'SHV':  0.02,    # iShares Short Treasury Bond ETF

    # ── Semiconductors (3%) ──
    'AMD':  0.03,    # Advanced Micro Devices
}


# ── 2. Display Names ──────────────────────────────────────────────────────

display_names = {
    'AAPL': 'Apple',           'MSFT': 'Microsoft',       'NVDA': 'Nvidia',
    'TSLA': 'Tesla',           'JNJ':  'Johnson & Johnson','PFE':  'Pfizer',
    'CAT':  'Caterpillar',     'HON':  'Honeywell',
    'LMT':  'Lockheed Martin', 'RTX':  'Raytheon',
    'JPM':  'JPMorgan Chase',  'HDB':  'HDFC Bank (ADR)',
    'XOM':  'Exxon Mobil',     'CVX':  'Chevron',
    'NEE':  'NextEra Energy',  'DUK':  'Duke Energy',
    'WMT':  'Walmart',         'PG':   'Procter & Gamble',
    'HD':   'Home Depot',      'NKE':  'Nike',
    'FDX':  'FedEx',           'UNP':  'Union Pacific',
    'AMT':  'American Tower',  'O':    'Realty Income',
    'GLD':  'SPDR Gold ETF',   'SLV':  'iShares Silver ETF',
    'DIA':  'Dow Jones ETF',   'VTI':  'Total Market ETF',
    'BIL':  '1-3 Month T-Bill','SHV':  'Short Treasury ETF',
    'AMD':  'AMD',
    'SPY':  'S&P 500 ETF',     # Benchmark
}


# ── 3. Sector Map ─────────────────────────────────────────────────────────

sector_map = {
    'AAPL': 'Technology',     'MSFT': 'Technology',     'NVDA': 'Technology',
    'TSLA': 'Technology',     'AMD':  'Technology',
    'JNJ':  'Healthcare',    'PFE':  'Healthcare',
    'CAT':  'Industrials',   'HON':  'Industrials',
    'LMT':  'Defence',       'RTX':  'Defence',
    'JPM':  'Financials',    'HDB':  'Financials',
    'XOM':  'Energy',        'CVX':  'Energy',
    'NEE':  'Utilities',     'DUK':  'Utilities',
    'WMT':  'Consumer',      'PG':   'Consumer',
    'HD':   'Consumer',      'NKE':  'Consumer',
    'FDX':  'Industrials',   'UNP':  'Industrials',
    'AMT':  'Real Estate',   'O':    'Real Estate',
    'GLD':  'Commodities',   'SLV':  'Commodities',
    'DIA':  'Broad Market',  'VTI':  'Broad Market',
    'BIL':  'Cash',          'SHV':  'Cash',
}


# ── 4. Sector Caps & Minimums ─────────────────────────────────────────────

sector_caps = {
    'Technology':   0.40,    # Max 40% in Tech
    'Healthcare':   0.20,
    'Industrials':  0.25,
    'Defence':      0.25,
    'Financials':   0.20,
    'Energy':       0.15,
    'Utilities':    0.15,
    'Consumer':     0.25,
    'Real Estate':  0.10,
    'Commodities':  0.15,
    'Broad Market': 0.15,
    'Cash':         0.10,
}

sector_mins = {
    'Technology':   0.05,    # At least 5% in Tech
    'Healthcare':   0.02,
    'Industrials':  0.02,
    'Defence':      0.02,
    'Financials':   0.02,
    'Energy':       0.01,
    'Utilities':    0.01,
    'Consumer':     0.02,
    'Real Estate':  0.01,
    'Commodities':  0.02,
    'Broad Market': 0.01,
    'Cash':         0.03,    # At least 3% in Cash
}


# ── 5. Run the Report ─────────────────────────────────────────────────────

def main():
    benchmark = 'SPY'
    REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(REPORTS_DIR, 'Advanced_Optimization_Report.html')

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("  ADVANCED PORTFOLIO OPTIMIZATION REPORT")
    logger.info("=" * 70)
    logger.info("  Tickers     : %d", len(portfolio))
    logger.info("  Sectors     : %d", len(set(sector_map.values())))
    logger.info("  Strategies  : 8 (4 traditional + 4 advanced)")
    logger.info("  Period      : 2020-01-01 → 2024-12-31")
    logger.info("  Benchmark   : %s", benchmark)
    logger.info("  Output      : %s", report_path)
    logger.info("=" * 70)
    logger.info("")
    logger.info("  Advanced Optimizers:")
    logger.info("    ✓ Risk Parity - Equal risk contribution")
    logger.info("    ✓ HRP - Hierarchical clustering")
    logger.info("    ✓ Min Correlation - Minimize pairwise correlation")
    logger.info("    ✓ Max Diversification - Maximize diversification ratio")
    logger.info("=" * 70)

    try:
        qr.create_optimization_report(
            portfolio_dict=portfolio,
            benchmark_ticker=benchmark,
            start_date='2020-01-01',
            end_date='2024-12-31',
            risk_free_rate=0.05,
            filename=report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
            sector_mins=sector_mins,
            denoise_cov=True,
            n_components=3
        )
        logger.info("")
        logger.info("✅  Report saved: %s", report_path)
        logger.info("")
        logger.info("📊  Open the report to compare all 8 strategies!")
        
    except Exception as e:
        logger.error("❌  Report failed: %s", e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
