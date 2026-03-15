"""
Black-Litterman Portfolio Optimization — Sample Report
========================================================
This script generates a comprehensive combined report using the
quant_reporter library with the Black-Litterman model.

Configuration:
    - 30 diversified stocks + ETFs across 11 sectors
    - 8 absolute investor views + 3 relative views
    - Sector caps (max allocation) and sector minimums
    - Per-asset cap at 40%
    - Walk-Forward Validation:
        Train: 2010-01-01 → 2022-12-31
        Test:  2023-01-01 → present (with quarterly rebalancing)
    - Benchmark: SPY (S&P 500)
"""

import logging
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import quant_reporter as qr
import traceback
import warnings
warnings.filterwarnings('ignore')

# Enable library logging so you can see progress
qr.enable_logging(logging.INFO)


# ── 1. Portfolio Definition (30 holdings, weights sum to 1.0) ──────────────

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


# ── 5. Investor Views (Black-Litterman) ───────────────────────────────────

# 5a. Absolute views — "NVDA returns 25% per annum"
bl_views = {
    'NVDA': 0.25,     # Bullish: AI/GPU demand → 25% expected annual return
    'MSFT': 0.15,     # Bullish: Cloud + AI integration → 15%
    'AAPL': 0.08,     # Mildly bullish: Mature growth → 8%
    'PFE':  -0.05,    # Bearish: Post-COVID revenue decline → -5%
    'XOM':  0.12,     # Bullish: Energy prices staying elevated → 12%
    'GLD':  0.10,     # Bullish: Inflation hedge, central bank buying → 10%
    'JPM':  0.11,     # Bullish: Higher interest rate margin → 11%
    'AMD':  0.20,     # Bullish: AI chip competition with NVDA → 20%
}

# Confidence in each absolute view (0.0 = very uncertain, 1.0 = very confident).
bl_view_confidences = {
    'NVDA': 0.9,      # Very high conviction
    'MSFT': 0.8,      # High conviction
    'AAPL': 0.6,      # Moderate conviction
    'PFE':  0.5,      # Moderate conviction
    'XOM':  0.7,      # Fairly confident
    'GLD':  0.8,      # High conviction (macro view)
    'JPM':  0.6,      # Moderate conviction
    'AMD':  0.7,      # Fairly confident
}

# 5b. Relative views — "NVDA outperforms AMD by 5%"
# Each tuple is (outperformer_ticker, underperformer_ticker, expected_spread)
bl_relative_views = [
    ('NVDA', 'AMD',  0.05),   # NVDA outperforms AMD by 5%
    ('XOM',  'NEE',  0.04),   # Exxon outperforms NextEra by 4% (energy > utilities)
    ('JPM',  'HDB',  0.03),   # JPMorgan outperforms HDFC by 3% (US > India financials)
]

# Confidence per relative view (same order as above)
bl_relative_view_confidences = [0.8, 0.6, 0.5]


# ── 6. Run the Report ─────────────────────────────────────────────────────

def main():
    benchmark = 'SPY'
    REPORTS_DIR = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(REPORTS_DIR, 'Black_Litterman_Report.html')

    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("  BLACK-LITTERMAN PORTFOLIO REPORT")
    logger.info("=" * 70)
    logger.info("  Tickers     : %d", len(portfolio))
    logger.info("  Sectors     : %d", len(set(sector_map.values())))
    logger.info("  Abs Views   : %d", len(bl_views))
    logger.info("  Rel Views   : %d", len(bl_relative_views))
    logger.info("  Rebalance   : Quarterly")
    logger.info("  Train       : 2010-01-01 → 2022-12-31")
    logger.info("  Test        : 2023-01-01 → today")
    logger.info("  Benchmark   : %s", benchmark)
    logger.info("  Output      : %s", report_path)
    logger.info("=" * 70)

    try:
        qr.create_combined_report(
            portfolio_dict=portfolio,
            benchmark_ticker=benchmark,
            train_start='2010-01-01',
            train_end='2022-12-31',
            risk_free_rate=0.05,
            filename=report_path,
            display_names=display_names,
            sector_map=sector_map,
            sector_caps=sector_caps,
            sector_mins=sector_mins,
            bl_views=bl_views,
            bl_view_confidences=bl_view_confidences,
            bl_relative_views=bl_relative_views,
            bl_relative_view_confidences=bl_relative_view_confidences,
            rebalance_freq='Q',    # Quarterly rebalancing
            desc=True,
            denoise_cov=True,
            n_components=3
        )
        logger.info("✅  Report saved: %s", report_path)
    except Exception as e:
        logger.error("❌  Report failed: %s", e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
