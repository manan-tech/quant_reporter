import quant_reporter as qr
import os

# --- 1. Import the NEW validation function ---
from quant_reporter.validation import create_validation_report

# --- 2. Define your portfolio (using real tickers as keys) ---
my_portfolio = {
    'AAPL': 0.15,
    'MSFT': 0.15,
    'GOOG': 0.10,
    'AMZN': 0.10,
    'NVDA': 0.10,
    'QQQ': 0.10,
    'VTI': 0.10,
    'GLD': 0.10,
    'SLV': 0.05,
    'SCHD': 0.05
}

# --- 3. Define your display name map (optional, but good) ---
display_names = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOG': 'Google',
    'AMZN': 'Amazon',
    'NVDA': 'Nvidia',
    'QQQ': 'Nasdaq 100 ETF',
    'VTI': 'Total Market ETF',
    'GLD': 'Gold ETF',
    'SLV': 'Silver ETF',
    'SCHD': 'Schwab Dividend ETF',
    'SPY': 'S&P 500' # <-- Also map the benchmark!
}

# --- 4. Define benchmark and paths ---
benchmark_ticker = 'SPY'
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
report_path = os.path.join(desktop, 'Validation_Report.html')

# --- 5. Run the validation report ---
create_validation_report(
    portfolio_dict=my_portfolio,
    benchmark_ticker=benchmark_ticker,
    # Train Period:
    train_start='2012-01-01',
    train_end='2019-12-31',
    # Test Period:
    test_start='2020-01-01',
    test_end='2025-11-06',
    # Other params
    risk_free_rate=0.065, # Use your 6.5% rate
    filename=report_path,
    display_names=display_names
)

print(f"Validation report generated: {report_path}")