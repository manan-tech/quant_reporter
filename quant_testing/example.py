import quant_reporter as qr
import os

desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
report_path = os.path.join(desktop, 'AAPL_Report.html')

qr.create_full_report(
    assets='AAPL', # Pass a string
    benchmark_ticker='SPY',
    start_date='2015-01-01',
    end_date='2024-12-31',
    filename=report_path
)

import quant_reporter as qr
import os

desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
report_path = os.path.join(desktop, 'Portfolio_Report.html')

# --- Define your portfolio "mix" ---
my_portfolio = {
    'AAPL': 0.4, # 40% Apple
    'MSFT': 0.3, # 30% Microsoft
    'GOOG': 0.3  # 30% Google
}

qr.create_full_report(
    assets=my_portfolio, 
    benchmark_ticker='SPY',
    start_date='2010-01-01',
    end_date='2024-12-31',
    filename=report_path
)