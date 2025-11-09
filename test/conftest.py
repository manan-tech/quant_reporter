import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """
    Creates a sample DataFrame to avoid hitting yfinance in tests.
    """
    dates = pd.date_range(start='2020-01-01', periods=252, freq='B') 
    
    bench_returns = np.random.normal(0.0005, 0.01, 252)
    asset_returns = 0.0002 + (1.2 * bench_returns) + np.random.normal(0, 0.005, 252)
    
    bench_price = 100 * (1 + bench_returns).cumprod()
    asset_price = 150 * (1 + asset_returns).cumprod()
    
    data = pd.DataFrame({
        'MSFT': asset_price,
        'SPY': bench_price
    }, index=dates)
    
    return data