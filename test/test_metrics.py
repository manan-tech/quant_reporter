from quant_reporter.metrics import calculate_metrics
import pytest

def test_calculate_metrics(sample_data):
    """
    Tests the calculate_metrics function with fixture data.
    """
    metrics, plot_data = calculate_metrics(sample_data, 'MSFT', 'SPY')
    
    assert "CAGR (Asset)" in metrics
    assert "Beta (vs Benchmark)" in metrics
    assert "Sharpe Ratio (Asset)" in metrics
    
    assert "daily_returns" in plot_data
    assert "cumulative_returns" in plot_data
    
    beta = float(metrics["Beta (vs Benchmark)"])
    assert beta == pytest.approx(1.2, abs=0.1)