from quant_reporter.metrics import calculate_metrics
import pytest


def test_calculate_metrics(sample_data):
    """
    Tests the calculate_metrics function with fixture data.
    """
    metrics, plot_data = calculate_metrics(sample_data, 'MSFT', 'SPY')
    
    assert_metrics(metrics)
    assert_plot_data(plot_data)
    assert_beta(metrics)


def assert_metrics(metrics):
    assert "CAGR (Asset)" in metrics
    assert "Beta (vs Benchmark)" in metrics
    assert "Sharpe Ratio (Asset)" in metrics


def assert_plot_data(plot_data):
    assert "daily_returns" in plot_data
    assert "cumulative_returns" in plot_data


def assert_beta(metrics):
    beta = float(metrics["Beta (vs Benchmark)"])
    assert beta == pytest.approx(1.2, abs=0.1)