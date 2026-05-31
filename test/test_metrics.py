import pytest


def test_calculate_metrics_removed():
    """Guard: calculate_metrics must not exist anywhere after the clean break."""
    import quant_reporter as qr
    import quant_reporter.metrics as m
    assert not hasattr(qr, "calculate_metrics")
    assert not hasattr(m, "calculate_metrics")