import numpy as np
import pandas as pd
import pytest
from quant_reporter.metrics import compute_drawdown, calculate_max_drawdown, DrawdownResult


def test_compute_drawdown_scalar_equals_curve_min():
    cum = pd.Series([1.0, 1.2, 0.9, 1.1, 0.6])
    dd = compute_drawdown(cum)
    assert isinstance(dd, DrawdownResult)
    assert dd.max_dd == dd.curve.min()
    assert dd.max_dd == pytest.approx((0.6 - 1.2) / 1.2)  # -0.5 from the 1.2 peak


def test_calculate_max_drawdown_backcompat_scalar():
    cum = pd.Series([1.0, 1.2, 0.9])
    assert calculate_max_drawdown(cum) == pytest.approx((0.9 - 1.2) / 1.2)
