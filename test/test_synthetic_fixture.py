import pandas as pd
from conftest import make_synthetic_prices


def test_synthetic_prices_deterministic():
    a = make_synthetic_prices()
    b = make_synthetic_prices()
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == ["AAA", "BBB", "CCC", "BMK"]
    assert len(a) == 756
    assert (a > 0).all().all()
