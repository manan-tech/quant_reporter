import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from conftest import make_synthetic_prices
from example_walk_forward_backtest import run_walk_forward_demo


def test_walk_forward_demo_runs_offline():
    prices = make_synthetic_prices(n_days=756)
    out = run_walk_forward_demo(prices, {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
                                "BMK", "2021-01-01", "2022-06-30",
                                commission_bps=1.0, spread_bps=5.0)
    comp = out["comparison"]
    assert comp["best_by_dsr"] in out["backtests"]
    for s in comp["summary"].values():
        assert 0.0 <= s["psr"] <= 1.0
        assert 0.0 <= s["dsr"] <= 1.0
    for bt in out["backtests"].values():
        assert np.isfinite(bt["summary"]["terminal_wealth"]) and bt["summary"]["terminal_wealth"] > 0
