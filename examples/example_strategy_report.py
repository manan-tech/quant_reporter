"""example_strategy_report.py — define strategies, backtest, write an HTML report.

Offline (synthetic fixture). Run:
    python examples/example_strategy_report.py
Produces examples/Strategy_Report.html
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import functools
from conftest import make_synthetic_prices
import quant_reporter as qr


def main():
    prices = make_synthetic_prices(seed=42, n_days=900)  # AAA/BBB/CCC + BMK
    cost = functools.partial(qr.transaction_cost_model, commission_bps=1.0, spread_bps=5.0)

    strategies = {
        "EqualWeight": qr.equal_weight,
        "RiskParity": qr.risk_parity,
        "TrendFollowing": qr.trend_following,
    }
    results = qr.backtest_many(strategies, prices, benchmark="BMK",
                               rebalance="M", cost_model=cost)

    for name, res in results.items():
        m = res.metrics
        print(f"{name:16s} CAGR={m['CAGR']:.2%}  Sharpe={m['Sharpe']:.2f}  "
              f"MaxDD={m['Max Drawdown']:.2%}  DSR={res.oos_stats['dsr']:.2f}")

    out = os.path.join(os.path.dirname(__file__), "Strategy_Report.html")
    qr.create_backtest_report(results, path=out)
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
