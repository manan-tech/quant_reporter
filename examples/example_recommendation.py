"""example_recommendation.py — opt-in recommendation bundle + transparent report.

Offline (synthetic fixture). Run:
    python examples/example_recommendation.py
Produces examples/Recommendation_Report.html and embeds the section into a
backtest report (examples/Backtest_With_Reco.html).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test"))

import functools
from conftest import make_synthetic_prices
import quant_reporter as qr


def main():
    prices = make_synthetic_prices(seed=42, n_days=900)        # AAA/BBB/CCC + BMK
    assets = prices[["AAA", "BBB", "CCC"]]
    current = {"AAA": 0.50, "BBB": 0.30, "CCC": 0.20}
    cost = functools.partial(qr.transaction_cost_model, commission_bps=1.0, spread_bps=5.0)

    results = qr.backtest_many(
        {"EqualWeight": qr.equal_weight, "RiskParity": qr.risk_parity,
         "MaxSharpe": qr.max_sharpe},
        prices, benchmark="BMK", rebalance="M", cost_model=cost)

    rec = qr.recommend(assets, current_weights=current, results=results,
                       cost_model=cost, vol_target=0.10, max_drawdown_limit=0.20,
                       max_weight=0.40)

    print(rec.to_text())

    out = os.path.join(os.path.dirname(__file__), "Recommendation_Report.html")
    qr.create_recommendation_report(rec, path=out)
    print(f"\nRecommendation report: {out}")

    bt = os.path.join(os.path.dirname(__file__), "Backtest_With_Reco.html")
    results["RiskParity"].report(bt, recommendation=rec)
    print(f"Backtest report with recommendation: {bt}")


if __name__ == "__main__":
    main()
