"""Flagship walk-forward, cost-aware backtest.

Rolling windows -> per-strategy weight schedule -> cost-aware simulate_strategy ->
honest selection (Deflated Sharpe) across Equal / Min-Vol / Max-Sharpe / User.

`run_walk_forward_demo` takes a price DataFrame so it is offline-testable on the
synthetic fixture; `main()` fetches real data via the public API.
"""
import functools
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import quant_reporter as qr


def run_walk_forward_demo(price_data, portfolio, benchmark, train_start, train_end,
                          commission_bps=1.0, spread_bps=5.0):
    """Run the full walk-forward, cost-aware pipeline and return the pieces.

    Returns {'rolling', 'schedule', 'backtests', 'comparison'}.
    """
    ctx = qr.build_context_from_prices(price_data, portfolio, benchmark, train_start, train_end)
    rolling_df, schedule = qr.run_rolling_windows(ctx, return_schedule=True)

    cost_model = functools.partial(qr.transaction_cost_model,
                                   commission_bps=commission_bps, spread_bps=spread_bps)
    asset_prices = price_data[ctx.friendly_tickers]
    oos_returns = {}
    backtests = {}
    for strat, wsched in schedule.items():
        if wsched.dropna(how="all").empty:
            continue
        res = qr.simulate_strategy(asset_prices, wsched, cost_model=cost_model)
        backtests[strat] = res
        oos_returns[strat] = res["wealth"].pct_change().dropna()

    comparison = qr.compare_strategies_oos(oos_returns, n_trials=len(oos_returns) or 1)
    return {"rolling": rolling_df, "schedule": schedule,
            "backtests": backtests, "comparison": comparison}


def main():
    portfolio = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.20, "NVDA": 0.15, "AMZN": 0.15}
    benchmark = "SPY"
    train_start, train_end = "2018-01-01", "2023-12-31"
    price_data = qr.get_data(list(portfolio) + [benchmark], train_start, train_end)

    out = run_walk_forward_demo(price_data, portfolio, benchmark, train_start, train_end)
    print("Best strategy by Deflated Sharpe:", out["comparison"]["best_by_dsr"])
    for name, s in out["comparison"]["summary"].items():
        print(f"  {name:16s}  SR={s['sharpe']:.2f}  PSR={s['psr']:.2f}  DSR={s['dsr']:.2f}")
    for name, bt in out["backtests"].items():
        print(f"  {name:16s}  terminal={bt['summary']['terminal_wealth']:.3f}  "
              f"cost_drag={bt['cost_drag']:.4f}  maxDD={bt['summary']['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
