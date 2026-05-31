"""Cost-aware backtest engine & execution primitives.

SP1a slice: portfolio_turnover, drawdown_stats. (SP1b adds transaction_cost_model,
generate_rebalance_dates, simulate_strategy.) Pure, offline-testable, no new deps.
"""
import numpy as np
import pandas as pd

from .metrics import compute_drawdown

TRADING_DAYS = 252


def _to_series(weights):
    if isinstance(weights, pd.Series):
        return weights.astype(float)
    return pd.Series(weights, dtype=float)


def portfolio_turnover(weights_before, weights_after, convention="one_way"):
    """Turnover between two weight vectors.

    Aligns on the union of tickers (missing => 0). `trades` are signed deltas
    (after - before). one_way turnover = 0.5*sum(|trades|); two_way = sum(|trades|).

    Returns dict: {'turnover', 'buys', 'sells', 'trades' (Series of signed deltas)}.
    """
    if convention not in ("one_way", "two_way"):
        raise ValueError(f"Unknown convention {convention!r}; expected 'one_way' or 'two_way'.")
    before = _to_series(weights_before)
    after = _to_series(weights_after)
    idx = before.index.union(after.index)
    before = before.reindex(idx).fillna(0.0)
    after = after.reindex(idx).fillna(0.0)
    trades = after - before
    abs_sum = float(trades.abs().sum())
    buys = float(trades[trades > 0].sum())
    sells = float(-trades[trades < 0].sum())
    turnover = abs_sum if convention == "two_way" else 0.5 * abs_sum
    return {"turnover": turnover, "buys": buys, "sells": sells, "trades": trades}


def _drawdown_episodes(curve):
    """Split an underwater curve into episodes: (peak_date, trough_date, recovery_date, depth, length)."""
    episodes = []
    in_dd = False
    peak_date = None
    trough_date = None
    trough_val = 0.0
    dates = curve.index
    for i, (date, val) in enumerate(curve.items()):
        if not in_dd and val < 0:
            in_dd = True
            peak_date = dates[i - 1] if i > 0 else date
            trough_date = date
            trough_val = val
        elif in_dd:
            if val < trough_val:
                trough_val = val
                trough_date = date
            if val >= 0:  # recovered
                episodes.append({"peak_date": peak_date, "trough_date": trough_date,
                                 "recovery_date": date, "depth": float(trough_val),
                                 "length": int(curve.index.get_loc(date) - curve.index.get_loc(peak_date))})
                in_dd = False
    if in_dd:  # ongoing drawdown at series end
        episodes.append({"peak_date": peak_date, "trough_date": trough_date,
                         "recovery_date": None, "depth": float(trough_val),
                         "length": int(curve.index.get_loc(dates[-1]) - curve.index.get_loc(peak_date))})
    return episodes


def drawdown_stats(wealth, top_n=5, periods_per_year=252):
    """Drawdown analytics for a realized wealth (Growth-of-$1) path.

    Returns dict: {'max_drawdown' (neg decimal), 'underwater_curve' (Series),
    'worst_drawdowns' (top_n episodes by depth), 'ulcer_index' (pct points),
    'pain_index' (mean |dd|, decimal)}.
    """
    dd = compute_drawdown(wealth)
    curve = dd.curve
    episodes = _drawdown_episodes(curve)
    episodes.sort(key=lambda e: e["depth"])  # most negative first
    ulcer = float(np.sqrt(np.mean((100.0 * curve.values) ** 2)))
    pain = float(curve.abs().mean())
    return {
        "max_drawdown": float(dd.max_dd),
        "underwater_curve": curve,
        "worst_drawdowns": episodes[:top_n],
        "ulcer_index": ulcer,
        "pain_index": pain,
    }
