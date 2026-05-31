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


def transaction_cost_model(trades, commission_bps=1.0, spread_bps=5.0,
                           impact_model=None, portfolio_value=1.0):
    """Trading cost = commission (bps) + half-spread (bps), applied to traded notional.

    Args:
        trades: dict/Series of signed per-asset weight deltas (e.g. portfolio_turnover()['trades']).
        commission_bps: commission in basis points, charged on traded notional.
        spread_bps: full bid-ask spread in bps; you cross HALF of it per unit traded.
        impact_model: optional callable(trades)->float extra cost_frac (market impact);
                      None for SP1 (bps + half-spread only). Reserved future hook.
        portfolio_value: scales cost_cash; cost_frac is value-independent.

    Returns:
        dict {'cost_frac', 'cost_cash', 'cost_breakdown': {'commission','spread','impact'}}.
    """
    t = _to_series(trades)
    notional = float(t.abs().sum())  # two-way traded notional (fraction of portfolio)
    commission = (commission_bps / 1e4) * notional
    spread = (spread_bps / 2.0 / 1e4) * notional
    impact = float(impact_model(trades)) if callable(impact_model) else 0.0
    cost_frac = commission + spread + impact
    return {
        "cost_frac": cost_frac,
        "cost_cash": cost_frac * portfolio_value,
        "cost_breakdown": {"commission": commission, "spread": spread, "impact": impact},
    }


def generate_rebalance_dates(index, mode="calendar", freq="M", tau=0.05,
                             per_asset_band=None, target_weights=None):
    """Rebalance trading days for the simulate loop.

    mode='calendar' (implemented): first trading day of each period — drop-in for
    rebalancing.py's inline freq logic. freq: None (only the first day), 'M'/'Q'/'Y',
    or an int (every N trading days). The first index date is always included.

    mode='threshold'/'band' (NOT in SP1): drift-/band-triggered rebalancing is path-
    dependent and is decided inside simulate_strategy's loop on data up to d-1.
    """
    if mode != "calendar":
        raise NotImplementedError(
            "Only mode='calendar' is implemented in SP1; threshold/band rebalancing is "
            "decided in-loop by simulate_strategy (future work)."
        )
    idx = pd.DatetimeIndex(index)
    if len(idx) == 0:
        return idx
    first = pd.DatetimeIndex([idx[0]])
    if freq is None:
        return first
    if isinstance(freq, int):
        sel = idx[::freq]
    elif freq == "M":
        sel = idx[idx.to_series().dt.month.diff() != 0]
    elif freq == "Q":
        sel = idx[idx.to_series().dt.quarter.diff() != 0]
    elif freq == "Y":
        sel = idx[idx.to_series().dt.year.diff() != 0]
    else:
        raise ValueError(f"Unknown freq {freq!r}; expected None, 'M'/'Q'/'Y', or int.")
    return sel.union(first).sort_values()


def simulate_strategy(price_data, target_weights, cost_model=None, rebalance="M",
                      initial_value=1.0, cash_drag=0.0):
    """Cost-aware backtest of a target allocation (dict) or a dated weight schedule.

    target_weights: dict (constant target, like simulate_rebalanced_portfolio) OR a
        DataFrame schedule (index = rebalance dates, columns = tickers). When a DataFrame
        is given, `rebalance` is ignored (the schedule carries its own dates) and the
        schedule is forward-filled causally onto trading days.
    cost_model: callable(trades_series)->{'cost_frac',...} or None (frictionless). The
        shipped default for the create_* layer is functools.partial(transaction_cost_model, ...).
    cash_drag: annual drag applied daily (value *= 1 - cash_drag/252 each day).

    Returns {'wealth' (Growth series starting at initial_value), 'weights' (drifted history),
    'blotter' (per-rebalance dicts), 'turnover' (series by rebalance date), 'cost_drag'
    (float = 1 - Π(1-cost_frac)), 'summary' (headline dict)}.
    """
    if isinstance(target_weights, pd.DataFrame):
        tickers = list(target_weights.columns)
        prices = price_data[tickers]
        target_by_day = target_weights.reindex(prices.index, method="ffill")
        valid = target_by_day.dropna(how="all")
        if valid.empty:
            raise ValueError("schedule has no dates within price_data's range")
        prices = prices.loc[valid.index[0]:]
        target_by_day = target_by_day.loc[valid.index[0]:].fillna(0.0)
        changed = target_by_day.diff().abs().sum(axis=1) > 1e-12
        changed.iloc[0] = True
        rebalance_days = set(target_by_day.index[changed])
    else:
        tickers = list(target_weights.keys())
        prices = price_data[tickers]
        tw = pd.Series(target_weights, dtype=float).reindex(tickers).fillna(0.0)
        target_by_day = pd.DataFrame(np.tile(tw.values, (len(prices), 1)),
                                     index=prices.index, columns=tickers)
        if rebalance is None:
            rebalance_days = {prices.index[0]}
        else:
            rebalance_days = set(generate_rebalance_dates(prices.index, "calendar", freq=rebalance))
            rebalance_days.add(prices.index[0])

    daily_ret = prices.pct_change().fillna(0.0)
    n = len(prices)
    value = float(initial_value)
    cur = target_by_day.iloc[0].values.astype(float)
    daily_cash = 1.0 - cash_drag / 252.0

    blotter = []
    turnovers = {}
    cost_factor = 1.0

    def _rebalance_to(prev_w, new_w, date, vnow):
        nonlocal cost_factor
        to = portfolio_turnover(dict(zip(tickers, prev_w)), dict(zip(tickers, new_w)))
        cf = 0.0
        if cost_model is not None:
            cf = float(cost_model(to["trades"]).get("cost_frac", 0.0))
        cost_factor *= (1.0 - cf)
        turnovers[date] = to["turnover"]
        blotter.append({"date": date, "turnover": to["turnover"], "cost_frac": cf})
        return vnow * (1.0 - cf)

    # entry at day 0 (from cash to first target)
    value = _rebalance_to(np.zeros(len(tickers)), cur, prices.index[0], value)
    values = [value]
    weight_hist = [dict(zip(tickers, cur))]

    for i in range(1, n):
        date = prices.index[i]
        r = daily_ret.iloc[i].values
        value *= (1.0 + float(np.dot(cur, r)))
        value *= daily_cash
        cur = cur * (1.0 + r)
        s = cur.sum()
        if s > 0:
            cur = cur / s
        if date in rebalance_days:
            tgt = target_by_day.loc[date].values.astype(float)
            value = _rebalance_to(cur, tgt, date, value)
            cur = tgt
        values.append(value)
        weight_hist.append(dict(zip(tickers, cur)))

    wealth = pd.Series(values, index=prices.index, name="Portfolio")
    weights_df = pd.DataFrame(weight_hist, index=prices.index)
    turnover_series = pd.Series(turnovers, dtype=float).sort_index()
    cost_drag = 1.0 - cost_factor
    dd = drawdown_stats(wealth)
    summary = {
        "terminal_wealth": float(wealth.iloc[-1]),
        "total_return": float(wealth.iloc[-1] / initial_value - 1.0),
        "n_rebalances": int(len(turnover_series)),
        "avg_turnover": float(turnover_series.mean()) if len(turnover_series) else 0.0,
        "cost_drag": float(cost_drag),
        "max_drawdown": float(dd["max_drawdown"]),
    }
    return {"wealth": wealth, "weights": weights_df, "blotter": blotter,
            "turnover": turnover_series, "cost_drag": float(cost_drag), "summary": summary}
