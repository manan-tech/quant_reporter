"""
Portfolio rebalancing simulation.

Simulates periodic rebalancing of a portfolio during a backtest period.
Supports buy-and-hold (None), monthly ('M'), quarterly ('Q'), yearly ('Y'),
or custom day intervals (int).
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def simulate_rebalanced_portfolio(price_data, weights_dict, rebalance_freq=None):
    """
    Simulates a portfolio with periodic rebalancing, returning Growth-of-$1.
    
    Between rebalance dates, weights drift with price changes (buy & hold).
    At each rebalance date, weights are reset to the target allocation.
    
    Args:
        price_data (pd.DataFrame): Daily price data for all assets.
        weights_dict (dict): Target weights, e.g. {'AAPL': 0.3, 'MSFT': 0.7}.
        rebalance_freq (str|int|None): Rebalancing frequency:
            - None: Buy-and-hold (no rebalancing)
            - 'M': Monthly (first trading day of each month)
            - 'Q': Quarterly (first trading day of each quarter)
            - 'Y': Yearly (first trading day of each year)
            - int: Every N trading days
            
    Returns:
        pd.Series: Growth-of-$1 series indexed by date.
    """
    tickers = list(weights_dict.keys())
    target_weights = np.array([weights_dict[t] for t in tickers])
    prices = price_data[tickers].copy()
    
    if prices.empty:
        return pd.Series(dtype=float)
    
    # -- Buy-and-hold (no rebalancing) -- identical to get_portfolio_price
    if rebalance_freq is None:
        normalized = prices / prices.iloc[0]
        return (normalized * target_weights).sum(axis=1)
    
    # -- Determine rebalance dates --
    rebalance_dates = _get_rebalance_dates(prices.index, rebalance_freq)
    logger.info("Rebalancing %d times over %d trading days (freq=%s)",
                len(rebalance_dates), len(prices), rebalance_freq)
    
    # -- Simulate --
    daily_returns = prices.pct_change().fillna(0)
    
    portfolio_value = 1.0
    current_weights = target_weights.copy()
    portfolio_values = [portfolio_value]
    
    for i in range(1, len(prices)):
        date = prices.index[i]
        
        # Update portfolio value with today's returns
        asset_returns = daily_returns.iloc[i].values
        weighted_return = np.sum(current_weights * asset_returns)
        portfolio_value *= (1 + weighted_return)
        
        # Update drifted weights
        current_weights = current_weights * (1 + asset_returns)
        weight_sum = current_weights.sum()
        if weight_sum > 0:
            current_weights = current_weights / weight_sum
        
        # Rebalance if this is a rebalance date
        if date in rebalance_dates:
            current_weights = target_weights.copy()
        
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=prices.index, name="Portfolio")


def _get_rebalance_dates(date_index, freq):
    """
    Returns the set of dates on which to rebalance.
    
    Args:
        date_index: DatetimeIndex of trading days.
        freq: 'M' | 'Q' | 'Y' | int (every N days).
        
    Returns:
        set: Dates to rebalance on.
    """
    if isinstance(freq, int):
        # Every N trading days
        return set(date_index[::freq])
    
    if freq == 'M':
        # First trading day of each month
        grouped = pd.Series(date_index, index=date_index).groupby(
            [date_index.year, date_index.month]
        ).first()
        return set(grouped.values)
    
    if freq == 'Q':
        # First trading day of each quarter
        grouped = pd.Series(date_index, index=date_index).groupby(
            [date_index.year, (date_index.month - 1) // 3]
        ).first()
        return set(grouped.values)
    
    if freq == 'Y':
        # First trading day of each year
        grouped = pd.Series(date_index, index=date_index).groupby(
            date_index.year
        ).first()
        return set(grouped.values)
    
    logger.warning("Unknown rebalance_freq '%s', defaulting to buy-and-hold.", freq)
    return set()
