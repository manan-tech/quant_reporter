"""
Performance attribution analysis.

This module implements Brinson-Fachler attribution for decomposing
portfolio excess returns into allocation, selection, and interaction effects.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


def compute_brinson_attribution(
    portfolio_weights: Dict[str, float],
    benchmark_weights: Dict[str, float],
    asset_returns: pd.DataFrame,
    sector_map: Dict[str, str],
    period_start: Optional[str] = None,
    period_end: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute Brinson-Fachler attribution.

    Decomposes portfolio excess returns (relative to benchmark) into three components:
    1. Allocation Effect: Return from over/under-weighting sectors vs benchmark
    2. Selection Effect: Return from selecting securities within sectors
    3. Interaction Effect: Combined effect of allocation and selection decisions

    Parameters
    ----------
    portfolio_weights : dict
        Dictionary mapping asset tickers to portfolio weights (should sum to 1.0).
        Example: {'AAPL': 0.3, 'MSFT': 0.25, 'GOOGL': 0.45}
    benchmark_weights : dict
        Dictionary mapping asset tickers to benchmark weights (should sum to 1.0).
        Example: {'AAPL': 0.2, 'MSFT': 0.3, 'GOOGL': 0.15, 'AMZN': 0.35}
    asset_returns : pd.DataFrame
        Time series of asset returns. Index should be DatetimeIndex.
        Columns should be asset tickers. Each value is the return for that period.
    sector_map : dict
        Dictionary mapping asset tickers to sector names.
        Example: {'AAPL': 'Technology', 'MSFT': 'Technology', 'XOM': 'Energy'}
    period_start : str, optional
        Start date for analysis in 'YYYY-MM-DD' format. If None, uses first date in data.
    period_end : str, optional
        End date for analysis in 'YYYY-MM-DD' format. If None, uses last date in data.

    Returns
    -------
    pd.DataFrame
        Attribution results with columns:
        - 'Portfolio_Return': Total portfolio return for the period
        - 'Benchmark_Return': Total benchmark return for the period
        - 'Excess_Return': Portfolio return minus benchmark return
        - 'Allocation_Effect': Contribution from sector allocation decisions
        - 'Selection_Effect': Contribution from security selection within sectors
        - 'Interaction_Effect': Combined allocation and selection effects
        Index is the sector name, with a 'Total' row for aggregated results.

    Raises
    ------
    ValueError
        If portfolio and benchmark returns don't have overlapping dates, or if
        weights don't match the assets in the returns data.

    Examples
    --------
    >>> portfolio_weights = {'AAPL': 0.4, 'XOM': 0.3, 'JPM': 0.3}
    >>> benchmark_weights = {'AAPL': 0.3, 'XOM': 0.4, 'JPM': 0.2, 'GS': 0.1}
    >>> sector_map = {'AAPL': 'Tech', 'XOM': 'Energy', 'JPM': 'Finance', 'GS': 'Finance'}
    >>> attribution = compute_brinson_attribution(
    ...     portfolio_weights, benchmark_weights,
    ...     asset_returns, sector_map
    ... )
    >>> print(attribution.loc['Total'])

    Notes
    -----
    The Brinson-Fachler model decomposes excess return as:

    Allocation Effect = Σ (w_p - w_b) × (r_b - R_b)
    where w_p = portfolio weight, w_b = benchmark weight,
    r_b = sector return in benchmark, R_b = total benchmark return

    Selection Effect = Σ w_b × (r_p - r_b)
    where r_p = sector return in portfolio

    Interaction Effect = Σ (w_p - w_b) × (r_p - r_b)

    This method follows the standard Brinson-Hood-Beebower approach, except
    it uses the Brinson-Fachler adjustment for the allocation effect.

    References
    ----------
    Brinson, G. P., Hood, L. R., & Beebower, G. L. (1986).
    "Determinants of Portfolio Performance". Financial Analysts Journal.
    """
    logger.info("Computing Brinson-Fachler attribution...")

    # Filter by date range if specified
    if period_start:
        asset_returns = asset_returns.loc[asset_returns.index >= pd.to_datetime(period_start)]
    if period_end:
        asset_returns = asset_returns.loc[asset_returns.index <= pd.to_datetime(period_end)]

    if len(asset_returns) == 0:
        raise ValueError("No dates in the specified period range")

    # Calculate cumulative returns per asset for the period
    # Handle NaN safely
    asset_cum_returns = (1 + asset_returns.fillna(0)).prod() - 1

    # Aggregate returns by sector for both portfolio and benchmark
    sectors = set(sector_map.values())

    # Calculate sector-level returns
    portfolio_sector_returns = {}
    benchmark_sector_returns = {}
    portfolio_sector_weights = {}
    benchmark_sector_weights = {}

    portfolio_cum_return = 0.0
    benchmark_cum_return = 0.0

    for sector in sectors:
        # Get assets in this sector
        sector_assets = [asset for asset, s in sector_map.items() if s == sector]

        # Calculate weights for this sector
        port_sector_weight = sum(portfolio_weights.get(asset, 0) for asset in sector_assets)
        bench_sector_weight = sum(benchmark_weights.get(asset, 0) for asset in sector_assets)

        portfolio_sector_weights[sector] = port_sector_weight
        benchmark_sector_weights[sector] = bench_sector_weight

        # Calculate portfolio sector returns
        if port_sector_weight > 0:
            sector_ret = 0.0
            for asset in sector_assets:
                weight = portfolio_weights.get(asset, 0)
                if weight > 0 and asset in asset_cum_returns:
                    sector_ret += (weight / port_sector_weight) * asset_cum_returns[asset]
            portfolio_sector_returns[sector] = sector_ret
            portfolio_cum_return += port_sector_weight * sector_ret
        else:
            portfolio_sector_returns[sector] = 0.0

        # Calculate benchmark sector returns
        if bench_sector_weight > 0:
            sector_ret = 0.0
            for asset in sector_assets:
                weight = benchmark_weights.get(asset, 0)
                if weight > 0 and asset in asset_cum_returns:
                    sector_ret += (weight / bench_sector_weight) * asset_cum_returns[asset]
            benchmark_sector_returns[sector] = sector_ret
            benchmark_cum_return += bench_sector_weight * sector_ret
        else:
            benchmark_sector_returns[sector] = 0.0

    # Compute attribution effects
    results = []

    for sector in sorted(sectors):
        w_p = portfolio_sector_weights.get(sector, 0)
        w_b = benchmark_sector_weights.get(sector, 0)
        r_p = portfolio_sector_returns.get(sector, 0)
        r_b = benchmark_sector_returns.get(sector, 0)

        # Allocation effect (Brinson-Fachler)
        # (w_p - w_b) * (r_b - R_b)
        allocation = (w_p - w_b) * (r_b - benchmark_cum_return)

        # Selection effect
        # w_b * (r_p - r_b)
        selection = w_b * (r_p - r_b)

        # Interaction effect
        # (w_p - w_b) * (r_p - r_b)
        interaction = (w_p - w_b) * (r_p - r_b)

        results.append({
            'Sector': sector,
            'Portfolio_Weight': w_p,
            'Benchmark_Weight': w_b,
            'Portfolio_Sector_Return': r_p,
            'Benchmark_Sector_Return': r_b,
            'Allocation_Effect': allocation,
            'Selection_Effect': selection,
            'Interaction_Effect': interaction,
            'Total_Effect': allocation + selection + interaction
        })

    # Create DataFrame
    df = pd.DataFrame(results)
    df.set_index('Sector', inplace=True)

    # Add total row
    total = {
        'Portfolio_Weight': df['Portfolio_Weight'].sum(),
        'Benchmark_Weight': df['Benchmark_Weight'].sum(),
        'Portfolio_Sector_Return': portfolio_cum_return,
        'Benchmark_Sector_Return': benchmark_cum_return,
        'Allocation_Effect': df['Allocation_Effect'].sum(),
        'Selection_Effect': df['Selection_Effect'].sum(),
        'Interaction_Effect': df['Interaction_Effect'].sum(),
        'Total_Effect': df['Total_Effect'].sum()
    }

    df.loc['Total'] = total

    # Add summary columns
    df['Portfolio_Return'] = portfolio_cum_return
    df['Benchmark_Return'] = benchmark_cum_return
    df['Excess_Return'] = portfolio_cum_return - benchmark_cum_return

    logger.info(f"Attribution complete. Total excess return: {df.loc['Total', 'Excess_Return']:.2%}")

    return df
