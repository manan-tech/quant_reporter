import logging
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from .data import get_data
from .opt_core import get_risk_free_rate, get_optimization_inputs

logger = logging.getLogger(__name__)

@dataclass
class ReportContext:
    # Formatted Date Strings
    full_start: str
    full_end: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    # Raw inputs
    portfolio_dict: Dict[str, float]
    benchmark_ticker: str
    display_names: Optional[Dict[str, str]]
    sector_map: Optional[Dict[str, str]]
    risk_free_rate: float
    
    # New constraints for optimization
    sector_caps: Optional[Dict[str, float]]
    sector_mins: Optional[Dict[str, float]]
    
    # Black-Litterman Config
    bl_views: Optional[Dict[str, float]]
    bl_view_confidences: Optional[Dict[str, float]]
    bl_relative_views: Optional[List[Dict]]
    bl_relative_view_confidences: Optional[List[float]]
    
    # Rebalancing
    rebalance_freq: Optional[str]

    # Pre-calculated Identifiers
    tickers: List[str]
    friendly_tickers: List[str]
    friendly_benchmark: str
    friendly_sector_map: Optional[Dict[str, str]]
    user_friendly_weights: Dict[str, float]

    # Pre-calculated DataFrames
    price_data_full: pd.DataFrame
    price_data_train: pd.DataFrame
    price_data_test: pd.DataFrame
    
    # Optimization inputs from train data
    mean_returns: pd.Series
    cov_matrix: pd.DataFrame
    log_returns: pd.DataFrame

def build_context(portfolio_dict: Dict[str, float], benchmark_ticker: str,
                  train_start: str, train_end: str,
                  risk_free_rate: Union[float, str] = "auto",
                  display_names: Optional[Dict[str, str]] = None,
                  sector_map: Optional[Dict[str, str]] = None,
                  sector_caps: Optional[Dict[str, float]] = None,
                  sector_mins: Optional[Dict[str, float]] = None,
                  bl_views: Optional[Dict[str, float]] = None,
                  bl_view_confidences: Optional[Dict[str, float]] = None,
                  bl_relative_views: Optional[List[Dict]] = None,
                  bl_relative_view_confidences: Optional[List[float]] = None,
                  rebalance_freq: Optional[str] = None,
                  denoise_cov: bool = False, n_components: int = 3, **kwargs) -> ReportContext:
    """
    Factory function to fetch data and construct the ReportContext, removing duplication
    across all individual report generators.
    """
    # 1. Date Resolution
    test_start_dt = pd.to_datetime(train_end) + timedelta(days=1)
    test_end_dt = datetime.now() - timedelta(days=1)
    test_start = test_start_dt.strftime('%Y-%m-%d')
    test_end = test_end_dt.strftime('%Y-%m-%d')
    full_start = train_start
    full_end = test_end
    
    # 2. Risk Free Rate Resolution
    if isinstance(risk_free_rate, str) and risk_free_rate.lower() == 'auto':
        rfr = get_risk_free_rate()
    elif isinstance(risk_free_rate, (int, float)):
        rfr = float(risk_free_rate)
    else:
        rfr = 0.02

    # 3. Handle Dictionary Ticker Mapping
    tickers = list(portfolio_dict.keys())
    all_tickers = tickers + [benchmark_ticker]
    
    if display_names:
        friendly_tickers = [display_names.get(t, t) for t in tickers]
        friendly_benchmark = display_names.get(benchmark_ticker, benchmark_ticker)
        friendly_sector_map = {
            display_names.get(k, k): v 
            for k, v in sector_map.items()
        } if sector_map else None
        
        user_friendly_weights = {
            display_names.get(k, k): v 
            for k, v in portfolio_dict.items()
        }
    else:
        friendly_tickers = tickers
        friendly_benchmark = benchmark_ticker
        friendly_sector_map = sector_map
        user_friendly_weights = portfolio_dict

    # 4. Fetch Data Once
    price_data_full = get_data(all_tickers, full_start, full_end)
    if price_data_full is None or price_data_full.empty:
        raise ValueError("Failed to fetch price data.")

    # Apply display names to columns
    if display_names:
        price_data_full.rename(columns=display_names, inplace=True)
        # Ensure benchmark is explicitly renamed if provided in display_names
        if benchmark_ticker in display_names:
            pass # Already covered
            
    # Standardize column order (assets first, then benchmark)
    ordered_cols = friendly_tickers + [friendly_benchmark]
    missing_cols = [c for c in ordered_cols if c not in price_data_full.columns]
    if missing_cols:
         logger.warning(f"Warning: these tickers returned no data and will be dropped: {missing_cols}")
         friendly_tickers = [t for t in friendly_tickers if t not in missing_cols]
         if friendly_benchmark in missing_cols:
             raise ValueError(f"Benchmark ticker {friendly_benchmark} failed to download. Cannot proceed.")
         ordered_cols = friendly_tickers + [friendly_benchmark]
         
         # Update weights to handle dropped assets
         valid_tickers_original = [t for t in tickers if display_names.get(t, t) in friendly_tickers] if display_names else friendly_tickers
         
         total_valid_weight = sum([portfolio_dict[t] for t in valid_tickers_original])
         for ticker in list(user_friendly_weights.keys()):
             if ticker not in friendly_tickers:
                 del user_friendly_weights[ticker]
         for ticker in user_friendly_weights:
             user_friendly_weights[ticker] /= total_valid_weight

         tickers = valid_tickers_original

    price_data_full = price_data_full[ordered_cols]
    
    # 5. Split periods
    train_mask = (price_data_full.index >= pd.to_datetime(train_start)) & (price_data_full.index <= pd.to_datetime(train_end))
    price_data_train = price_data_full.loc[train_mask]
    
    test_mask = (price_data_full.index >= pd.to_datetime(test_start)) & (price_data_full.index <= pd.to_datetime(test_end))
    price_data_test = price_data_full.loc[test_mask]

    # 6. Calculate Train Optimization Inputs
    asset_data_train = price_data_train[friendly_tickers]
    mean_returns, cov_matrix, log_returns = get_optimization_inputs(
        asset_data_train, denoise_cov=denoise_cov, n_components=n_components
    )
    
    return ReportContext(
        full_start=full_start,
        full_end=full_end,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        portfolio_dict=portfolio_dict,
        benchmark_ticker=benchmark_ticker,
        display_names=display_names,
        sector_map=sector_map,
        risk_free_rate=rfr,
        sector_caps=sector_caps,
        sector_mins=sector_mins,
        bl_views=bl_views,
        bl_view_confidences=bl_view_confidences,
        bl_relative_views=bl_relative_views,
        bl_relative_view_confidences=bl_relative_view_confidences,
        rebalance_freq=rebalance_freq,
        tickers=tickers,
        friendly_tickers=friendly_tickers,
        friendly_benchmark=friendly_benchmark,
        friendly_sector_map=friendly_sector_map,
        user_friendly_weights=user_friendly_weights,
        price_data_full=price_data_full,
        price_data_train=price_data_train,
        price_data_test=price_data_test,
        mean_returns=mean_returns,
        cov_matrix=cov_matrix,
        log_returns=log_returns
    )
