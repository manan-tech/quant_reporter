import logging
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

from .data import get_data
from .opt_core import get_risk_free_rate, get_optimization_inputs, DEFAULT_RISK_FREE_RATE
from .analytics import PortfolioAnalytics
from .providers import DataProvider, get_default_provider

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

    # Memoized analytics accessor (set in build_context)
    analytics: object = None

    # Covariance treatment used for the optimization inputs (so downstream
    # consumers like walk-forward validation can match it).
    denoise_cov: bool = False
    n_components: int = 3

    # The data backend this context was built with (yfinance by default).
    # Downstream consumers that need extra fetches (market caps, fundamentals)
    # reuse it so the whole report draws from one source.
    data_provider: object = None

def _resolve_dates_and_rfr(train_start: str, train_end: str,
                           risk_free_rate: Union[float, str],
                           provider: Optional[DataProvider] = None):
    """Shared step 1+2: date resolution and risk-free-rate resolution."""
    test_start_dt = pd.to_datetime(train_end) + timedelta(days=1)
    test_end_dt = datetime.now() - timedelta(days=1)
    test_start = test_start_dt.strftime('%Y-%m-%d')
    test_end = test_end_dt.strftime('%Y-%m-%d')
    full_start = train_start
    full_end = test_end

    if isinstance(risk_free_rate, str) and risk_free_rate.lower() == 'auto':
        rfr = get_risk_free_rate(provider=provider)
    elif isinstance(risk_free_rate, (int, float)):
        rfr = float(risk_free_rate)
    else:
        rfr = DEFAULT_RISK_FREE_RATE

    return test_start, test_end, full_start, full_end, rfr


def _resolve_friendly_names(portfolio_dict: Dict[str, float], benchmark_ticker: str,
                            display_names: Optional[Dict[str, str]],
                            sector_map: Optional[Dict[str, str]]):
    """Shared step 3: build friendly name mappings."""
    tickers = list(portfolio_dict.keys())

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
        user_friendly_weights = dict(portfolio_dict)

    return tickers, friendly_tickers, friendly_benchmark, friendly_sector_map, user_friendly_weights


def _assemble_context(price_data_full: pd.DataFrame,
                      portfolio_dict: Dict[str, float],
                      benchmark_ticker: str,
                      friendly_tickers: List[str],
                      friendly_benchmark: str,
                      friendly_sector_map: Optional[Dict[str, str]],
                      user_friendly_weights: Dict[str, float],
                      tickers: List[str],
                      train_start: str,
                      train_end: str,
                      test_start: str,
                      test_end: str,
                      full_start: str,
                      full_end: str,
                      rfr: float,
                      display_names: Optional[Dict[str, str]],
                      sector_map: Optional[Dict[str, str]],
                      sector_caps: Optional[Dict[str, float]],
                      sector_mins: Optional[Dict[str, float]],
                      bl_views: Optional[Dict[str, float]],
                      bl_view_confidences: Optional[Dict[str, float]],
                      bl_relative_views: Optional[List[Dict]],
                      bl_relative_view_confidences: Optional[List[float]],
                      rebalance_freq: Optional[str],
                      denoise_cov: bool,
                      n_components: int,
                      data_provider: Optional[DataProvider] = None) -> ReportContext:
    """
    Post-fetch assembly: rename columns, drop missing tickers, renormalize weights,
    split train/test, compute optimization inputs, construct and return ReportContext.

    Both build_context and build_context_from_prices share this logic.
    """
    # Work on a copy so we don't mutate the caller's data
    price_data_full = price_data_full.copy()

    # Apply display names to columns (only needed when display_names were provided)
    if display_names:
        price_data_full.rename(columns=display_names, inplace=True)

    # Standardize column order (assets first, then benchmark) and drop missing.
    # Append the benchmark only if it isn't already a holding — otherwise its
    # column is duplicated (e.g. a 60/40 of SPY+AGG benchmarked against SPY),
    # which inflates the asset count and breaks every weights-vs-assets op.
    ordered_cols = friendly_tickers + ([friendly_benchmark] if friendly_benchmark not in friendly_tickers else [])
    missing_cols = [c for c in ordered_cols if c not in price_data_full.columns]
    if missing_cols:
        logger.warning(f"Warning: these tickers returned no data and will be dropped: {missing_cols}")
        friendly_tickers = [t for t in friendly_tickers if t not in missing_cols]
        if friendly_benchmark in missing_cols:
            raise ValueError(f"Benchmark ticker {friendly_benchmark} failed to download. Cannot proceed.")
        ordered_cols = friendly_tickers + ([friendly_benchmark] if friendly_benchmark not in friendly_tickers else [])

        # Update weights to handle dropped assets
        valid_tickers_original = (
            [t for t in tickers if display_names.get(t, t) in friendly_tickers]
            if display_names else list(friendly_tickers)
        )
        total_valid_weight = sum([portfolio_dict[t] for t in valid_tickers_original])
        for ticker in list(user_friendly_weights.keys()):
            if ticker not in friendly_tickers:
                del user_friendly_weights[ticker]
        for ticker in user_friendly_weights:
            user_friendly_weights[ticker] /= total_valid_weight

        tickers = valid_tickers_original

    price_data_full = price_data_full[ordered_cols]

    # 5. Split periods
    train_mask = (
        (price_data_full.index >= pd.to_datetime(train_start)) &
        (price_data_full.index <= pd.to_datetime(train_end))
    )
    price_data_train = price_data_full.loc[train_mask]

    test_mask = (
        (price_data_full.index >= pd.to_datetime(test_start)) &
        (price_data_full.index <= pd.to_datetime(test_end))
    )
    price_data_test = price_data_full.loc[test_mask]

    # 6. Calculate Train Optimization Inputs
    asset_data_train = price_data_train[friendly_tickers]
    mean_returns, cov_matrix, log_returns = get_optimization_inputs(
        asset_data_train, denoise_cov=denoise_cov, n_components=n_components
    )

    ctx = ReportContext(
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
        log_returns=log_returns,
        denoise_cov=denoise_cov,
        n_components=n_components,
        data_provider=data_provider,
    )
    ctx.analytics = PortfolioAnalytics(ctx)
    return ctx


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
                  denoise_cov: bool = False, n_components: int = 3,
                  data_provider: Optional[DataProvider] = None, **kwargs) -> ReportContext:
    """
    Factory function to fetch data and construct the ReportContext, removing duplication
    across all individual report generators.

    Pass ``data_provider`` to fetch from a custom backend (Bloomberg, Refinitiv,
    a local CSV, a test fixture) instead of the default yfinance scraper. The
    provider must satisfy the :class:`~quant_reporter.providers.DataProvider`
    protocol. Defaults to the globally configured provider.
    """
    provider = data_provider if data_provider is not None else get_default_provider()

    # 1+2. Date and risk-free-rate resolution
    test_start, test_end, full_start, full_end, rfr = _resolve_dates_and_rfr(
        train_start, train_end, risk_free_rate, provider=provider
    )

    # 3. Ticker/display-name mapping
    tickers, friendly_tickers, friendly_benchmark, friendly_sector_map, user_friendly_weights = (
        _resolve_friendly_names(portfolio_dict, benchmark_ticker, display_names, sector_map)
    )

    # 4. Fetch Data Once
    all_tickers = tickers + [benchmark_ticker]
    price_data_full = get_data(all_tickers, full_start, full_end, provider=provider)
    if price_data_full is None or price_data_full.empty:
        raise ValueError("Failed to fetch price data.")

    return _assemble_context(
        price_data_full=price_data_full,
        portfolio_dict=portfolio_dict,
        benchmark_ticker=benchmark_ticker,
        friendly_tickers=friendly_tickers,
        friendly_benchmark=friendly_benchmark,
        friendly_sector_map=friendly_sector_map,
        user_friendly_weights=user_friendly_weights,
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        full_start=full_start,
        full_end=full_end,
        rfr=rfr,
        display_names=display_names,
        sector_map=sector_map,
        sector_caps=sector_caps,
        sector_mins=sector_mins,
        bl_views=bl_views,
        bl_view_confidences=bl_view_confidences,
        bl_relative_views=bl_relative_views,
        bl_relative_view_confidences=bl_relative_view_confidences,
        rebalance_freq=rebalance_freq,
        denoise_cov=denoise_cov,
        n_components=n_components,
        data_provider=provider,
    )


def build_context_from_prices(price_data_full: pd.DataFrame,
                              portfolio_dict: Dict[str, float],
                              benchmark_ticker: str,
                              train_start: str,
                              train_end: str,
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
                              denoise_cov: bool = False,
                              n_components: int = 3,
                              data_provider: Optional[DataProvider] = None,
                              **kwargs) -> ReportContext:
    """Build a ReportContext from an already-fetched price DataFrame (no network).

    price_data_full must contain every portfolio ticker column + the benchmark column.
    Accepts the same optional keyword arguments as build_context.

    No prices are fetched. ``data_provider`` is still recorded on the context (and
    used only if ``risk_free_rate="auto"`` requires a rate lookup) so downstream
    consumers reuse the same backend; pass a numeric ``risk_free_rate`` to stay
    fully offline.
    """
    provider = data_provider if data_provider is not None else get_default_provider()

    # 1+2. Date and risk-free-rate resolution (identical to build_context)
    test_start, test_end, full_start, full_end, rfr = _resolve_dates_and_rfr(
        train_start, train_end, risk_free_rate, provider=provider
    )

    # 3. Ticker/display-name mapping
    tickers, friendly_tickers, friendly_benchmark, friendly_sector_map, user_friendly_weights = (
        _resolve_friendly_names(portfolio_dict, benchmark_ticker, display_names, sector_map)
    )

    # No network fetch — use supplied data directly
    return _assemble_context(
        price_data_full=price_data_full,
        portfolio_dict=portfolio_dict,
        benchmark_ticker=benchmark_ticker,
        friendly_tickers=friendly_tickers,
        friendly_benchmark=friendly_benchmark,
        friendly_sector_map=friendly_sector_map,
        user_friendly_weights=user_friendly_weights,
        tickers=tickers,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        full_start=full_start,
        full_end=full_end,
        rfr=rfr,
        display_names=display_names,
        sector_map=sector_map,
        sector_caps=sector_caps,
        sector_mins=sector_mins,
        bl_views=bl_views,
        bl_view_confidences=bl_view_confidences,
        bl_relative_views=bl_relative_views,
        bl_relative_view_confidences=bl_relative_view_confidences,
        rebalance_freq=rebalance_freq,
        denoise_cov=denoise_cov,
        n_components=n_components,
        data_provider=provider,
    )
