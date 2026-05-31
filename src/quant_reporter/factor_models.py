"""
Factor models for portfolio analysis.

This module implements Fama-French factor models including data fetching,
regression analysis, factor attribution, and rolling exposure analysis.
"""

import logging
import numpy as np
import pandas as pd
import requests
import io
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import statsmodels.api as sm
from datetime import datetime

logger = logging.getLogger(__name__)


def _safe_infer_freq(index):
    """Return an inferred pandas frequency string, or None.

    pd.infer_freq raises TypeError on a non-DatetimeIndex (e.g. a default
    integer index) and ValueError on fewer than 3 points; callers treat a
    None result as 'assume daily' (252) for annualization.
    """
    if isinstance(index, pd.DatetimeIndex) and len(index) >= 3:
        try:
            return pd.infer_freq(index)
        except (TypeError, ValueError):
            return None
    return None

# Cache directory for factor data
_CACHE_DIR = Path.home() / ".quant_reporter" / "factor_cache"


def fetch_fama_french_factors(
    dataset: str = "F-F_Research_Data_Factors_daily",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache: bool = True
) -> pd.DataFrame:
    """
    Fetch Fama-French factor data from Kenneth French Data Library.

    Downloads factor data (Market, SMB, HML, Risk-Free Rate) from the Kenneth French
    data library. Supports daily and monthly frequencies.

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to fetch. Common options:
        - 'F-F_Research_Data_Factors_daily' (default) - Daily 3-factor
        - 'F-F_Research_Data_Factors' - Monthly 3-factor
        - 'F-F_Research_Data_5_Factors_2x3_daily' - Daily 5-factor
        - 'F-F_Research_Data_5_Factors_2x3' - Monthly 5-factor
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, returns all available data.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, returns data up to latest available.
    cache : bool, optional
        If True, caches downloaded data locally for faster subsequent access.
        Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Mkt-RF': Market excess return (market return minus risk-free rate)
        - 'SMB': Small Minus Big (size factor)
        - 'HML': High Minus Low (value factor)
        - 'RF': Risk-free rate
        Index is DatetimeIndex with the date of each observation.

    Raises
    ------
    ValueError
        If the dataset cannot be fetched or parsed.

    Examples
    --------
    >>> factors = fetch_fama_french_factors()
    >>> factors.head()
                Mkt-RF   SMB   HML    RF
    Date
    1926-07-01    0.10 -0.20 -0.30  0.01
    1926-07-02    0.25 -0.10  0.05  0.01

    Notes
    -----
    Data is sourced from Kenneth R. French's data library:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    """
    # Setup cache directory
    if cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{dataset}_v2.parquet"

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading {dataset} from cache: {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            df = _download_fama_french(dataset)
            df.to_parquet(cache_file)
            logger.info(f"Cached {dataset} to {cache_file}")
    else:
        df = _download_fama_french(dataset)

    # Filter by date range
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    return df


def _download_fama_french(dataset: str) -> pd.DataFrame:
    """
    Internal function to download and parse Fama-French data.

    Parameters
    ----------
    dataset : str
        Name of the dataset to fetch.

    Returns
    -------
    pd.DataFrame
        Parsed factor data.

    Raises
    ------
    ValueError
        If download fails or data cannot be parsed.
    """
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
    url = f"{base_url}/{dataset}_CSV.zip"

    logger.info(f"Fetching {dataset} from Kenneth French Data Library...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch factor data: {e}")

    # Parse CSV from zip
    import zipfile
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get the CSV file (usually only one file in the zip)
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in the downloaded archive")

            # Read CSV, skip header rows that contain metadata
            with z.open(csv_files[0]) as f:
                # Read as text
                content = f.read().decode('utf-8')
                lines = content.split('\n')

                # Find where the actual data starts (skip metadata)
                # Data starts after a blank line following the column headers
                data_start = 0
                for i, line in enumerate(lines):
                    if line.strip() == '' and i > 0:
                        data_start = i + 1
                        break

                # Read the data portion
                df = pd.read_csv(
                    io.StringIO('\n'.join(lines[data_start:])),
                    index_col=0,
                    parse_dates=False,  # We'll parse dates manually
                    na_values=['-99.99', '-999', ' NaN']
                )

                # Clean column names (remove spaces)
                df.columns = df.columns.str.strip()

                # Drop rows with NaN (often footer rows)
                df = df.dropna()

                # Convert to numeric (values are in percentages)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                df = df.dropna()

                # Convert from percentage points to decimals (e.g. 1.5 -> 0.015)
                df = df / 100.0

                # Parse index as dates (Fama-French uses YYYYMMDD format for daily, YYYYMM for monthly)
                raw_index = df.index.astype(str).str.strip()
                parsed_index = pd.to_datetime(raw_index, format='%Y%m%d', errors='coerce')
                # If that fails, try monthly format
                if parsed_index.isna().all():
                    parsed_index = pd.to_datetime(raw_index + '01', format='%Y%m%d', errors='coerce')
                
                df.index = parsed_index

                # Drop any rows where date parsing failed
                df = df[df.index.notna()]

                return df

    except Exception as e:
        raise ValueError(f"Failed to parse factor data: {e}")


def _prepare_excess_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.02,
):
    """Shared regression prep: align dates, compute EXCESS portfolio returns, and
    build the (constant-augmented) factor design matrix.

    This is the single place that defines the excess-return convention used by BOTH
    the static (`run_factor_regression`) and rolling (`run_rolling_factor_regression`)
    paths, so they share one OLS engine instead of diverging (raw vs excess).

    Returns
    -------
    (y, X, factor_cols)
        y : pd.Series of excess portfolio returns aligned to the factors.
        X : pd.DataFrame of factors with an added 'const' column (statsmodels-ready).
        factor_cols : list[str] of factor column names used (excludes 'const').
    """
    common_dates = portfolio_returns.index.intersection(factor_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between portfolio returns and factor returns")

    portfolio_returns = portfolio_returns.loc[common_dates]
    factor_returns = factor_returns.loc[common_dates]

    # Excess returns (subtract the risk-free / RF factor). Threaded rfr is used only
    # when the factor frame lacks an 'RF' column.
    if 'RF' in factor_returns.columns:
        risk_free = factor_returns['RF']
    else:
        risk_free = risk_free_rate / 252  # Daily rate

    portfolio_excess = portfolio_returns - risk_free

    # Factor set matches run_factor_regression's historical behavior (3-factor core).
    # RMW/CMA from the 5-factor dataset are still surfaced separately by
    # identify_factor_regimes; keeping the regression on the same 3 factors keeps the
    # static and rolling engines identical and preserves existing regression numbers.
    factor_cols = [col for col in ['Mkt-RF', 'SMB', 'HML'] if col in factor_returns.columns]
    if 'Mkt-RF' not in factor_cols:
        raise ValueError("Factor data must contain 'Mkt-RF' column")

    X = sm.add_constant(factor_returns[factor_cols])
    y = portfolio_excess
    return y, X, factor_cols


def run_factor_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> Dict[str, Any]:
    """
    Perform factor regression using OLS.

    Regresses portfolio excess returns against factor returns to estimate
    factor loadings (betas) and alpha.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Time series of portfolio returns. Index should be DatetimeIndex.
    factor_returns : pd.DataFrame
        Factor returns data. Should contain columns 'Mkt-RF', 'SMB', 'HML', and 'RF'.
        Output from `fetch_fama_french_factors()`.
    risk_free_rate : float, optional
        Annual risk-free rate to use if 'RF' column is not in factor_returns.
        Default is 0.02 (2%). Only used if risk-free rate not in factor data.

    Returns
    -------
    dict
        Dictionary containing:
        - 'alpha': Annualized alpha (intercept from regression)
        - 'betas': pd.Series of factor loadings (betas)
        - 'r_squared': R-squared of the regression
        - 'residuals': Residuals from the regression
        - 'factor_pvalues': p-values for each factor coefficient
        - 'summary': Full regression summary from statsmodels

    Examples
    --------
    >>> factors = fetch_fama_french_factors()
    >>> results = run_factor_regression(portfolio_returns, factors)
    >>> print(f"Alpha: {results['alpha']:.2%}")
    >>> print(f"Market Beta: {results['betas']['Mkt-RF']:.3f}")

    Notes
    -----
    The regression model is:
    (Rp - Rf) = α + β_MKT(Mkt-RF) + β_SMB(SMB) + β_HML(HML) + ε

    Alpha is annualized assuming daily data (multiplied by 252).
    For monthly data, adjust by multiplying by 12.
    """
    # Shared prep: align dates, compute EXCESS returns, build factor design matrix.
    # This is the single excess-return OLS engine used by both static and rolling paths.
    y, X, factor_cols = _prepare_excess_regression(
        portfolio_returns, factor_returns, risk_free_rate
    )

    # Run OLS regression
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()

    # Extract results
    alpha_daily = results.params['const']

    # Annualize alpha (assuming daily data)
    # Detect frequency
    freq = _safe_infer_freq(factor_returns.index)
    if freq and 'M' in freq:
        alpha_annual = alpha_daily * 12  # Monthly
    else:
        alpha_annual = alpha_daily * 252  # Daily

    # Extract betas (exclude intercept)
    betas = results.params.drop('const')
    betas.name = 'Factor Loadings'

    # Factor p-values
    pvalues = results.pvalues.drop('const')
    pvalues.name = 'p-values'

    return {
        'alpha': alpha_annual,
        'betas': betas,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj,
        'residuals': results.resid,
        'factor_pvalues': pvalues,
        'summary': results.summary()
    }


def run_rolling_factor_regression(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window_days: int = 126,
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Compute rolling factor betas over a trailing window using the SAME excess-return
    OLS engine as `run_factor_regression`.

    Previously this lived in factor_report.py and regressed RAW (non-excess) returns with
    a hand-rolled (XᵀX)⁻¹XᵀY, diverging from the static path. It now reuses
    `_prepare_excess_regression` (excess returns, rfr threaded) and statsmodels OLS so
    there is one engine. The alpha (intercept) is estimated within each window and dropped
    from the returned exposures (this function tracks factor betas over time).

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio daily returns (DatetimeIndex).
    factor_returns : pd.DataFrame
        Fama-French factors including 'RF' (output of fetch_fama_french_factors).
    window_days : int, optional
        Rolling window length in observations. Default 126 (~6 months).
    risk_free_rate : float, optional
        Annual risk-free rate, used only if factor_returns lacks an 'RF' column.

    Returns
    -------
    pd.DataFrame
        Rolling factor betas indexed by window-end date (Alpha intercept dropped).
        Empty DataFrame if there is insufficient overlapping data.
    """
    try:
        y, X, factor_cols = _prepare_excess_regression(
            portfolio_returns, factor_returns, risk_free_rate
        )
    except ValueError:
        return pd.DataFrame()

    aligned = pd.concat([y.rename("__excess__"), X], axis=1).dropna()
    if len(aligned) < window_days + 10:
        return pd.DataFrame()  # Not enough data

    y_arr = aligned["__excess__"].values
    X_arr = aligned[X.columns].values  # already includes the 'const' column

    betas = []
    dates = []
    for i in range(window_days, len(y_arr)):
        y_win = y_arr[i - window_days:i]
        X_win = X_arr[i - window_days:i]
        try:
            beta = np.linalg.solve(X_win.T @ X_win, X_win.T @ y_win)
            betas.append(beta)
            dates.append(aligned.index[i])
        except np.linalg.LinAlgError:
            pass  # Skip singular windows

    if not betas:
        return pd.DataFrame()

    col_names = list(X.columns)  # ['const', 'Mkt-RF', ...]
    rolling_betas = pd.DataFrame(betas, index=dates, columns=col_names)

    # Drop the intercept ('const'); this function tracks factor exposures, not alpha.
    return rolling_betas.drop(columns=['const'])


def compute_factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    betas: pd.Series,
    alpha: float,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Decompose portfolio returns by factor contributions.

    Attributed returns show how much of the portfolio's performance can be
    attributed to each factor exposure versus idiosyncratic alpha.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Time series of portfolio returns.
    factor_returns : pd.DataFrame
        Factor returns data from `fetch_fama_french_factors()`.
    betas : pd.Series
        Factor loadings from `run_factor_regression()`.
    alpha : float
        Annualized alpha from factor regression.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'Total_Return': Actual portfolio return
        - 'Market_Contribution': Contribution from market factor
        - 'SMB_Contribution': Contribution from size factor (if present)
        - 'HML_Contribution': Contribution from value factor (if present)
        - 'Alpha_Contribution': Contribution from idiosyncratic alpha
        - 'Unexplained': Residual return not explained by factors
        Index matches the input returns index.

    Examples
    --------
    >>> factors = fetch_fama_french_factors()
    >>> results = run_factor_regression(portfolio_returns, factors)
    >>> attribution = compute_factor_attribution(
    ...     portfolio_returns, factors, results['betas'], results['alpha']
    ... )
    >>> attribution.head()

    Notes
    -----
    Factor contributions are calculated as:
    Contribution_i = β_i × Factor_Return_i

    For the period. Alpha is annualized and divided by trading days.
    """
    # Align dates
    common_dates = portfolio_returns.index.intersection(factor_returns.index)
    portfolio_returns = portfolio_returns.loc[common_dates]
    factor_returns = factor_returns.loc[common_dates]

    # Initialize attribution DataFrame
    attribution = pd.DataFrame(index=common_dates)
    attribution['Total_Return'] = portfolio_returns

    # Calculate factor contributions
    contribution_data = {}

    for factor_name in betas.index:
        if factor_name in factor_returns.columns:
            contribution_col = f"{factor_name}_Contribution"
            contribution_data[contribution_col] = betas[factor_name] * factor_returns[factor_name]
            attribution[contribution_col] = contribution_data[contribution_col]

    # Add alpha contribution (de-annualize based on frequency)
    freq = _safe_infer_freq(factor_returns.index)
    if freq and 'M' in freq:
        alpha_contribution = alpha / 12  # Monthly
    else:
        alpha_contribution = alpha / 252  # Daily

    attribution['Alpha_Contribution'] = alpha_contribution

    # Add risk-free contribution
    if 'RF' in factor_returns.columns:
        attribution['RiskFree_Contribution'] = factor_returns['RF']
    else:
        freq = _safe_infer_freq(factor_returns.index)
        if freq and 'M' in freq:
            attribution['RiskFree_Contribution'] = risk_free_rate / 12  # Monthly
        else:
            attribution['RiskFree_Contribution'] = risk_free_rate / 252  # Daily

    # Calculate explained vs unexplained
    explained_cols = [col for col in attribution.columns if 'Contribution' in col]
    attribution['Explained'] = attribution[explained_cols].sum(axis=1)
    attribution['Unexplained'] = attribution['Total_Return'] - attribution['Explained']

    return attribution
