import logging
import pandas as pd
import numpy as np
import scipy.optimize as sco
import yfinance as yf

logger = logging.getLogger(__name__)
from .data import get_data
from .metrics import calculate_metrics

# --- 1. Core Portfolio Math ---

def get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Helper function to calculate portfolio return, volatility, and sharpe.
    """
    weights = np.array(weights)
    port_return = np.sum(mean_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0.00001 else 0
    return port_return, port_vol, sharpe

def get_portfolio_price(ticker_prices, weights_dict):
    """
    Calculates the normalized 'Growth of $1' for a portfolio.
    """
    # Ensure weights_dict keys match ticker_prices columns
    weights = pd.Series(weights_dict).reindex(ticker_prices.columns).fillna(0)
    
    normalized_prices = ticker_prices / ticker_prices.iloc[0]
    weighted_prices = normalized_prices * weights
    return weighted_prices.sum(axis=1)

def denoise_covariance_matrix(cov_matrix, n_components=3):
    """
    Denoises a covariance matrix using Spectral Thresholding (Eigenvalue Clipping)
    and rescales to preserve the original matrix trace.
    Returns the denoised, positive-semidefinite DataFrame.
    """
    import numpy as np
    import pandas as pd
    
    # 1. Calculate eigenvalues and eigenvectors
    evals, evecs = np.linalg.eigh(cov_matrix)
    
    # 2. Sort evals descending, but np.linalg.eigh returns ascending, so we reverse
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    
    # 3. Clip the noise tail
    if n_components < len(evals):
        # Calculate the average of the discarded "noise" eigenvalues
        noise_variance = np.mean(evals[n_components:])
        
        # Replace the tail with the noise average
        evals[n_components:] = noise_variance
    
    # 4. Reconstruct the matrix: C' = V * L * V^T
    denoised_cov_matrix = evecs @ np.diag(evals) @ evecs.T
    
    # 5. Restore the DataFrame structure
    return pd.DataFrame(denoised_cov_matrix, index=cov_matrix.index, columns=cov_matrix.columns)

def get_optimization_inputs(asset_price_data, denoise_cov=False, n_components=3):
    """
    Calculates mean returns, covariance matrix, and log returns from asset price data.
    """
    log_returns = np.log(asset_price_data / asset_price_data.shift(1)).dropna()
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252
    
    if denoise_cov and n_components is not None:
        logger.info("Denoising covariance matrix (Eigenvalue Clipping with %d components)", n_components)
        cov_matrix = denoise_covariance_matrix(cov_matrix, n_components)
    
    return mean_returns, cov_matrix, log_returns

# --- 2. Optimization Objectives ---

def objective_neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Objective function: Minimize the negative Sharpe Ratio."""
    _, _, sharpe = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe

def objective_min_variance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """Objective function: Minimize the variance."""
    _, port_vol, _ = get_portfolio_stats(weights, mean_returns, cov_matrix, risk_free_rate)
    return port_vol

# --- 3. Optimization Engine ---

def find_optimal_portfolio(objective_func, mean_returns, cov_matrix, bounds, constraints, risk_free_rate=0.02):
    """
    Runs the SciPy optimization.
    """
    num_assets = len(mean_returns)
    initial_weights = np.array([1./num_assets] * num_assets)

    result = sco.minimize(
        objective_func,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result.success:
        logger.warning("Optimization warning: %s", result.message)
    return result.x

def calculate_efficient_frontier_curve(mean_returns, cov_matrix):
    """
    Calculates the smooth efficient frontier curve.
    """
    logger.info("Calculating Efficient Frontier curve...")
    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))
    base_constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    min_ret = mean_returns.quantile(0.1)
    max_ret = mean_returns.quantile(0.9) * 1.2
    target_returns = np.linspace(min_ret, max_ret, 50)
    frontier_vols = []

    for target_ret in target_returns:
        target_ret_constraint = ({'type': 'eq', 'fun': lambda w: get_portfolio_stats(w, mean_returns, cov_matrix)[0] - target_ret})
        all_constraints = (base_constraints, target_ret_constraint)
        
        result = sco.minimize(
            objective_min_variance,
            np.array([1./num_assets] * num_assets),
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=all_constraints
        )
        if result.success:
            frontier_vols.append(result.fun)
        else:
            frontier_vols.append(np.nan)

    return pd.DataFrame({'Return': target_returns, 'Volatility': frontier_vols}).dropna()

# --- 4. Sector Constraints Logic ---

def build_constraints(num_assets, raw_tickers, sector_map=None, sector_caps=None, sector_mins=None):
    """
    Builds a list of constraints for the optimizer, including new sector caps and mins.
    
    Args:
        num_assets (int): Number of assets.
        raw_tickers (list): The list of *original* tickers (e.g., ['AAPL', 'NEE']).
        sector_map (dict): Map of *raw tickers* to sectors (e.g., {'AAPL': 'Tech'}).
        sector_caps (dict): Map of sectors to max weights (e.g., {'Tech': 0.4}).
        sector_mins (dict): Map of sectors to min weights (e.g., {'Tech': 0.05}).
    """
    # Base constraint: weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    if sector_map and sector_caps:
        logger.info("Adding sector cap constraints...")
        for sector, cap in sector_caps.items():
            sector_indices = [i for i, ticker in enumerate(raw_tickers) if sector_map.get(ticker) == sector]
            
            if sector_indices:
                logger.info("   ...adding cap for %s (%.1f%%)", sector, cap*100)
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda weights, indices=sector_indices, cap=cap: cap - np.sum(weights[indices])
                })
    
    if sector_map and sector_mins:
        logger.info("Adding sector minimum constraints...")
        for sector, min_val in sector_mins.items():
            sector_indices = [i for i, ticker in enumerate(raw_tickers) if sector_map.get(ticker) == sector]
            
            if sector_indices:
                logger.info("   ...adding min for %s (%.1f%%)", sector, min_val*100)
                constraints.append({
                    'type': 'ineq',
                    # Constraint function: sum(weights_for_this_sector) - min_val >= 0
                    'fun': lambda weights, indices=sector_indices, min_val=min_val: np.sum(weights[indices]) - min_val
                })
                
    return tuple(constraints)

# --- 5. Helper Functions ---

def get_risk_free_rate():
    """
    Fetches the latest 13-week US T-bill rate (^IRX) as a decimal.
    """
    try:
        logger.info("Fetching live risk-free rate (^IRX)...")
        tbill = yf.download("^IRX", period="5d") 
        
        if tbill is None or tbill.empty:
            raise Exception("^IRX download failed or returned no data")
        
        latest_rate = tbill['Close'].iloc[-1] / 100 
        
        if not 0 <= latest_rate <= 0.2:
             raise Exception(f"Fetched rate ({latest_rate}) is unrealistic.")
             
        logger.info("Using live risk-free rate: %.2f%%", latest_rate * 100)
        return latest_rate
    except Exception as e:
        logger.warning("Could not fetch live risk-free rate. Defaulting to 0.06. Error: %s", e)
        return 0.06

def calculate_rolling_returns(cumulative_df):
    """
    Calculates 1, 3, and 5-year annualized rolling returns.
    """
    periods = {'1-Year': 252, '3-Year': 252*3, '5-Year': 252*5}
    rolling_returns = {}
    
    # Get the first column (usually the benchmark or first portfolio)
    if isinstance(cumulative_df, pd.DataFrame):
        first_col = cumulative_df.iloc[:, 0]
    else:
        first_col = cumulative_df
    
    for name, days in periods.items():
        if len(first_col) > days:
            years = days / 252
            rolling_returns[name] = {'Return': (first_col.iloc[-1] / first_col.iloc[-days-1])**(1/years) - 1}
        else:
            rolling_returns[name] = {'Return': np.nan}
            
    return pd.DataFrame.from_dict(rolling_returns, orient='index').map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")


# --- 6. Covariance Matrix Validation and Regularization ---

def validate_covariance_matrix(cov_matrix: pd.DataFrame) -> None:
    """
    Validate that a covariance matrix is suitable for optimization.
    
    Checks:
    - No NaN values
    - Symmetric matrix
    - Positive semi-definite (all eigenvalues >= 0)
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix to validate
        
    Raises
    ------
    ValueError
        If validation fails with diagnostic information
    """
    # Check for NaN values
    if cov_matrix.isnull().any().any():
        raise ValueError(
            "Covariance matrix contains NaN values. "
            f"Columns with NaN: {cov_matrix.columns[cov_matrix.isnull().any()].tolist()}"
        )
    
    # Check symmetry
    if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-8):
        max_asymmetry = np.abs(cov_matrix - cov_matrix.T).max().max()
        raise ValueError(
            f"Covariance matrix is not symmetric. "
            f"Maximum asymmetry: {max_asymmetry:.2e}"
        )
    
    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
    min_eigenvalue = np.min(eigenvalues)
    
    if min_eigenvalue < -1e-8:  # Allow small numerical errors
        raise ValueError(
            f"Covariance matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {min_eigenvalue:.2e}. "
            f"Consider using regularization."
        )
    
    logger.debug("Covariance matrix validation passed")


def regularize_covariance(cov_matrix: pd.DataFrame, threshold: float = 1e10) -> pd.DataFrame:
    """
    Regularize a covariance matrix to improve numerical stability.
    
    If the condition number exceeds the threshold, adds a small diagonal term
    to ensure positive definiteness and reduce condition number.
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix to regularize
    threshold : float
        Condition number threshold above which regularization is applied
        
    Returns
    -------
    pd.DataFrame
        Regularized covariance matrix
    """
    cov_array = cov_matrix.values
    
    # Compute condition number
    eigenvalues = np.linalg.eigvalsh(cov_array)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Filter out near-zero eigenvalues
    
    if len(eigenvalues) == 0:
        raise ValueError("Covariance matrix has no positive eigenvalues")
    
    condition_number = np.max(eigenvalues) / np.min(eigenvalues)
    
    if condition_number > threshold:
        logger.info(
            f"Covariance matrix has high condition number ({condition_number:.2e}). "
            f"Applying regularization..."
        )
        
        # Add small diagonal term (1e-8 times trace)
        trace = np.trace(cov_array)
        regularization_term = 1e-8 * trace / len(cov_array)
        cov_array_reg = cov_array + np.eye(len(cov_array)) * regularization_term
        
        # Verify regularization worked
        eigenvalues_reg = np.linalg.eigvalsh(cov_array_reg)
        eigenvalues_reg = eigenvalues_reg[eigenvalues_reg > 0]
        condition_number_reg = np.max(eigenvalues_reg) / np.min(eigenvalues_reg)
        
        logger.info(f"Regularized condition number: {condition_number_reg:.2e}")
        
        return pd.DataFrame(cov_array_reg, index=cov_matrix.index, columns=cov_matrix.columns)
    
    return cov_matrix
