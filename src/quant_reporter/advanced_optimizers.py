"""
Advanced portfolio optimization methods.

This module implements Risk Parity, Hierarchical Risk Parity (HRP),
Minimum Correlation, and Maximum Diversification portfolio optimization methods.

These methods extend the traditional Mean-Variance Optimization (MVO) framework
with alternative risk-based allocation strategies.
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def optimize_risk_parity(
    cov_matrix: pd.DataFrame,
    initial_weights: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> np.ndarray:
    """
    Compute Risk Parity portfolio weights.
    
    Risk Parity seeks to equalize the risk contribution of each asset.
    The risk contribution of asset i is: RC_i = w_i * (Σw)_i
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns (annualized)
    initial_weights : np.ndarray, optional
        Starting weights for optimization. Default: equal weights
    max_iter : int
        Maximum optimization iterations
    tol : float
        Convergence tolerance
        
    Returns
    -------
    np.ndarray
        Optimal weights that equalize risk contributions
        
    Raises
    ------
    ValueError
        If covariance matrix is not positive semi-definite
    RuntimeError
        If optimization fails to converge within max_iter
        
    References
    ----------
    Maillard, S., Roncalli, T., & Teïletche, J. (2010).
    The properties of equally weighted risk contribution portfolios.
    The Journal of Portfolio Management, 36(4), 60-70.
    """
    from .opt_core import validate_covariance_matrix, regularize_covariance
    
    # Validate and regularize covariance matrix
    validate_covariance_matrix(cov_matrix)
    cov_matrix_reg = regularize_covariance(cov_matrix)
    
    n_assets = len(cov_matrix_reg)
    cov_array = cov_matrix_reg.values
    
    # Initial weights: equal weight if not provided
    if initial_weights is None:
        initial_weights = np.ones(n_assets) / n_assets
    
    def risk_contribution(weights, cov):
        """Calculate risk contribution of each asset."""
        portfolio_vol = np.sqrt(weights @ cov @ weights)
        marginal_contrib = cov @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib
    
    def objective(weights, cov):
        """Objective: minimize variance of risk contributions."""
        rc = risk_contribution(weights, cov)
        target_rc = np.mean(rc)
        return np.sum((rc - target_rc) ** 2)
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        args=(cov_array,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    if not result.success:
        raise RuntimeError(
            f"Risk Parity optimization failed to converge. "
            f"Status: {result.message}. Iterations: {result.nit}"
        )
    
    weights = result.x
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Validate output
    if not np.all(weights >= -1e-6):  # Allow small numerical errors
        logger.warning("Some weights are slightly negative, clipping to zero")
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
    
    return weights


def optimize_hrp(
    cov_matrix: pd.DataFrame,
    returns: Optional[pd.Series] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute Hierarchical Risk Parity portfolio weights.
    
    HRP uses hierarchical clustering to group similar assets, then allocates
    weights recursively based on cluster variance.
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns (annualized)
    returns : pd.Series, optional
        Mean returns (used for dendrogram coloring only)
        
    Returns
    -------
    weights : np.ndarray
        Optimal HRP weights
    metadata : dict
        Contains 'linkage_matrix', 'ordered_indices', 'dendrogram_data'
        
    Raises
    ------
    ValueError
        If covariance matrix is not positive semi-definite
        
    References
    ----------
    Lopez de Prado, M. (2016).
    Building diversified portfolios that outperform out of sample.
    The Journal of Portfolio Management, 42(4), 59-69.
    """
    from .opt_core import validate_covariance_matrix
    
    # Validate covariance matrix
    validate_covariance_matrix(cov_matrix)
    
    n_assets = len(cov_matrix)
    cov_array = cov_matrix.values
    
    # Step 1: Convert covariance to correlation matrix
    std_devs = np.sqrt(np.diag(cov_array))
    corr_matrix = cov_array / np.outer(std_devs, std_devs)
    
    # Step 2: Compute distance matrix
    # Distance metric: d_ij = sqrt((1 - rho_ij) / 2)
    dist_matrix = np.sqrt((1 - corr_matrix) / 2)
    
    # Convert to condensed distance matrix for scipy
    dist_condensed = squareform(dist_matrix, checks=False)
    
    # Step 3: Perform hierarchical clustering
    linkage_matrix = linkage(dist_condensed, method='single')
    
    # Step 4: Get quasi-diagonal ordering from dendrogram
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    dend = scipy_dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = dend['leaves']
    
    # Step 5: Recursive bisection to allocate weights
    def _compute_cluster_variance(cov, items):
        """Compute variance of equally-weighted cluster."""
        if len(items) == 0:
            return 0.0
        cluster_cov = cov[np.ix_(items, items)]
        equal_weights = np.ones(len(items)) / len(items)
        return equal_weights @ cluster_cov @ equal_weights
    
    def _recursive_bisection(items):
        """Recursively allocate weights via bisection."""
        if len(items) == 1:
            return {items[0]: 1.0}
        
        # Split into two clusters
        mid = len(items) // 2
        left_items = items[:mid]
        right_items = items[mid:]
        
        # Compute cluster variances
        left_var = _compute_cluster_variance(cov_array, left_items)
        right_var = _compute_cluster_variance(cov_array, right_items)
        
        # Allocate weight inversely proportional to variance
        # alpha is the weight for the left cluster
        if left_var + right_var == 0:
            alpha = 0.5
        else:
            alpha = 1 - left_var / (left_var + right_var)
        
        # Recurse
        left_weights = _recursive_bisection(left_items)
        right_weights = _recursive_bisection(right_items)
        
        # Scale and combine
        result = {}
        for k, v in left_weights.items():
            result[k] = v * alpha
        for k, v in right_weights.items():
            result[k] = v * (1 - alpha)
        
        return result
    
    # Allocate weights using ordered indices
    weight_dict = _recursive_bisection(ordered_indices)
    
    # Convert to array in original order
    weights = np.zeros(n_assets)
    for idx, weight in weight_dict.items():
        weights[idx] = weight
    
    # Prepare metadata
    metadata = {
        'linkage_matrix': linkage_matrix,
        'ordered_indices': ordered_indices,
        'dendrogram_data': dend
    }
    
    return weights, metadata


def optimize_min_correlation(
    cov_matrix: pd.DataFrame,
    bounds: Optional[Tuple[float, float]] = (0, 1)
) -> np.ndarray:
    """
    Compute Minimum Correlation portfolio weights.
    
    Minimizes the average pairwise correlation of the portfolio.
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns (annualized)
    bounds : tuple, optional
        Weight bounds for each asset (min, max). Default: (0, 1)
        
    Returns
    -------
    np.ndarray
        Optimal weights minimizing average correlation
        
    Raises
    ------
    ValueError
        If covariance matrix is not positive semi-definite
        
    References
    ----------
    Christoffersen, P., Errunza, V., Jacobs, K., & Langlois, H. (2012).
    Is the potential for international diversification disappearing?
    Review of Financial Studies, 25(12), 3711-3751.
    """
    from .opt_core import validate_covariance_matrix, regularize_covariance
    
    # Validate and regularize covariance matrix
    validate_covariance_matrix(cov_matrix)
    cov_matrix_reg = regularize_covariance(cov_matrix)
    
    n_assets = len(cov_matrix_reg)
    cov_array = cov_matrix_reg.values
    
    # Convert to correlation matrix
    std_devs = np.sqrt(np.diag(cov_array))
    corr_matrix = cov_array / np.outer(std_devs, std_devs)
    
    def objective(weights):
        """Minimize average pairwise correlation."""
        # Portfolio correlation = w^T R w (normalized by portfolio variance)
        # We want to minimize the off-diagonal correlation terms
        portfolio_corr = weights @ corr_matrix @ weights
        # Subtract diagonal (self-correlation = 1) to get only pairwise correlations
        self_corr = np.sum(weights ** 2)
        avg_pairwise_corr = (portfolio_corr - self_corr) / (1 - self_corr + 1e-10)
        return avg_pairwise_corr
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds for each weight
    weight_bounds = tuple([bounds] * n_assets)
    
    # Initial weights: equal weight
    initial_weights = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        raise RuntimeError(
            f"Min Correlation optimization failed to converge. "
            f"Status: {result.message}"
        )
    
    weights = result.x
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Validate output
    if not np.all(weights >= -1e-6):
        logger.warning("Some weights are slightly negative, clipping to zero")
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
    
    return weights


def optimize_max_diversification(
    cov_matrix: pd.DataFrame,
    bounds: Optional[Tuple[float, float]] = (0, 1)
) -> np.ndarray:
    """
    Compute Maximum Diversification portfolio weights.
    
    Maximizes the diversification ratio: (weighted average volatility) / (portfolio volatility).
    
    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix of asset returns (annualized)
    bounds : tuple, optional
        Weight bounds for each asset (min, max). Default: (0, 1)
        
    Returns
    -------
    np.ndarray
        Optimal weights maximizing diversification ratio
        
    Raises
    ------
    ValueError
        If covariance matrix is not positive semi-definite
        
    References
    ----------
    Choueifaty, Y., & Coignard, Y. (2008).
    Toward maximum diversification.
    The Journal of Portfolio Management, 35(1), 40-51.
    """
    from .opt_core import validate_covariance_matrix, regularize_covariance
    
    # Validate and regularize covariance matrix
    validate_covariance_matrix(cov_matrix)
    cov_matrix_reg = regularize_covariance(cov_matrix)
    
    n_assets = len(cov_matrix_reg)
    cov_array = cov_matrix_reg.values
    
    # Asset volatilities
    volatilities = np.sqrt(np.diag(cov_array))
    
    def objective(weights):
        """Minimize negative diversification ratio."""
        # Diversification ratio = (w^T sigma) / sqrt(w^T Sigma w)
        weighted_vol = weights @ volatilities
        portfolio_vol = np.sqrt(weights @ cov_array @ weights)
        div_ratio = weighted_vol / (portfolio_vol + 1e-10)
        return -div_ratio  # Minimize negative = maximize positive
    
    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    
    # Bounds for each weight
    weight_bounds = tuple([bounds] * n_assets)
    
    # Initial weights: equal weight
    initial_weights = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    if not result.success:
        raise RuntimeError(
            f"Max Diversification optimization failed to converge. "
            f"Status: {result.message}"
        )
    
    weights = result.x
    
    # Normalize to ensure sum = 1.0
    weights = weights / np.sum(weights)
    
    # Validate output
    if not np.all(weights >= -1e-6):
        logger.warning("Some weights are slightly negative, clipping to zero")
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
    
    # Validate diversification ratio >= 1.0
    weighted_vol = weights @ volatilities
    portfolio_vol = np.sqrt(weights @ cov_array @ weights)
    div_ratio = weighted_vol / portfolio_vol
    
    if div_ratio < 0.99:  # Allow small numerical errors
        logger.warning(f"Diversification ratio {div_ratio:.3f} is less than 1.0")
    
    return weights
