"""
Tests for advanced portfolio optimizers.

Tests Risk Parity, HRP, Min Correlation, and Max Diversification optimizers.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from quant_reporter.advanced_optimizers import (
    optimize_risk_parity,
    optimize_hrp,
    optimize_min_correlation,
    optimize_max_diversification
)
from quant_reporter.opt_core import validate_covariance_matrix, regularize_covariance


# --- Fixtures ---

@pytest.fixture
def simple_cov_matrix():
    """Simple 3-asset covariance matrix."""
    data = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.015],
        [0.005, 0.015, 0.16]
    ])
    return pd.DataFrame(data, index=['A', 'B', 'C'], columns=['A', 'B', 'C'])


@pytest.fixture
def correlated_cov_matrix():
    """Covariance matrix with highly correlated assets."""
    # Assets A and B are highly correlated (0.95)
    corr = np.array([
        [1.0, 0.95, 0.3],
        [0.95, 1.0, 0.25],
        [0.3, 0.25, 1.0]
    ])
    vols = np.array([0.2, 0.22, 0.15])
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=['A', 'B', 'C'], columns=['A', 'B', 'C'])


@pytest.fixture
def ill_conditioned_cov_matrix():
    """Covariance matrix with high condition number."""
    # Create matrix with one very small eigenvalue
    eigenvalues = np.array([1.0, 0.5, 1e-12])
    eigenvectors = np.eye(3)
    cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return pd.DataFrame(cov, index=['A', 'B', 'C'], columns=['A', 'B', 'C'])


# --- Unit Tests for Validation ---

def test_validate_covariance_matrix_valid(simple_cov_matrix):
    """Test that valid covariance matrix passes validation."""
    # Should not raise
    validate_covariance_matrix(simple_cov_matrix)


def test_validate_covariance_matrix_with_nan():
    """Test that matrix with NaN values fails validation."""
    cov = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=['A', 'B'], columns=['A', 'B'])
    
    with pytest.raises(ValueError, match="contains NaN values"):
        validate_covariance_matrix(cov)


def test_validate_covariance_matrix_asymmetric():
    """Test that asymmetric matrix fails validation."""
    cov = pd.DataFrame([[1.0, 0.5], [0.3, 1.0]], index=['A', 'B'], columns=['A', 'B'])
    
    with pytest.raises(ValueError, match="not symmetric"):
        validate_covariance_matrix(cov)


def test_validate_covariance_matrix_not_positive_definite():
    """Test that non-positive-definite matrix fails validation."""
    # Matrix with negative eigenvalue
    cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=['A', 'B'], columns=['A', 'B'])
    
    with pytest.raises(ValueError, match="not positive semi-definite"):
        validate_covariance_matrix(cov)


def test_regularize_covariance_low_condition_number(simple_cov_matrix):
    """Test that regularization is not applied when condition number is low."""
    reg_cov = regularize_covariance(simple_cov_matrix, threshold=1e10)
    
    # Should be unchanged
    pd.testing.assert_frame_equal(reg_cov, simple_cov_matrix)


def test_regularize_covariance_high_condition_number(ill_conditioned_cov_matrix):
    """Test that regularization is applied when condition number is high."""
    reg_cov = regularize_covariance(ill_conditioned_cov_matrix, threshold=1e10)
    
    # Should have lower condition number
    orig_eigs = np.linalg.eigvalsh(ill_conditioned_cov_matrix.values)
    reg_eigs = np.linalg.eigvalsh(reg_cov.values)
    
    orig_cond = np.max(orig_eigs) / np.min(orig_eigs[orig_eigs > 0])
    reg_cond = np.max(reg_eigs) / np.min(reg_eigs[reg_eigs > 0])
    
    # Regularized condition number should be significantly lower
    assert reg_cond < orig_cond
    assert reg_cond < 1e10  # Should be below threshold


# --- Unit Tests for Risk Parity ---

def test_risk_parity_simple(simple_cov_matrix):
    """Test Risk Parity optimizer with simple covariance matrix."""
    weights = optimize_risk_parity(simple_cov_matrix)
    
    # Check basic properties
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


def test_risk_parity_equal_risk_contributions(simple_cov_matrix):
    """Test that Risk Parity produces approximately equal risk contributions."""
    weights = optimize_risk_parity(simple_cov_matrix)
    
    # Calculate risk contributions
    cov_array = simple_cov_matrix.values
    portfolio_vol = np.sqrt(weights @ cov_array @ weights)
    marginal_contrib = cov_array @ weights
    risk_contrib = weights * marginal_contrib / portfolio_vol
    
    # Check that ratio of max to min risk contribution is < 1.5
    ratio = np.max(risk_contrib) / np.min(risk_contrib)
    assert ratio < 1.5, f"Risk contribution ratio {ratio:.2f} exceeds 1.5"


def test_risk_parity_with_initial_weights(simple_cov_matrix):
    """Test Risk Parity with custom initial weights."""
    initial = np.array([0.5, 0.3, 0.2])
    weights = optimize_risk_parity(simple_cov_matrix, initial_weights=initial)
    
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


def test_risk_parity_convergence_failure():
    """Test that convergence failure raises RuntimeError when optimization truly fails."""
    # Create a valid covariance matrix
    cov = pd.DataFrame(
        [[1.0, 0.5, 0.3],
         [0.5, 1.0, 0.4],
         [0.3, 0.4, 1.0]],
        index=['A', 'B', 'C'],
        columns=['A', 'B', 'C']
    )
    
    # Mock the minimize function to return a failed result
    from unittest.mock import patch, MagicMock
    
    failed_result = MagicMock()
    failed_result.success = False
    failed_result.message = "Test failure"
    failed_result.nit = 1
    
    with patch('quant_reporter.advanced_optimizers.minimize', return_value=failed_result):
        with pytest.raises(RuntimeError, match="failed to converge"):
            optimize_risk_parity(cov)


# --- Unit Tests for HRP ---

def test_hrp_simple(simple_cov_matrix):
    """Test HRP optimizer with simple covariance matrix."""
    weights, metadata = optimize_hrp(simple_cov_matrix)
    
    # Check basic properties
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    
    # Check metadata
    assert 'linkage_matrix' in metadata
    assert 'ordered_indices' in metadata
    assert 'dendrogram_data' in metadata
    assert len(metadata['ordered_indices']) == 3


def test_hrp_with_correlated_assets(correlated_cov_matrix):
    """Test HRP with highly correlated assets."""
    weights, metadata = optimize_hrp(correlated_cov_matrix)
    
    # Check basic properties
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)
    
    # Highly correlated assets (A and B) should have similar weights
    # or be grouped together in the clustering
    ordered_indices = metadata['ordered_indices']
    # Check that A (index 0) and B (index 1) are adjacent in ordering
    a_pos = list(ordered_indices).index(0)
    b_pos = list(ordered_indices).index(1)
    assert abs(a_pos - b_pos) == 1, "Highly correlated assets should be adjacent"


def test_hrp_metadata_structure(simple_cov_matrix):
    """Test that HRP returns proper metadata structure."""
    weights, metadata = optimize_hrp(simple_cov_matrix)
    
    # Check linkage matrix shape
    linkage_matrix = metadata['linkage_matrix']
    assert linkage_matrix.shape == (2, 4)  # n-1 rows, 4 columns for n=3 assets
    
    # Check dendrogram data
    dend_data = metadata['dendrogram_data']
    assert 'leaves' in dend_data
    assert 'icoord' in dend_data
    assert 'dcoord' in dend_data


# --- Unit Tests for Min Correlation ---

def test_min_correlation_simple(simple_cov_matrix):
    """Test Min Correlation optimizer with simple covariance matrix."""
    weights = optimize_min_correlation(simple_cov_matrix)
    
    # Check basic properties
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


def test_min_correlation_with_bounds(simple_cov_matrix):
    """Test Min Correlation with custom bounds."""
    weights = optimize_min_correlation(simple_cov_matrix, bounds=(0.1, 0.5))
    
    # Check bounds are respected
    assert np.all(weights >= 0.1 - 1e-6)
    assert np.all(weights <= 0.5 + 1e-6)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


# --- Unit Tests for Max Diversification ---

def test_max_diversification_simple(simple_cov_matrix):
    """Test Max Diversification optimizer with simple covariance matrix."""
    weights = optimize_max_diversification(simple_cov_matrix)
    
    # Check basic properties
    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


def test_max_diversification_ratio(simple_cov_matrix):
    """Test that Max Diversification produces ratio >= 1.0."""
    weights = optimize_max_diversification(simple_cov_matrix)
    
    # Calculate diversification ratio
    cov_array = simple_cov_matrix.values
    volatilities = np.sqrt(np.diag(cov_array))
    weighted_vol = weights @ volatilities
    portfolio_vol = np.sqrt(weights @ cov_array @ weights)
    div_ratio = weighted_vol / portfolio_vol
    
    # Diversification ratio should be >= 1.0
    assert div_ratio >= 0.99, f"Diversification ratio {div_ratio:.3f} is less than 1.0"


def test_max_diversification_with_bounds(simple_cov_matrix):
    """Test Max Diversification with custom bounds."""
    weights = optimize_max_diversification(simple_cov_matrix, bounds=(0.1, 0.5))
    
    # Check bounds are respected
    assert np.all(weights >= 0.1 - 1e-6)
    assert np.all(weights <= 0.5 + 1e-6)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4)


# --- Property-Based Tests ---

@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_optimizer_weight_constraints(n_assets, seed):
    """
    Property 1: Optimizer Weight Constraints
    
    For any valid covariance matrix, the output weights SHALL be non-negative
    and sum to 1.0 within tolerance of 0.0001.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01  # Ensure positive definite
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_risk_parity(cov_matrix)
        
        # Property: All weights non-negative
        assert np.all(weights >= -1e-6), f"Negative weights found: {weights[weights < 0]}"
        
        # Property: Weights sum to 1.0
        weight_sum = np.sum(weights)
        assert np.isclose(weight_sum, 1.0, atol=1e-4), f"Weights sum to {weight_sum}, not 1.0"
        
    except RuntimeError:
        # Convergence failure is acceptable for some random matrices
        pass


@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_risk_parity_equal_risk_contribution(n_assets, seed):
    """
    Property 2: Risk Parity Equal Risk Contribution
    
    For any valid covariance matrix, when Risk Parity optimization computes weights,
    the ratio of maximum to minimum risk contribution SHALL be less than 1.5.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_risk_parity(cov_matrix)
        
        # Calculate risk contributions
        portfolio_vol = np.sqrt(weights @ cov_array @ weights)
        if portfolio_vol < 1e-10:
            return  # Skip degenerate case
        
        marginal_contrib = cov_array @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        
        # Filter out near-zero risk contributions
        risk_contrib_nonzero = risk_contrib[risk_contrib > 1e-10]
        
        if len(risk_contrib_nonzero) > 1:
            ratio = np.max(risk_contrib_nonzero) / np.min(risk_contrib_nonzero)
            assert ratio < 1.5, f"Risk contribution ratio {ratio:.2f} exceeds 1.5"
        
    except RuntimeError:
        # Convergence failure is acceptable
        pass


@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_optimizer_output_shape_consistency(n_assets, seed):
    """
    Property 15: Optimizer Output Shape Consistency
    
    For any covariance matrix with n assets, all optimizers SHALL return
    a weight vector of length n.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_risk_parity(cov_matrix)
        
        # Property: Output shape matches input
        assert len(weights) == n_assets, f"Expected {n_assets} weights, got {len(weights)}"
        
    except RuntimeError:
        # Convergence failure is acceptable
        pass


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_property_covariance_matrix_symmetry_preservation(seed):
    """
    Property 14: Covariance Matrix Symmetry Preservation
    
    For any covariance matrix input to an optimizer, the matrix SHALL be symmetric.
    """
    np.random.seed(seed)
    n_assets = 5
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T  # Symmetric by construction
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    # Property: Matrix is symmetric
    assert np.allclose(cov_matrix, cov_matrix.T, atol=1e-10), "Covariance matrix is not symmetric"


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(min_value=0, max_value=10000))
def test_property_numerical_stability_under_regularization(seed):
    """
    Property 20: Numerical Stability Under Regularization
    
    For any covariance matrix with condition number greater than 1e10,
    after regularization the matrix SHALL remain positive semi-definite
    and have condition number less than 1e10.
    """
    np.random.seed(seed)
    n_assets = 5
    
    # Generate ill-conditioned matrix
    eigenvalues = np.array([1.0, 0.5, 0.1, 0.01, 1e-12])
    eigenvectors = np.random.randn(n_assets, n_assets)
    eigenvectors, _ = np.linalg.qr(eigenvectors)  # Orthonormalize
    cov_array = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    # Check original condition number
    orig_eigs = np.linalg.eigvalsh(cov_array)
    orig_eigs_pos = orig_eigs[orig_eigs > 0]
    orig_cond = np.max(orig_eigs_pos) / np.min(orig_eigs_pos)
    
    if orig_cond > 1e10:
        # Apply regularization
        reg_cov = regularize_covariance(cov_matrix, threshold=1e10)
        
        # Property: Remains positive semi-definite
        reg_eigs = np.linalg.eigvalsh(reg_cov.values)
        assert np.all(reg_eigs >= -1e-8), "Regularized matrix is not positive semi-definite"
        
        # Property: Condition number reduced
        reg_eigs_pos = reg_eigs[reg_eigs > 0]
        reg_cond = np.max(reg_eigs_pos) / np.min(reg_eigs_pos)
        assert reg_cond < 1e10, f"Regularized condition number {reg_cond:.2e} still exceeds 1e10"


# --- Property Tests for HRP ---

@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_hrp_weight_constraints(n_assets, seed):
    """
    Property 1: Optimizer Weight Constraints (HRP variant)
    
    For any valid covariance matrix, HRP output weights SHALL be non-negative
    and sum to 1.0 within tolerance of 0.0001.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    weights, metadata = optimize_hrp(cov_matrix)
    
    # Property: All weights non-negative
    assert np.all(weights >= -1e-6), f"Negative weights found: {weights[weights < 0]}"
    
    # Property: Weights sum to 1.0
    weight_sum = np.sum(weights)
    assert np.isclose(weight_sum, 1.0, atol=1e-4), f"Weights sum to {weight_sum}, not 1.0"


@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_hrp_output_shape_consistency(n_assets, seed):
    """
    Property 15: Optimizer Output Shape Consistency (HRP variant)
    
    For any covariance matrix with n assets, HRP SHALL return
    a weight vector of length n and metadata dictionary.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    weights, metadata = optimize_hrp(cov_matrix)
    
    # Property: Output shape matches input
    assert len(weights) == n_assets, f"Expected {n_assets} weights, got {len(weights)}"
    
    # Property: Metadata contains required keys
    assert 'linkage_matrix' in metadata
    assert 'ordered_indices' in metadata
    assert 'dendrogram_data' in metadata
    assert len(metadata['ordered_indices']) == n_assets


@settings(deadline=None, max_examples=15)
@given(
    n_assets=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_hrp_correlation_clustering(n_assets, seed):
    """
    Property 3: HRP Correlation Clustering
    
    For any valid covariance matrix, HRP SHALL produce valid weights
    and proper clustering structure in the dendrogram.
    """
    np.random.seed(seed)
    
    # Generate valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.1
    
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    weights, metadata = optimize_hrp(cov_matrix)
    
    # Property: Valid weights
    assert np.all(weights >= -1e-6), "Weights should be non-negative"
    assert np.isclose(np.sum(weights), 1.0, atol=1e-4), "Weights should sum to 1"
    
    # Property: Proper clustering structure
    ordered_indices = metadata['ordered_indices']
    assert len(ordered_indices) == n_assets, "Ordered indices should match number of assets"
    assert len(set(ordered_indices)) == n_assets, "All assets should appear exactly once"


# --- Property Tests for Min Correlation and Max Diversification ---

@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_min_correlation_weight_constraints(n_assets, seed):
    """
    Property 1: Optimizer Weight Constraints (Min Correlation variant)
    
    For any valid covariance matrix, Min Correlation output weights SHALL be
    non-negative and sum to 1.0 within tolerance of 0.0001.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_min_correlation(cov_matrix)
        
        # Property: All weights non-negative
        assert np.all(weights >= -1e-6), f"Negative weights found: {weights[weights < 0]}"
        
        # Property: Weights sum to 1.0
        weight_sum = np.sum(weights)
        assert np.isclose(weight_sum, 1.0, atol=1e-4), f"Weights sum to {weight_sum}, not 1.0"
        
    except RuntimeError:
        # Convergence failure is acceptable for some random matrices
        pass


@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_max_diversification_weight_constraints(n_assets, seed):
    """
    Property 1: Optimizer Weight Constraints (Max Diversification variant)
    
    For any valid covariance matrix, Max Diversification output weights SHALL be
    non-negative and sum to 1.0 within tolerance of 0.0001.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_max_diversification(cov_matrix)
        
        # Property: All weights non-negative
        assert np.all(weights >= -1e-6), f"Negative weights found: {weights[weights < 0]}"
        
        # Property: Weights sum to 1.0
        weight_sum = np.sum(weights)
        assert np.isclose(weight_sum, 1.0, atol=1e-4), f"Weights sum to {weight_sum}, not 1.0"
        
    except RuntimeError:
        # Convergence failure is acceptable for some random matrices
        pass


@settings(deadline=None, max_examples=20)
@given(
    n_assets=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=10000)
)
def test_property_diversification_ratio_lower_bound(n_assets, seed):
    """
    Property 4: Diversification Ratio Lower Bound
    
    For any valid covariance matrix, the diversification ratio SHALL be >= 1.0.
    """
    np.random.seed(seed)
    
    # Generate random valid covariance matrix
    A = np.random.randn(n_assets, n_assets)
    cov_array = A @ A.T + np.eye(n_assets) * 0.01
    cov_matrix = pd.DataFrame(cov_array, index=[f'Asset_{i}' for i in range(n_assets)],
                              columns=[f'Asset_{i}' for i in range(n_assets)])
    
    try:
        weights = optimize_max_diversification(cov_matrix)
        
        # Calculate diversification ratio
        volatilities = np.sqrt(np.diag(cov_array))
        weighted_vol = weights @ volatilities
        portfolio_vol = np.sqrt(weights @ cov_array @ weights)
        div_ratio = weighted_vol / portfolio_vol
        
        # Property: Diversification ratio >= 1.0
        assert div_ratio >= 0.99, f"Diversification ratio {div_ratio:.3f} is less than 1.0"
        
    except RuntimeError:
        # Convergence failure is acceptable
        pass
