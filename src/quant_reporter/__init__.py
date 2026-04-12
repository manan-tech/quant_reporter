import logging

# Library logging best practice: add NullHandler so users can control output.
logging.getLogger('quant_reporter').addHandler(logging.NullHandler())


def enable_logging(level=logging.INFO):
    """Enable console logging for quant_reporter. Call with logging.DEBUG for verbose output."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s | %(name)s | %(message)s'))
    root_logger = logging.getLogger('quant_reporter')
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


# --- Core Utilities (for advanced use) ---
from .data import get_data
from .metrics import calculate_metrics
from .report_context import build_context

# --- Simple Report Generator ---
# Note: create_full_report is deprecated, use create_portfolio_report or combined_report
from .portfolio_report import create_portfolio_report as create_full_report

# --- Optimization Core (for advanced use) ---
from .opt_core import (
    get_risk_free_rate,
    get_portfolio_stats,
    find_optimal_portfolio,
    get_optimization_inputs,
    build_constraints,
    objective_min_variance,  
    objective_neg_sharpe,
    validate_covariance_matrix,
    regularize_covariance
)

# --- Advanced Optimizers ---
from .advanced_optimizers import (
    optimize_risk_parity,
    optimize_hrp,
    optimize_min_correlation,
    optimize_max_diversification
)

# --- Factor Models ---
from .factor_models import (
    fetch_fama_french_factors,
    run_factor_regression,
    compute_factor_attribution
)

# --- Performance Attribution ---
from .attribution import (
    compute_brinson_attribution
)

# --- Simple Plotting (for advanced use) ---
from .plotting import (
    plot_cumulative_returns,
    plot_rolling_volatility,
    plot_regression,
    plot_rolling_sharpe,
    plot_monthly_distribution,
    plot_yearly_returns
)

# --- Optimization Plotting (for advanced use) ---
from .opt_plotting import (
    plot_efficient_frontier,
    plot_correlation_heatmap,
    plot_cumulative_comparison,
    plot_drawdown_comparison,
    plot_rolling_sharpe,
    plot_composition_pies,
    plot_risk_contribution,
    plot_monthly_heatmaps,
    plot_portfolio_vs_constituents
)

# --- Monte Carlo Simulations ---
from .monte_carlo import (
    simulate_portfolio_paths,
    calculate_simulation_metrics,
    calculate_success_probabilities,
    plot_simulation_paths,
    plot_simulation_distribution,
    plot_probability_curve,
    create_monte_carlo_report
)

# --- Black-Litterman ---
from .black_litterman import (
    get_market_caps,
    calculate_market_weights,
    calculate_implied_equilibrium_returns,
    calculate_black_litterman_posterior
)

# --- Full Report Generators ---
from .portfolio_report import create_portfolio_report
from .optimization_report import create_optimization_report
from .validation_report import create_validation_report
from .combined_report import create_combined_report
from .factor_report import create_factor_report
from .monte_carlo_report import create_monte_carlo_report

# --- Rebalancing ---
from .rebalancing import simulate_rebalanced_portfolio

__version__ = "2.0.0" 