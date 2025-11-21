# --- Core Utilities (for advanced use) ---
from .data import get_data
from .metrics import calculate_metrics

# --- Simple Report Generator ---
from .analysis import create_full_report

# --- Optimization Core (for advanced use) ---
from .opt_core import (
    get_risk_free_rate,
    get_portfolio_stats,
    find_optimal_portfolio,
    get_optimization_inputs,
    build_constraints,
    objective_min_variance,  
    objective_neg_sharpe    
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

# --- Full Report Generators ---
from .optimization import create_optimization_report
from .validation import create_validation_report
from .combined_report import create_combined_report

__version__ = "1.0.0" 