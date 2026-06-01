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
from .report_context import build_context, build_context_from_prices, ReportContext

# --- Analytics Core (single source of truth) ---
from .analytics import (
    portfolio_returns,
    ReturnsBundle,
    compute_metrics,
    format_metrics,
    PortfolioAnalytics,
)
from .metrics import compute_drawdown, DrawdownResult

# --- Simple Report Generator (create_full_report kept as a back-compat alias) ---
from .portfolio_report import create_portfolio_report
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

# --- Monte Carlo Simulations (low-level helpers) ---
from .monte_carlo import (
    simulate_portfolio_paths,
    calculate_simulation_metrics,
    calculate_success_probabilities,
    plot_simulation_paths,
    plot_simulation_distribution,
    plot_probability_curve
)
# ctx-based Monte Carlo report (supersedes the old monte_carlo.create_monte_carlo_report)
from .monte_carlo_report import create_monte_carlo_report

# --- Black-Litterman ---
from .black_litterman import (
    get_market_caps,
    calculate_market_weights,
    calculate_implied_equilibrium_returns,
    calculate_black_litterman_posterior
)

# --- Full Report Generators ---
from .optimization_report import create_optimization_report
from .validation_report import create_validation_report
from .factor_report import create_factor_report
from .combined_report import create_combined_report

# --- Rebalancing ---
from .rebalancing import simulate_rebalanced_portfolio

# --- Signals (volatility estimation & vol-targeting) ---
from .signals import (
    compute_trailing_volatility,
    volatility_target_positions,
)

# --- Robust Estimators ---
from .robust_estimators import ledoit_wolf_covariance

# --- Backtest Engine (Phase-1 primitives) ---
from .backtest import (
    portfolio_turnover,
    drawdown_stats,
)

# --- Backtest Engine (SP1b: costs, scheduling, the hub) ---
from .backtest import (
    transaction_cost_model,
    generate_rebalance_dates,
    simulate_strategy,
)

# --- Performance Statistics (honest OOS selection) ---
from .performance_stats import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    compare_strategies_oos,
)

# --- Walk-Forward (schedule unlock) ---
from .validation_report import run_rolling_windows

# --- SP2 Phase 3: Risk overlays & position sizing ---
from .sizing import (
    forecast_portfolio_vol,
    target_volatility_scalar,
    inverse_volatility_weights,
    realized_tracking_error,
    kelly_fraction,
    cppi_weights,
)

# --- SP2 Phase 4: Advanced risk decomposition & CVaR ---
from .opt_core import (
    risk_contributions,
    optimize_risk_budget,
    portfolio_cvar,
)

# --- SP2 Phase 5: Tactical signals ---
from .signals import (
    time_series_momentum_signal,
    moving_average_crossover_signal,
    cross_sectional_momentum_score,
    zscore_reversion_signal,
)

# --- SP2 Phase 6: Factor tilts ---
from .factor_tilts import (
    characteristic_tilt_weights,
    factor_neutralize_returns,
    resample_portfolio,
)

# --- SP3: Per-asset info layer ---
from .asset_info import (
    compute_asset_analytics,
    compute_asset_factor_exposures,
    get_asset_fundamentals,
    narrate_asset,
    build_asset_info_table,
)

# --- SP-Strategy: shared metrics library ---
from .metrics import (
    cagr, annual_volatility, sharpe, sortino, calmar, omega,
    max_drawdown, avg_drawdown, ulcer_index, value_at_risk, conditional_var,
    downside_deviation, tracking_error, information_ratio, hit_rate,
    win_loss_ratio, tail_ratio, skewness, kurtosis, summary_metrics,
)

# --- SP-Strategy: objectives / loss surface ---
from .objectives import (
    neg_sharpe, variance, cvar_objective, tracking_error_objective,
    mean_squared_error, mean_absolute_error,
)

# --- SP-Strategy: prebuilt strategies ---
from .strategies import (
    equal_weight, inverse_vol, min_variance, risk_parity, max_sharpe,
    trend_following, cross_sectional_momentum, vol_target_overlay, REGISTRY,
)

# --- SP-Strategy: strategy runner + result + report ---
from .strategy import Strategy, backtest, backtest_many, BacktestResult
from .backtest_report import create_backtest_report

__version__ = "2.0.0"