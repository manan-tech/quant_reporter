# src/quant_reporter/objectives.py
"""Optimization objective / loss functions (SP-Strategy).

Each is minimize-ready. The portfolio objectives use the same
(weights, mean_returns, cov_matrix, risk_free_rate) arg order as
opt_core.find_optimal_portfolio, so they are drop-in replacements.
"""
import numpy as np


def _cov(cov_matrix):
    return cov_matrix.values if hasattr(cov_matrix, "values") else np.asarray(cov_matrix, dtype=float)


def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mean_returns, dtype=float)
    cov = _cov(cov_matrix)
    port_return = float(w @ mu)
    port_vol = float(np.sqrt(max(w @ cov @ w, 1e-30)))
    return float(-(port_return - risk_free_rate) / port_vol)


def variance(weights, mean_returns=None, cov_matrix=None, risk_free_rate=0.02):
    w = np.asarray(weights, dtype=float)
    cov = _cov(cov_matrix)
    return float(w @ cov @ w)


def cvar_objective(weights, returns_matrix, confidence=0.95):
    w = np.asarray(weights, dtype=float)
    R = returns_matrix.values if hasattr(returns_matrix, "values") else np.asarray(returns_matrix, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        port = R @ w
    threshold = np.quantile(port, 1.0 - confidence)
    tail = port[port <= threshold]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


def tracking_error_objective(weights, returns_matrix, benchmark_returns):
    w = np.asarray(weights, dtype=float)
    R = returns_matrix.values if hasattr(returns_matrix, "values") else np.asarray(returns_matrix, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        active = R @ w - b
    if active.size < 2:
        return 0.0
    return float(np.std(active, ddof=1))


def mean_squared_error(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yt - yp) ** 2))


def mean_absolute_error(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))
