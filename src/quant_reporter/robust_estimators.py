# src/quant_reporter/robust_estimators.py
"""Robust covariance estimation (Ledoit-Wolf shrinkage).

Hardens the covariance input every optimizer consumes. Returns annualized matrices
matching the get_optimization_inputs contract (cov scaled by periods_per_year).
No new dependencies: sklearn.covariance is already pinned.
"""
import numpy as np
import pandas as pd


def _sample_cov(X):
    """MLE sample covariance (divide by T), matching Ledoit-Wolf's derivation."""
    T = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / T, Xc


def _constant_correlation_target(S):
    """LW(2004) constant-correlation target F: F_ii=S_ii, F_ij=r_bar*sqrt(S_ii*S_jj)."""
    d = np.sqrt(np.diag(S))
    outer = np.outer(d, d)
    corr = S / outer
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    r_bar = corr[mask].mean()
    F = r_bar * outer
    np.fill_diagonal(F, np.diag(S))
    return F, r_bar


def _lw_constant_correlation_delta(Xc, S, F, r_bar):
    """Optimal shrinkage intensity for the constant-correlation target (LW 2004)."""
    T, n = Xc.shape
    # pi: sum of asymptotic variances of sample-cov entries
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / T - S ** 2
    pi_hat = pi_mat.sum()
    # rho: estimator of sum of asy covariances between F and S entries
    d = np.sqrt(np.diag(S))
    # theta_ii,ij terms
    term1 = ((Xc ** 3).T @ Xc) / T - np.diag(S)[:, None] * S
    term2 = ((Xc).T @ (Xc ** 3)) / T - np.diag(S)[None, :] * S
    theta_ii = term1  # (i diag with ij)
    theta_jj = term2
    ratio_ji = np.outer(1.0 / d, d)  # sqrt(S_jj/S_ii)
    ratio_ij = np.outer(d, 1.0 / d)  # sqrt(S_ii/S_jj)
    rho_off = (r_bar / 2.0) * (ratio_ji * theta_ii + ratio_ij * theta_jj)
    mask = ~np.eye(n, dtype=bool)
    rho_hat = np.diag(pi_mat).sum() + rho_off[mask].sum()
    # gamma: misfit between sample cov and target
    gamma_hat = ((F - S) ** 2).sum()
    if gamma_hat <= 0:
        return 0.0
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(max(0.0, min(1.0, kappa / T)))


def ledoit_wolf_covariance(returns, target="constant_correlation", periods_per_year=252, delta=None):
    """Ledoit-Wolf shrinkage covariance, annualized.

    Args:
        returns: DataFrame of periodic asset returns.
        target: 'constant_correlation' (LW 2004 closed-form) or 'identity'
                (sklearn spherical LedoitWolf).
        periods_per_year: annualization factor applied to every returned matrix.
        delta: if given, use this shrinkage intensity directly (skip estimation).

    Returns:
        dict with 'cov_matrix' (annualized PD DataFrame), 'shrinkage' (float in [0,1]),
        'target' (str), 'sample_cov' (annualized DataFrame), 'target_matrix'
        (annualized DataFrame).
    """
    cols = list(returns.columns)
    X = returns.values.astype(float)
    if X.shape[0] < 2:
        raise ValueError(
            f"ledoit_wolf_covariance requires at least 2 periods; got T={X.shape[0]}."
        )
    var = X.var(axis=0, ddof=1)
    bad = [cols[i] for i, v in enumerate(var) if v < 1e-14 or not np.isfinite(v)]
    if bad:
        raise ValueError(
            f"Zero or non-finite variance for asset(s) {bad}; cannot form "
            "constant-correlation shrinkage target (division by zero in correlation). "
            "Drop or floor these columns before estimation."
        )
    S, Xc = _sample_cov(X)

    if target == "identity":
        mu = np.trace(S) / S.shape[0]
        F = mu * np.eye(S.shape[0])
    elif target == "constant_correlation":
        F, r_bar = _constant_correlation_target(S)
    else:
        raise ValueError(f"Unknown target {target!r}; expected 'constant_correlation' or 'identity'.")

    if delta is None:
        if target == "identity":
            from sklearn.covariance import ledoit_wolf as _skl
            _, shrink = _skl(X)
            delta = float(shrink)
        else:
            delta = _lw_constant_correlation_delta(Xc, S, F, r_bar)
    else:
        delta = float(delta)

    shrunk = delta * F + (1.0 - delta) * S
    f = periods_per_year

    def _df(m):
        return pd.DataFrame(m * f, index=cols, columns=cols)

    return {
        "cov_matrix": _df(shrunk),
        "shrinkage": delta,
        "target": target,
        "sample_cov": _df(S),
        "target_matrix": _df(F),
    }
