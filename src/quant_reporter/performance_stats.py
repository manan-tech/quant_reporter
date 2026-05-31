"""Honest out-of-sample strategy selection: Probabilistic & Deflated Sharpe Ratios
(Bailey & Lopez de Prado) and a pairwise Sharpe-difference test (Jobson-Korkie /
Memmel). Pure, offline, no new dependencies (scipy.stats only)."""
import numpy as np
import pandas as pd
from scipy import stats

_EULER = 0.5772156649015329


def _moments(returns):
    r = pd.Series(returns).dropna().values.astype(float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or not np.isfinite(sd) or sd == 0:
        return None
    sr = r.mean() / sd
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))  # non-excess kurtosis
    return n, sr, g3, g4


def _psr_from(sr, sr_star, n, g3, g4, periods_per_year=1):
    """Probabilistic Sharpe Ratio from moments.

    The skew/kurtosis-adjusted standard error (Bailey & Lopez de Prado) is computed on
    the per-period Sharpe; both the estimated Sharpe (``sr``) and the threshold
    (``sr_star``) are annualized by ``sqrt(periods_per_year)`` before forming the
    z-statistic, so a threshold expressed on the annualized basis is compared like-for-
    like. ``periods_per_year=1`` recovers the raw per-period PSR.
    """
    ann = np.sqrt(periods_per_year)
    sr_a = sr * ann
    denom = np.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2, 1e-12))
    z = (sr_a - sr_star) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def probabilistic_sharpe_ratio(returns, sr_threshold=0.0, periods_per_year=252):
    """P(true annualized SR > sr_threshold). ``sr_threshold`` is on the annualized basis
    (same units as the annualized Sharpe). Returns a float in [0,1] (0.5 if undefined)."""
    m = _moments(returns)
    if m is None:
        return 0.5
    n, sr, g3, g4 = m
    return _psr_from(sr, sr_threshold, n, g3, g4, periods_per_year)


def _expected_max_sr(n_trials, sr_var):
    """Expected maximum SR across n_trials i.i.d. trials under the null (deflation threshold)."""
    if n_trials is None or n_trials <= 1:
        return 0.0
    e = np.e
    z = ((1 - _EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + _EULER * stats.norm.ppf(1 - 1.0 / (n_trials * e)))
    return np.sqrt(max(sr_var, 0.0)) * z


def deflated_sharpe_ratio(returns, n_trials, periods_per_year=252):
    """PSR with the threshold set to the expected max SR across n_trials (single-series
    approximation: cross-trial SR variance ~ the SR estimation variance). The deflation
    threshold and PSR are on the annualized basis. Returns [0,1]."""
    m = _moments(returns)
    if m is None:
        return 0.5
    n, sr, g3, g4 = m
    # SR estimation variance on the annualized basis (matches the annualized z-stat).
    sr_var = periods_per_year * (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2) / (n - 1)
    sr_star = _expected_max_sr(n_trials, sr_var)
    return _psr_from(sr, sr_star, n, g3, g4, periods_per_year)


def _sharpe_diff_pvalue(ra, rb):
    """Two-sided p-value for SR_a - SR_b via Jobson-Korkie with Memmel (2003) correction."""
    a = pd.Series(ra).dropna()
    b = pd.Series(rb).dropna()
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    x, y = df["a"].values, df["b"].values
    n = len(df)
    mx, my = x.mean(), y.mean()
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    if sx == 0 or sy == 0:
        return float("nan")
    srx, sry = mx / sx, my / sy
    rho = np.corrcoef(x, y)[0, 1]
    var = (1.0 / n) * (2 - 2 * rho + 0.5 * (srx ** 2 + sry ** 2 - 2 * srx * sry * rho ** 2))
    if var <= 0:
        return float("nan")
    z = (srx - sry) / np.sqrt(var)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def compare_strategies_oos(returns_dict, benchmark_returns=None, n_trials=None,
                           sr_threshold=0.0, periods_per_year=252):
    """Rank strategies by Deflated Sharpe, with PSR and pairwise SR-difference p-values.

    Returns {'summary': {name: {'sharpe' (annualized), 'psr', 'dsr'}},
             'sharpe_diff_pvalues': {'A vs B': p, ...}, 'best_by_dsr': name}.
    """
    names = list(returns_dict)
    n_eff = n_trials if n_trials is not None else max(len(names), 1)
    ann = np.sqrt(periods_per_year)
    summary = {}
    for name, r in returns_dict.items():
        m = _moments(r)
        sr_ann = float((m[1] * ann)) if m is not None else float("nan")
        summary[name] = {
            "sharpe": sr_ann,
            "psr": probabilistic_sharpe_ratio(r, sr_threshold, periods_per_year),
            "dsr": deflated_sharpe_ratio(r, n_eff, periods_per_year),
        }

    pvals = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pvals[f"{names[i]} vs {names[j]}"] = _sharpe_diff_pvalue(
                returns_dict[names[i]], returns_dict[names[j]])
    if benchmark_returns is not None:
        for name in names:
            pvals[f"{name} vs Benchmark"] = _sharpe_diff_pvalue(returns_dict[name], benchmark_returns)

    best = max(summary, key=lambda k: (summary[k]["dsr"] if np.isfinite(summary[k]["dsr"]) else -1))
    return {"summary": summary, "sharpe_diff_pvalues": pvals, "best_by_dsr": best}
