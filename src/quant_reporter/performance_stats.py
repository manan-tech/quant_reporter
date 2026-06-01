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


def _psr_from(sr, sr_star_pp, n, g3, g4):
    """PSR z-statistic — all quantities on the per-period basis.

    ``sr`` and ``sr_star_pp`` are per-period Sharpe ratios. g3/g4 are
    dimensionless per-period moments. The Bailey & Lopez de Prado (2012)
    variance correction uses the raw per-period SR throughout so that the
    numerator and denominator remain on a consistent basis.
    """
    denom = np.sqrt(max(1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2, 1e-12))
    z = (sr - sr_star_pp) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z))


def probabilistic_sharpe_ratio(returns, sr_threshold=0.0, periods_per_year=252):
    """P(true annualized SR > sr_threshold). ``sr_threshold`` is on the annualized basis.

    Internally converts the annualized threshold to the per-period basis before
    computing the z-statistic, so both sides of the comparison are consistent.
    Returns a float in [0,1] (nan if undefined — fewer than 3 observations or
    zero/non-finite variance).

    Scale-invariant property: at sr_threshold=0, the result is identical for all
    values of periods_per_year (PSR is dimensionless at the zero threshold).
    """
    m = _moments(returns)
    if m is None:
        return float("nan")
    n, sr, g3, g4 = m
    sr_star_pp = sr_threshold / np.sqrt(periods_per_year)
    return _psr_from(sr, sr_star_pp, n, g3, g4)


def _expected_max_sr(n_trials, sr_var):
    """Expected maximum SR across n_trials i.i.d. trials under the null (deflation threshold)."""
    if n_trials is None or n_trials <= 1:
        return 0.0
    e = np.e
    z = ((1 - _EULER) * stats.norm.ppf(1 - 1.0 / n_trials)
         + _EULER * stats.norm.ppf(1 - 1.0 / (n_trials * e)))
    return np.sqrt(max(sr_var, 0.0)) * z


def deflated_sharpe_ratio(returns, n_trials, periods_per_year=252):
    """PSR with the threshold set to the expected max SR across n_trials trials.

    Uses the per-period SR estimation variance so the deflation threshold is on
    the same per-period basis as the z-statistic denominator. Returns [0,1] (nan
    if undefined)."""
    m = _moments(returns)
    if m is None:
        return float("nan")
    n, sr, g3, g4 = m
    # SR estimation variance on the per-period basis (Bailey & LdP eq. 10).
    sr_var_pp = (1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2) / (n - 1)
    sr_star_pp = _expected_max_sr(n_trials, sr_var_pp)
    return _psr_from(sr, sr_star_pp, n, g3, g4)


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

    if not summary:
        return {"summary": {}, "sharpe_diff_pvalues": {}, "best_by_dsr": None}
    # Only rank strategies whose Sharpe is defined (finite); degenerate series (nan
    # sharpe / undefined PSR/DSR) are excluded from selection so they cannot win.
    usable = [k for k in summary if np.isfinite(summary[k]["sharpe"])]
    best = max(usable, key=lambda k: (summary[k]["dsr"] if np.isfinite(summary[k]["dsr"]) else float("-inf"))) if usable else None
    return {"summary": summary, "sharpe_diff_pvalues": pvals, "best_by_dsr": best}
