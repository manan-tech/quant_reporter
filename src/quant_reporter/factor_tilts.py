# src/quant_reporter/factor_tilts.py
"""Factor-tilt and portfolio-resampling primitives (SP2 Phase 6).

Pure functions that tilt base allocations toward desired factor exposures or
produce factor-residual returns.  No dependencies beyond numpy/pandas/scipy.
"""
import numpy as np
import pandas as pd


def characteristic_tilt_weights(base_weights, scores, tilt_strength=0.1):
    """Tilt base portfolio weights toward high-score assets.

    New weights = base_weights + tilt_strength * (score_weights - base_weights),
    where score_weights are proportional to positive scores (assets with non-positive
    scores receive zero score weight).  Clipped to [0, 1] and renormalized.

    Args:
        base_weights: dict {ticker: weight}.  Must sum to > 0.
        scores: dict {ticker: numeric score}.  Higher score → more tilt toward that asset.
                Negative / zero scores contribute zero.
        tilt_strength: float in [0, 1].  0 = unchanged; 1 = fully score-driven.

    Returns:
        dict {ticker: new_weight} — weights sum to 1.0.
    """
    if not (0.0 <= tilt_strength <= 1.0):
        raise ValueError(f"tilt_strength must be in [0, 1]; got {tilt_strength}.")
    tickers = list(base_weights)
    base = np.array([base_weights[t] for t in tickers], dtype=float)
    base_sum = base.sum()
    if base_sum <= 0:
        raise ValueError("base_weights must sum to a positive value.")
    base = base / base_sum

    raw_scores = np.array([max(0.0, float(scores.get(t, 0.0))) for t in tickers])
    score_sum = raw_scores.sum()
    if score_sum <= 0:
        # No positive scores: return base weights unchanged
        return dict(zip(tickers, base))

    score_w = raw_scores / score_sum
    tilted = base + tilt_strength * (score_w - base)
    tilted = np.clip(tilted, 0.0, 1.0)
    total = tilted.sum()
    if total <= 0:
        return dict(zip(tickers, base))
    return dict(zip(tickers, tilted / total))


def factor_neutralize_returns(returns, factor_returns):
    """Remove factor exposures from asset returns via OLS, returning residuals.

    For each asset, fits: r_asset = alpha + sum_k beta_k * f_k + epsilon.
    Returns the residuals epsilon (factor-neutral returns).

    Args:
        returns: DataFrame of asset returns (T × N).
        factor_returns: DataFrame of factor returns (T × K).

    Returns:
        DataFrame of residuals (same shape and index as returns).
        Columns with insufficient data are filled with NaN.
    """
    aligned = returns.join(factor_returns, how="inner", lsuffix="", rsuffix="_f")
    # Resolve column name conflicts
    asset_cols = [c for c in aligned.columns if c in returns.columns]
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    residuals[:] = float("nan")

    F_cols = list(factor_returns.columns)
    if aligned.shape[0] < len(F_cols) + 2:
        return residuals

    F_raw = factor_returns.reindex(aligned.index).values
    for col in returns.columns:
        if col not in aligned.columns:
            continue
        y = aligned[col].values
        mask = np.isfinite(y) & np.all(np.isfinite(F_raw), axis=1)
        if mask.sum() < len(F_cols) + 2:
            continue
        Fm = np.column_stack([np.ones(mask.sum()), F_raw[mask]])
        try:
            coefs, _, _, _ = np.linalg.lstsq(Fm, y[mask], rcond=None)
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                fitted = Fm @ coefs
            resid = y[mask] - fitted
            # Place residuals back on original index
            idx = np.where(mask)[0]
            orig_idx = aligned.index[idx]
            residuals.loc[orig_idx, col] = resid
        except Exception:
            pass

    return residuals.astype(float)


def resample_portfolio(prices, weights, n_samples=100, periods_per_year=252, seed=42):
    """Resampled efficiency: average optimal weights across bootstrap draws.

    For each bootstrap draw of the return history, compute the minimum-variance
    weights using sklearn (or fallback equal weight) and average.  Reduces
    estimation error in the covariance matrix.

    Args:
        prices: DataFrame of asset prices.
        weights: dict {ticker: initial_weight} (used only for ticker ordering).
        n_samples: number of bootstrap resamples.
        periods_per_year: annualization factor for covariance estimation.
        seed: RNG seed for reproducibility.

    Returns:
        dict {ticker: averaged_weight} — weights sum to 1.0.
    """
    from .robust_estimators import ledoit_wolf_covariance
    import scipy.optimize as sco

    tickers = [t for t in weights if t in prices.columns]
    returns = prices[tickers].pct_change().dropna()
    n, k = returns.shape
    if n < k + 1:
        total = sum(weights.get(t, 0.0) for t in tickers)
        return {t: weights.get(t, 0.0) / total for t in tickers}

    rng = np.random.default_rng(seed)
    weight_draws = []

    for _ in range(n_samples):
        sample_idx = rng.choice(n, size=n, replace=True)
        sample = returns.iloc[sample_idx]
        try:
            lw = ledoit_wolf_covariance(sample, periods_per_year=periods_per_year)
            cov = lw["cov_matrix"].values
        except Exception:
            cov = np.cov(sample.values.T) * periods_per_year

        def _min_var(w):
            return float(w @ cov @ w)

        w0 = np.ones(k) / k
        bounds = [(0.0, 1.0)] * k
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = sco.minimize(_min_var, w0, method="SLSQP", bounds=bounds, constraints=cons,
                           options={"ftol": 1e-10, "maxiter": 200})
        w_opt = res.x if res.success else w0
        w_opt = np.clip(w_opt, 0.0, 1.0)
        w_opt /= w_opt.sum()
        weight_draws.append(w_opt)

    avg = np.mean(weight_draws, axis=0)
    avg /= avg.sum()
    return dict(zip(tickers, avg))
