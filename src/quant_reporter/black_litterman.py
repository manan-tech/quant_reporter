"""
Black-Litterman model for portfolio optimization.

Implements the He & Litterman (1999) Master Formula to combine market equilibrium
returns with investor views (both absolute and relative) to produce posterior 
expected returns and covariance.
"""
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_market_caps(tickers):
    """
    Fetches market capitalizations for the given tickers using yfinance.
    Returns a pandas Series indexed by ticker.
    """
    logger.info("Fetching market caps for Black-Litterman model: %s", tickers)
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            cap = info.get('marketCap')
            if cap is not None:
                caps[t] = float(cap)
        except Exception as e:
            logger.warning("Could not fetch market cap for %s: %s", t, e)
            
    s_caps = pd.Series(caps)
    missing = set(tickers) - set(s_caps.index)
    
    if missing:
        logger.warning("Using mean market cap for missing tickers: %s", missing)
        fill_val = s_caps.mean() if len(s_caps) > 0 else 1.0
        for m in missing:
            s_caps[m] = fill_val
            
    return s_caps


def calculate_market_weights(market_caps):
    """
    Calculates market capitalization weights.
    """
    return market_caps / market_caps.sum()


def calculate_implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion=2.5):
    """
    Calculates the implied equilibrium returns (Pi).
    
    Pi = delta * Sigma * w_mkt
    
    This reverse-engineers the returns that the market is implicitly
    pricing in, given the current market-cap weights and covariance.
    """
    w = market_weights.reindex(cov_matrix.index).fillna(0)
    return risk_aversion * cov_matrix.dot(w)


def calculate_black_litterman_posterior(
    hist_mean_returns, 
    cov_matrix, 
    market_caps=None, 
    risk_aversion=2.5, 
    tau=0.05, 
    view_dict=None, 
    view_confidences=None,
    relative_views=None,
    relative_view_confidences=None
):
    """
    Calculates the Black-Litterman posterior expected returns and covariance.
    
    Uses the Master Formula from He & Litterman (1999):
        E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
        Sigma_p = Sigma + [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1
    
    Args:
        hist_mean_returns (pd.Series): Historical mean returns (annualized).
        cov_matrix (pd.DataFrame): Historical covariance matrix (Sigma), annualized.
        market_caps (pd.Series): Market capitalizations. If None, uses equal weights.
        risk_aversion (float): Risk aversion coefficient (delta). Default 2.5.
        tau (float): Scalar for uncertainty in the prior (0.01-0.05 typical). Default 0.05.
        view_dict (dict): Absolute views, e.g. {'AAPL': 0.10} = "AAPL returns 10% p.a."
        view_confidences (dict): Confidence per absolute view (0.0-1.0).
        relative_views (list): Relative views as tuples:
            [('NVDA', 'AAPL', 0.03)] = "NVDA outperforms AAPL by 3%"
            Each tuple is (outperformer, underperformer, spread).
        relative_view_confidences (list): Confidence per relative view (0.0-1.0),
            same order as relative_views. If None, uses default Omega.
        
    Returns:
        (pd.Series, pd.DataFrame): Posterior expected returns, Posterior covariance matrix.
    """
    tickers = cov_matrix.index
    N = len(tickers)
    ticker_list = list(tickers)
    
    # -- 1. Market Weights -------------------------------------------------------
    if market_caps is not None:
        w_mkt = calculate_market_weights(market_caps)
    else:
        w_mkt = pd.Series(1.0 / N, index=tickers)
        
    w_mkt = w_mkt.reindex(tickers).fillna(0)
    w_mkt = w_mkt / w_mkt.sum()
    
    # -- 2. Implied Equilibrium Returns (Pi) ------------------------------------
    pi = calculate_implied_equilibrium_returns(cov_matrix, w_mkt, risk_aversion)
    
    # -- 3. No-views case -------------------------------------------------------
    has_abs_views = view_dict and len(view_dict) > 0
    has_rel_views = relative_views and len(relative_views) > 0
    
    if not has_abs_views and not has_rel_views:
        posterior_cov = (1 + tau) * cov_matrix
        return pi, posterior_cov
        
    # -- 4. Build P, Q, Omega for ABSOLUTE views --------------------------------
    abs_rows_P = []
    abs_rows_Q = []
    abs_rows_omega = []
    
    if has_abs_views:
        dropped = []
        for t, ret in view_dict.items():
            if t in tickers:
                row = np.zeros(N)
                idx = ticker_list.index(t)
                row[idx] = 1
                abs_rows_P.append(row)
                abs_rows_Q.append(ret)
                
                base_uncertainty = tau * cov_matrix.iloc[idx, idx]
                if view_confidences is not None and t in view_confidences:
                    conf = np.clip(view_confidences[t], 0.01, 1.0)
                    abs_rows_omega.append(base_uncertainty / conf)
                else:
                    abs_rows_omega.append(base_uncertainty)
            else:
                dropped.append(t)
        
        if dropped:
            logger.warning("BL: Absolute views on %s dropped -- tickers not in covariance matrix.", dropped)
            logger.warning("   Available tickers: %s", ticker_list)
    
    # -- 5. Build P, Q, Omega for RELATIVE views --------------------------------
    rel_rows_P = []
    rel_rows_Q = []
    rel_rows_omega = []
    
    if has_rel_views:
        for k, view in enumerate(relative_views):
            outperformer, underperformer, spread = view
            
            if outperformer not in tickers:
                logger.warning("BL: Relative view dropped -- '%s' not in covariance matrix.", outperformer)
                continue
            if underperformer not in tickers:
                logger.warning("BL: Relative view dropped -- '%s' not in covariance matrix.", underperformer)
                continue
                
            idx_out = ticker_list.index(outperformer)
            idx_und = ticker_list.index(underperformer)
            
            row = np.zeros(N)
            row[idx_out] = 1      # long the outperformer
            row[idx_und] = -1     # short the underperformer
            rel_rows_P.append(row)
            rel_rows_Q.append(spread)
            
            # Omega for relative view: variance of the spread
            # Var(A - B) = Sigma_AA + Sigma_BB - 2*Sigma_AB
            sigma_aa = cov_matrix.iloc[idx_out, idx_out]
            sigma_bb = cov_matrix.iloc[idx_und, idx_und]
            sigma_ab = cov_matrix.iloc[idx_out, idx_und]
            spread_var = tau * (sigma_aa + sigma_bb - 2 * sigma_ab)
            
            if relative_view_confidences is not None and k < len(relative_view_confidences):
                conf = np.clip(relative_view_confidences[k], 0.01, 1.0)
                rel_rows_omega.append(spread_var / conf)
            else:
                rel_rows_omega.append(spread_var)
            
            logger.info("BL relative view: %s outperforms %s by %.1f%% (Omega=%.6f)",
                        outperformer, underperformer, spread * 100, rel_rows_omega[-1])
    
    # -- 6. Combine absolute + relative into P, Q, Omega -----------------------
    all_P = abs_rows_P + rel_rows_P
    all_Q = abs_rows_Q + rel_rows_Q
    all_omega = abs_rows_omega + rel_rows_omega
    
    K = len(all_P)
    if K == 0:
        posterior_cov = (1 + tau) * cov_matrix
        return pi, posterior_cov
    
    P = np.array(all_P)               # (K x N)
    Q = np.array(all_Q)               # (K,)
    Omega = np.diag(all_omega)         # (K x K) diagonal
    
    logger.info("BL: %d absolute views + %d relative views = %d total views",
                len(abs_rows_P), len(rel_rows_P), K)
    
    # -- 7. Master Formula: Posterior Mean --------------------------------------
    tau_sigma = tau * cov_matrix.values
    tau_sigma_inv = np.linalg.inv(tau_sigma)
    omega_inv = np.linalg.inv(Omega)
    
    M = tau_sigma_inv + P.T @ omega_inv @ P
    M_inv = np.linalg.inv(M)
    
    posterior_means = M_inv @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
    
    # -- 8. Posterior Covariance ------------------------------------------------
    posterior_cov = cov_matrix.values + M_inv
    
    return (
        pd.Series(posterior_means, index=tickers),
        pd.DataFrame(posterior_cov, index=tickers, columns=tickers)
    )
