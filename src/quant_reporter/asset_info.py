# src/quant_reporter/asset_info.py
"""Per-asset info layer (SP3).

Produces a standardized per-holding DataFrame: analytics, factor exposures,
allocation context, optional fundamental data, and a text layer.

yfinance calls are isolated in get_asset_fundamentals and degrade gracefully —
a missing ticker or network failure returns NaN columns, never raises.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# 1. Per-asset computed analytics
# ---------------------------------------------------------------------------

def compute_asset_analytics(prices, weights, benchmark_col=None,
                            risk_free_rate=0.02, periods_per_year=252):
    """Per-asset return and risk analytics computed from prices.

    Args:
        prices: DataFrame of prices.  May include benchmark_col.
        weights: dict {ticker: weight} for the asset columns.
        benchmark_col: column name in prices to use as benchmark (optional).
        risk_free_rate: annualized risk-free rate for Sharpe/Sortino/alpha.
        periods_per_year: annualization factor.

    Returns:
        DataFrame indexed by ticker with columns:
          total_return, annualized_return, annualized_vol,
          sharpe, sortino, max_drawdown,
          beta, alpha (NaN when no benchmark), weight,
          correlation_to_portfolio.
    """
    asset_cols = [c for c in prices.columns if c != benchmark_col]
    returns = prices[asset_cols].pct_change().dropna()
    rfr_per_period = risk_free_rate / periods_per_year

    # Portfolio returns for correlation computation
    w_vec = pd.Series({c: weights.get(c, 0.0) for c in asset_cols}, dtype=float)
    w_sum = w_vec.sum()
    if w_sum > 0:
        w_vec = w_vec / w_sum
    port_returns = (returns * w_vec).sum(axis=1)

    benchmark_returns = None
    if benchmark_col and benchmark_col in prices.columns:
        benchmark_returns = prices[benchmark_col].pct_change().dropna().reindex(returns.index)

    rows = {}
    _nan_row = {
        "total_return": float("nan"), "annualized_return": float("nan"),
        "annualized_vol": float("nan"), "sharpe": float("nan"),
        "sortino": float("nan"), "max_drawdown": float("nan"),
        "beta": float("nan"), "alpha": float("nan"),
        "weight": float("nan"), "correlation_to_portfolio": float("nan"),
    }

    for col in asset_cols:
        r = returns[col].dropna()
        if len(r) < 2:
            rows[col] = {**_nan_row, "weight": float(weights.get(col, 0.0))}
            continue

        n = len(r)
        total_return = float((1 + r).prod() - 1)
        ann_return = float((1 + total_return) ** (periods_per_year / n) - 1)
        ann_vol = float(r.std(ddof=1) * np.sqrt(periods_per_year))

        sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else float("nan")

        downside = r[r < rfr_per_period]
        if len(downside) > 1:
            sortino_denom = float(downside.std(ddof=1) * np.sqrt(periods_per_year))
            sortino = float((ann_return - risk_free_rate) / sortino_denom) if sortino_denom > 0 else float("nan")
        else:
            sortino = float("nan")

        cum = (1 + r).cumprod()
        peak = cum.cummax()
        max_dd = float(((cum - peak) / peak).min())

        beta, alpha = float("nan"), float("nan")
        if benchmark_returns is not None:
            aligned = pd.concat([r, benchmark_returns], axis=1).dropna()
            if len(aligned) >= 10:
                a_ret = aligned.iloc[:, 0].values
                b_ret = aligned.iloc[:, 1].values
                b_var = float(np.var(b_ret, ddof=1))
                if b_var > 0:
                    beta = float(np.cov(a_ret, b_ret, ddof=1)[0, 1] / b_var)
                    n_b = len(b_ret)
                    bench_ann = float((1 + b_ret).prod() ** (periods_per_year / n_b) - 1)
                    alpha = float((ann_return - risk_free_rate) - beta * (bench_ann - risk_free_rate))

        aligned_port = pd.concat([r, port_returns], axis=1).dropna()
        corr_to_port = float(aligned_port.corr().iloc[0, 1]) if len(aligned_port) >= 2 else float("nan")

        rows[col] = {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "beta": beta,
            "alpha": alpha,
            "weight": float(weights.get(col, 0.0)),
            "correlation_to_portfolio": corr_to_port,
        }

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 2. Fundamental data (gracefully degrading network calls)
# ---------------------------------------------------------------------------

_FUNDAMENTAL_FIELDS = {
    "shortName": "name",
    "sector": "sector",
    "marketCap": "market_cap",
    "trailingPE": "pe_ratio",
    "dividendYield": "dividend_yield",
    "fiftyTwoWeekHigh": "week52_high",
    "fiftyTwoWeekLow": "week52_low",
}


def get_asset_fundamentals(tickers):
    """Fetch yfinance .info for each ticker.  Degrades gracefully on failure.

    Returns DataFrame indexed by ticker with columns:
      name, sector, market_cap, pe_ratio, dividend_yield, week52_high, week52_low.
    Any missing field or network error → NaN / None.  Never raises.
    """
    import yfinance as yf

    rows = {}
    for ticker in tickers:
        row = {col: (None if col in ("name", "sector") else float("nan"))
               for col in _FUNDAMENTAL_FIELDS.values()}
        try:
            info = yf.Ticker(ticker).info
            for yf_key, col in _FUNDAMENTAL_FIELDS.items():
                val = info.get(yf_key)
                if val is not None:
                    row[col] = val
        except Exception as exc:
            logger.debug("get_asset_fundamentals: %s failed: %s", ticker, exc)
        rows[ticker] = row
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 3. Factor exposures (OLS per asset)
# ---------------------------------------------------------------------------

def compute_asset_factor_exposures(returns, factor_returns):
    """OLS factor loadings per asset.

    Args:
        returns: DataFrame of asset returns (T × N).
        factor_returns: DataFrame of factor returns (T × K).

    Returns:
        DataFrame (N × K) of OLS factor betas.  NaN where regression fails.
    """
    min_obs = max(5, factor_returns.shape[1] + 2)
    aligned = returns.join(factor_returns, how="inner")
    if aligned.shape[0] < min_obs:
        return pd.DataFrame(
            float("nan"), index=returns.columns, columns=factor_returns.columns
        )

    F = factor_returns.reindex(aligned.index).values
    rows = {}
    for col in returns.columns:
        y = aligned[col].values
        mask = np.isfinite(y) & np.all(np.isfinite(F), axis=1)
        if mask.sum() < min_obs:
            rows[col] = {fc: float("nan") for fc in factor_returns.columns}
            continue
        try:
            Fm = np.column_stack([np.ones(mask.sum()), F[mask]])
            coefs, _, _, _ = np.linalg.lstsq(Fm, y[mask], rcond=None)
            rows[col] = dict(zip(factor_returns.columns, coefs[1:]))
        except Exception:
            rows[col] = {fc: float("nan") for fc in factor_returns.columns}

    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# 4. Text layer
# ---------------------------------------------------------------------------

def narrate_asset(row, llm_hook=None):
    """One-sentence narration for a single asset row.

    Args:
        row: dict-like with keys from compute_asset_analytics output.
        llm_hook: optional callable(row) → str.  Used in preference to the
                  templated default; silently falls back if it raises or
                  returns an empty string.

    Returns:
        str
    """
    if llm_hook is not None:
        try:
            result = llm_hook(row)
            if result and isinstance(result, str) and result.strip():
                return result.strip()
        except Exception:
            pass

    def _pct(val):
        if val is None:
            return "N/A"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return "N/A"
        return "N/A" if not np.isfinite(f) else f"{f * 100:.1f}%"

    def _num(val, dec=2):
        if val is None:
            return "N/A"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return "N/A"
        return "N/A" if not np.isfinite(f) else f"{f:.{dec}f}"

    parts = []
    w = row.get("weight")
    if _is_finite(w):
        parts.append(f"weight {_pct(w)}")
    ann_ret = row.get("annualized_return")
    if _is_finite(ann_ret):
        direction = "gained" if float(ann_ret) >= 0 else "lost"
        parts.append(f"{direction} {_pct(abs(float(ann_ret)))} p.a.")
    ann_vol = row.get("annualized_vol")
    if _is_finite(ann_vol):
        parts.append(f"vol {_pct(ann_vol)}")
    sharpe = row.get("sharpe")
    if _is_finite(sharpe):
        parts.append(f"Sharpe {_num(sharpe)}")
    max_dd = row.get("max_drawdown")
    if _is_finite(max_dd):
        parts.append(f"max drawdown {_pct(max_dd)}")

    return ("Asset: " + ", ".join(parts) + ".") if parts else "Insufficient data."


def _is_finite(val):
    if val is None:
        return False
    try:
        return np.isfinite(float(val))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 5. Master entry point
# ---------------------------------------------------------------------------

def build_asset_info_table(prices, weights, benchmark_col=None, factor_returns=None,
                           risk_free_rate=0.02, periods_per_year=252,
                           fetch_fundamentals=False, llm_hook=None):
    """Build a comprehensive per-asset information table.

    Args:
        prices: DataFrame of asset prices (may include benchmark_col).
        weights: dict {ticker: weight}.
        benchmark_col: column name to use as benchmark.
        factor_returns: optional DataFrame for factor exposure regression.
        risk_free_rate: annualized risk-free rate.
        periods_per_year: annualization factor.
        fetch_fundamentals: if True, fetch yfinance fundamentals (makes network calls).
        llm_hook: optional callable(row_dict) → str for text narration.

    Returns:
        DataFrame indexed by ticker with analytics + optional exposures/fundamentals
        + 'narration' column.
    """
    asset_cols = [c for c in prices.columns if c != benchmark_col]

    table = compute_asset_analytics(
        prices, weights,
        benchmark_col=benchmark_col,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    if factor_returns is not None:
        rets = prices[asset_cols].pct_change().dropna()
        exposures = compute_asset_factor_exposures(rets, factor_returns)
        table = table.join(exposures, how="left")

    if fetch_fundamentals:
        fundamentals = get_asset_fundamentals(list(table.index))
        table = table.join(fundamentals, how="left")

    narrations = {ticker: narrate_asset(row.to_dict(), llm_hook=llm_hook)
                  for ticker, row in table.iterrows()}
    table["narration"] = pd.Series(narrations)

    return table
