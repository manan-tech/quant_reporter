"""Backtest-honesty diagnostics — tell the user when a backtest is likely a fluke.

Implements the Bailey & Lopez de Prado family of overfitting controls:

- :func:`probability_of_backtest_overfitting` — PBO via Combinatorially Symmetric
  Cross-Validation (CSCV): how often the in-sample-best configuration lands at or
  below the out-of-sample median.
- :func:`min_track_record_length` — MinTRL: observations needed for a Sharpe to be
  statistically distinguishable from a target.
- :func:`min_backtest_length` — MinBTL: minimum backtest length before N trials can
  fake a target Sharpe by chance.
- :func:`assess_overfitting` — a one-call plain-language verdict bundling the above
  with the deflated Sharpe ratio.

Pure and offline (numpy/pandas/scipy only). Consume-only: callers pass a returns
matrix they already have (e.g. from ``backtest_many`` over a parameter set).

Refs: Bailey & Lopez de Prado, *The Deflated Sharpe Ratio* and *The Probability of
Backtest Overfitting*; Bailey, Borwein, Lopez de Prado & Zhu, *Pseudo-Mathematics
and Financial Charlatanism*.
"""
import logging
from dataclasses import dataclass, field
from html import escape
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .performance_stats import _moments, _expected_max_sr, deflated_sharpe_ratio

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {"pbo_robust": 0.10, "pbo_overfit": 0.50, "dsr_strong": 0.95}


def min_track_record_length(returns, sr_target=0.0, prob=0.95, periods_per_year=252):
    """Minimum Track Record Length: observations needed for ``PSR(sr_target) >= prob``.

    Given the observed Sharpe and its higher moments, returns the number of
    (per-period) observations required before the track record's Sharpe is
    statistically distinguishable from the annualized ``sr_target`` at confidence
    ``prob``. Returns ``nan`` when the observed Sharpe does not exceed the target
    (unreachable) or the moments are undefined (<3 obs / zero variance).
    """
    m = _moments(returns)
    if m is None:
        return float("nan")
    _n, sr, g3, g4 = m
    sr_star_pp = sr_target / np.sqrt(periods_per_year)
    if sr <= sr_star_pp:
        return float("nan")
    z = stats.norm.ppf(prob)
    var_factor = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr ** 2
    return 1.0 + max(var_factor, 0.0) * (z / (sr - sr_star_pp)) ** 2


def min_backtest_length(n_trials, sr_target_annual=1.0, periods_per_year=252):
    """Minimum Backtest Length (in years) before ``n_trials`` fake ``sr_target_annual``.

    The expected maximum Sharpe of ``n_trials`` skill-less i.i.d. strategies grows
    like ``Z_N / sqrt(T)``; solving for the horizon at which that expected fluke
    stays below the annualized target gives ``MinBTL = (Z_N / sr_target_annual)**2``
    years, where ``Z_N`` is the expected maximum of ``n_trials`` standard normals.
    One trial cannot be an overfit selection, so returns 0.0. Returns ``nan`` for a
    non-positive target.
    """
    if sr_target_annual <= 0:
        return float("nan")
    if n_trials is None or n_trials <= 1:
        return 0.0
    z_n = _expected_max_sr(n_trials, 1.0)  # expected max of n_trials standard normals
    return float((z_n / sr_target_annual) ** 2)


@dataclass
class PBOResult:
    """Result of a CSCV probability-of-backtest-overfitting computation."""
    pbo: float                      # probability of backtest overfitting, [0, 1]
    logits: list                    # per-split logit of the IS-best OOS relative rank
    performance_degradation: float  # OLS slope of OOS metric on IS metric (neg = overfit)
    prob_oos_loss: float            # fraction of splits where the IS-best loses OOS
    n_configs: int
    n_splits: int
    n_combinations: int


def _as_matrix(returns_matrix):
    """Coerce a DataFrame or dict[str, Series] of return streams to a clean
    (T x N) DataFrame aligned on a common index with NaN rows dropped."""
    df = returns_matrix if isinstance(returns_matrix, pd.DataFrame) else pd.DataFrame(returns_matrix)
    return df.dropna(how="any")


def _annualized_sharpe(rows, periods_per_year):
    """Per-column annualized Sharpe of a (rows x configs) array (0 where std==0)."""
    mu = rows.mean(axis=0)
    sd = rows.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mu / sd, 0.0)
    return sr * np.sqrt(periods_per_year)


def probability_of_backtest_overfitting(returns_matrix, n_splits=16, metric="sharpe",
                                        periods_per_year=252, max_combinations=20000, seed=0):
    """Probability of Backtest Overfitting (PBO) via CSCV (Bailey & Lopez de Prado).

    ``returns_matrix`` is a ``(T periods x N configs)`` DataFrame or a
    ``dict[str, Series]`` of N configuration return streams (e.g. from
    ``backtest_many`` over a parameter set). The timeline is split into ``n_splits``
    (even) contiguous blocks; over every ``C(n_splits, n_splits/2)`` in-sample /
    out-of-sample partition, the in-sample-best config's OOS relative rank is
    turned into a logit. ``pbo`` is the fraction of partitions where that logit is
    <= 0 — i.e. the IS-winner lands at or below the OOS median. A high PBO means
    the selection procedure is overfitting.

    Returns a :class:`PBOResult`. Raises ``ValueError`` for < 2 configs or an odd /
    too-large ``n_splits``.
    """
    if metric != "sharpe":
        raise ValueError(f"Unsupported metric {metric!r}; only 'sharpe' is supported.")
    df = _as_matrix(returns_matrix)
    T, N = df.shape
    if N < 2:
        raise ValueError(f"PBO requires at least 2 configurations; got {N}.")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(f"n_splits must be an even integer >= 2; got {n_splits}.")
    if n_splits > T:
        raise ValueError(f"n_splits ({n_splits}) exceeds the number of observations ({T}).")

    block = T // n_splits
    used = block * n_splits
    if used < T:
        logger.info("PBO: dropping %d trailing rows so %d splits divide evenly.", T - used, n_splits)
    values = df.values[:used]
    blocks = [values[s * block:(s + 1) * block] for s in range(n_splits)]

    all_combos = list(combinations(range(n_splits), n_splits // 2))
    if len(all_combos) > max_combinations:
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(all_combos), size=max_combinations, replace=False)
        combos = [all_combos[i] for i in picks]
        logger.info("PBO: %d combinations exceed cap %d; sampling %d (seed=%d).",
                    len(all_combos), max_combinations, max_combinations, seed)
    else:
        combos = all_combos

    logits, is_best, oos_best, n_loss = [], [], [], 0
    for is_blocks in combos:
        is_set = set(is_blocks)
        is_data = np.vstack([blocks[b] for b in range(n_splits) if b in is_set])
        oos_data = np.vstack([blocks[b] for b in range(n_splits) if b not in is_set])
        is_sr = _annualized_sharpe(is_data, periods_per_year)
        oos_sr = _annualized_sharpe(oos_data, periods_per_year)
        n_star = int(np.argmax(is_sr))
        # OOS relative rank of the IS-best (1 = worst .. N = best) -> logit.
        omega = float(stats.rankdata(oos_sr, method="average")[n_star]) / (N + 1)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        logits.append(float(np.log(omega / (1.0 - omega))))
        is_best.append(float(is_sr[n_star]))
        oos_best.append(float(oos_sr[n_star]))
        if oos_sr[n_star] < 0:
            n_loss += 1

    n_comb = len(combos)
    pbo = float(np.mean(np.asarray(logits) <= 0.0))
    xs, ys = np.asarray(is_best), np.asarray(oos_best)
    slope = float(np.polyfit(xs, ys, 1)[0]) if (len(xs) >= 2 and np.var(xs) > 0) else float("nan")
    return PBOResult(pbo=pbo, logits=logits, performance_degradation=slope,
                     prob_oos_loss=n_loss / n_comb, n_configs=N, n_splits=n_splits,
                     n_combinations=n_comb)


@dataclass
class OverfittingReport:
    """Plain-language overfitting verdict bundling PBO, deflated Sharpe and MinTRL."""
    pbo: Optional[float]
    deflated_sharpe: Optional[float]
    min_track_record_length: Optional[float]
    verdict: str                       # robust | caution | likely_overfit | inconclusive
    rationale: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "pbo": self.pbo,
            "deflated_sharpe": self.deflated_sharpe,
            "min_track_record_length": self.min_track_record_length,
            "verdict": self.verdict,
            "rationale": self.rationale,
            "evidence": self.evidence,
        }

    def to_text(self):
        lines = [f"Backtest honesty: {self.verdict.upper().replace('_', ' ')}",
                 f"  {self.rationale}"]
        if self.pbo is not None:
            lines.append(f"  PBO {self.pbo:.0%}")
        if self.deflated_sharpe is not None and np.isfinite(self.deflated_sharpe):
            lines.append(f"  Deflated Sharpe {self.deflated_sharpe:.2f}")
        if self.min_track_record_length is not None and np.isfinite(self.min_track_record_length):
            lines.append(f"  Min track record length {self.min_track_record_length:.0f} obs")
        return "\n".join(lines)


def _best_series(df, periods_per_year):
    """The full-sample annualized-Sharpe-best column — the config a user would keep."""
    sr = _annualized_sharpe(df.values, periods_per_year)
    return df.iloc[:, int(np.argmax(sr))]


def assess_overfitting(returns_matrix=None, returns=None, n_trials=None, n_splits=16,
                       periods_per_year=252, thresholds=None):
    """One-call overfitting verdict.

    Pass ``returns_matrix`` (the N tried configurations) to get PBO plus a deflated
    Sharpe / MinTRL on the full-sample-best config; and/or pass a single ``returns``
    track record (with ``n_trials``) to deflate that Sharpe directly. Returns an
    :class:`OverfittingReport` with a ``robust`` / ``caution`` / ``likely_overfit``
    / ``inconclusive`` verdict. Thresholds are documented heuristics, overridable
    via ``thresholds``; they are decision *aids*, not rules.
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    pbo = perf_degradation = prob_oos_loss = n_configs = None

    if returns_matrix is not None:
        df = _as_matrix(returns_matrix)
        n_configs = df.shape[1]
        if n_configs >= 2:
            res = probability_of_backtest_overfitting(
                df, n_splits=n_splits, periods_per_year=periods_per_year)
            pbo, perf_degradation, prob_oos_loss = res.pbo, res.performance_degradation, res.prob_oos_loss
        if returns is None and n_configs >= 1:
            returns = _best_series(df, periods_per_year)
        if n_trials is None:
            n_trials = n_configs

    dsr = mintrl = None
    if returns is not None:
        nt = n_trials if n_trials is not None else 1
        dsr = deflated_sharpe_ratio(returns, nt, periods_per_year)
        mintrl = min_track_record_length(returns, periods_per_year=periods_per_year)

    dsr_ok = dsr is not None and np.isfinite(dsr)
    if pbo is None and not dsr_ok:
        verdict = "inconclusive"
        rationale = ("Not enough information to assess overfitting: no trial matrix and "
                     "no usable track record.")
    elif pbo is not None:
        if pbo > th["pbo_overfit"]:
            verdict = "likely_overfit"
            rationale = (f"PBO {pbo:.0%} exceeds the {th['pbo_overfit']:.0%} heuristic — the "
                         f"in-sample-best configuration is more often than not at or below the "
                         f"out-of-sample median. Treat the selected result as likely overfit.")
        elif pbo <= th["pbo_robust"] and (not dsr_ok or dsr >= th["dsr_strong"]):
            verdict = "robust"
            extra = (f" and the deflated Sharpe {dsr:.2f} clears {th['dsr_strong']:.2f}"
                     if dsr_ok else "")
            rationale = (f"PBO {pbo:.0%} is within the {th['pbo_robust']:.0%} heuristic{extra}; "
                         f"the selection holds up out-of-sample on these diagnostics.")
        else:
            verdict = "caution"
            extra = f"; deflated Sharpe {dsr:.2f}" if dsr_ok else ""
            rationale = (f"PBO {pbo:.0%} sits between the robust ({th['pbo_robust']:.0%}) and "
                         f"overfit ({th['pbo_overfit']:.0%}) heuristics{extra}. Interpret with care.")
    else:
        if dsr >= th["dsr_strong"]:
            verdict = "robust"
            rationale = (f"Deflated Sharpe {dsr:.2f} clears the {th['dsr_strong']:.2f} heuristic "
                         f"after accounting for {n_trials} trial(s).")
        else:
            verdict = "caution"
            rationale = (f"Deflated Sharpe {dsr:.2f} is below the {th['dsr_strong']:.2f} heuristic "
                         f"after accounting for {n_trials} trial(s); the edge may not survive selection.")

    rationale += " (Thresholds are heuristics, not decision rules.)"
    evidence = {"pbo": pbo, "deflated_sharpe": dsr, "min_track_record_length": mintrl,
                "n_trials": n_trials, "n_configs": n_configs,
                "performance_degradation": perf_degradation, "prob_oos_loss": prob_oos_loss,
                "thresholds": th}
    return OverfittingReport(pbo=pbo, deflated_sharpe=dsr, min_track_record_length=mintrl,
                             verdict=verdict, rationale=rationale, evidence=evidence)


_VERDICT_STYLE = {
    "robust": ("#D1E7DD", "#0F5132", "&#10003;"),          # green / check
    "caution": ("#FFF3CD", "#664D03", "&#9888;"),          # amber / warning
    "likely_overfit": ("#F8D7DA", "#842029", "&#10007;"),  # red / cross
    "inconclusive": ("#E2E3E5", "#41464B", "&#8211;"),     # grey / dash
}


def overfitting_section(report):
    """Build a "Backtest Honesty" report section from an :class:`OverfittingReport`.

    Returns a section dict matching the ``data_quality_section`` shape (a
    ``table_html`` item of raw HTML) so it renders in the
    :func:`~quant_reporter.html_builder.generate_html_report` pipeline and can be
    embedded in the validation / combined reports.
    """
    bg, fg, icon = _VERDICT_STYLE.get(report.verdict, _VERDICT_STYLE["inconclusive"])
    rows = []
    if report.pbo is not None:
        rows.append(f"<li><strong>PBO:</strong> {report.pbo:.0%} probability of backtest overfitting.</li>")
    if report.deflated_sharpe is not None and np.isfinite(report.deflated_sharpe):
        rows.append(f"<li><strong>Deflated Sharpe:</strong> {report.deflated_sharpe:.2f}.</li>")
    if report.min_track_record_length is not None and np.isfinite(report.min_track_record_length):
        rows.append(f"<li><strong>Min track record length:</strong> "
                    f"{report.min_track_record_length:.0f} observations.</li>")
    degr = report.evidence.get("performance_degradation")
    if degr is not None and np.isfinite(degr):
        rows.append(f"<li><strong>Performance degradation:</strong> OOS-vs-IS slope {degr:.2f}.</li>")

    banner = (
        f'<div style="background:{bg};border-radius:8px;padding:12px 16px;color:{fg};">'
        f'<p style="margin:0 0 8px 0;"><strong>{icon} Verdict: '
        f'{escape(report.verdict.replace("_", " ").upper())}</strong></p>'
        f'<p style="margin:0 0 8px 0;">{escape(report.rationale)}</p>'
        f'<ul style="margin:0;padding-left:20px;">{"".join(rows)}</ul>'
        '</div>'
    )
    return {
        "title": "Backtest Honesty",
        "description": "Overfitting diagnostics (PBO / deflated Sharpe / MinTRL). "
                       "Thresholds are heuristics — read alongside the strategy logic.",
        "main_content": [
            {"title": "Overfitting Verdict", "type": "table_html", "data": banner}
        ],
    }
