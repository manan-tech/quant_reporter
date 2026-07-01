# test/test_overfitting.py
"""Backtest-honesty diagnostics (GH: backtest-honesty module).

PBO via CSCV, Minimum Track Record Length, Minimum Backtest Length, and a
one-call overfitting verdict. Pure/offline; known-answer + property tests.
Refs: Bailey & Lopez de Prado, Deflated Sharpe / Probability of Backtest
Overfitting / Pseudo-Mathematics.
"""
import math

import numpy as np
import pandas as pd
import pytest

from quant_reporter.overfitting import (
    min_track_record_length,
    min_backtest_length,
    probability_of_backtest_overfitting,
    PBOResult,
    assess_overfitting,
    OverfittingReport,
    overfitting_section,
)
from quant_reporter.html_builder import generate_html_report


def _series(mu, sigma, n, seed):
    return np.random.default_rng(seed).normal(mu, sigma, n)


def _noise_matrix(T, N, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 0.01, (T, N)),
                        columns=[f"c{i}" for i in range(N)])


def _dominant_matrix(T, N, seed):
    # One config has a real persistent edge (higher mean, same vol).
    rng = np.random.default_rng(seed)
    M = rng.normal(0.0, 0.01, (T, N))
    M[:, 0] += 0.004  # config c0 genuinely dominates in- and out-of-sample
    return pd.DataFrame(M, columns=[f"c{i}" for i in range(N)])


def _overfit_matrix(T, N, n_blocks, seed, drift=0.002, sigma=0.01):
    # Each config has a positive mean in the first half of the timeline and the
    # negative of it in the second half, with CONSTANT volatility (so Sharpe
    # actually distinguishes configs — scaling both mean and vol would cancel).
    # Higher-index configs have a larger swing, so whichever config looks best on
    # an in-sample block set is the worst on the complementary OOS set -> PBO -> 1.
    rng = np.random.default_rng(seed)
    block = T // n_blocks
    T = block * n_blocks
    M = rng.normal(0.0, sigma, (T, N))
    for c in range(N):
        m = (c + 1) * drift
        for b in range(n_blocks):
            M[b * block:(b + 1) * block, c] += m if b < n_blocks // 2 else -m
    return pd.DataFrame(M, columns=[f"c{i}" for i in range(N)])


# --------------------------------------------------------------------------
# MinBTL — minimum backtest length (years) before N trials fake a Sharpe
# --------------------------------------------------------------------------

def test_min_backtest_length_grows_with_trials():
    assert min_backtest_length(1000, sr_target_annual=1.0) > min_backtest_length(10, sr_target_annual=1.0) > 0


def test_min_backtest_length_shrinks_with_higher_target_sharpe():
    assert min_backtest_length(100, sr_target_annual=2.0) < min_backtest_length(100, sr_target_annual=1.0)


def test_min_backtest_length_single_trial_is_zero():
    # One trial cannot be an overfit selection.
    assert min_backtest_length(1, sr_target_annual=1.0) == 0.0


def test_min_backtest_length_nonpositive_target_is_nan():
    assert np.isnan(min_backtest_length(100, sr_target_annual=0.0))


# --------------------------------------------------------------------------
# MinTRL — observations needed for the Sharpe to be significant
# --------------------------------------------------------------------------

def test_min_track_record_length_positive_and_finite():
    r = _series(0.001, 0.01, 500, seed=1)  # per-period SR ~ 0.1
    trl = min_track_record_length(r, sr_target=0.0, prob=0.95)
    assert np.isfinite(trl) and trl > 0


def test_min_track_record_length_grows_with_confidence():
    r = _series(0.001, 0.01, 500, seed=2)
    assert min_track_record_length(r, prob=0.99) > min_track_record_length(r, prob=0.90)


def test_min_track_record_length_nan_when_target_unreachable():
    r = _series(0.001, 0.01, 500, seed=3)
    # A target annualized Sharpe far above the observed one cannot be reached.
    assert np.isnan(min_track_record_length(r, sr_target=50.0, prob=0.95))


# --------------------------------------------------------------------------
# PBO via CSCV — probability of backtest overfitting
# --------------------------------------------------------------------------

def test_pbo_noise_averages_near_half():
    # Selecting the IS-best among i.i.d. noise configs is ~a coin flip OOS. PBO is
    # noisy per-seed (std ~0.15), so average over seeds and check it centers ~0.5.
    vals = [probability_of_backtest_overfitting(_noise_matrix(600, 10, seed=s), n_splits=10).pbo
            for s in range(12)]
    assert isinstance(vals[0], float)
    assert 0.40 < float(np.mean(vals)) < 0.66


def test_pbo_low_when_a_config_genuinely_dominates():
    res = probability_of_backtest_overfitting(_dominant_matrix(600, 8, seed=11), n_splits=10)
    assert res.pbo < 0.30


def test_pbo_high_for_constructed_overfit():
    res = probability_of_backtest_overfitting(_overfit_matrix(600, 6, n_blocks=10, seed=12), n_splits=10)
    assert isinstance(res, PBOResult)
    assert res.pbo > 0.70


def test_pbo_result_fields_and_combination_count():
    res = probability_of_backtest_overfitting(_noise_matrix(400, 6, seed=13), n_splits=8)
    assert 0.0 <= res.pbo <= 1.0
    assert res.n_configs == 6 and res.n_splits == 8
    assert res.n_combinations == math.comb(8, 4)
    assert len(res.logits) == res.n_combinations
    assert 0.0 <= res.prob_oos_loss <= 1.0


def test_pbo_accepts_dict_and_dataframe_equivalently():
    df = _noise_matrix(400, 5, seed=14)
    as_dict = {c: df[c] for c in df.columns}
    r_df = probability_of_backtest_overfitting(df, n_splits=8)
    r_dict = probability_of_backtest_overfitting(as_dict, n_splits=8)
    assert r_df.pbo == pytest.approx(r_dict.pbo)


def test_pbo_raises_on_too_few_configs():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(400, 1, seed=15), n_splits=8)


def test_pbo_raises_on_odd_or_too_large_n_splits():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(400, 5, seed=16), n_splits=7)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(_noise_matrix(10, 5, seed=16), n_splits=16)


# --------------------------------------------------------------------------
# assess_overfitting — one-call verdict
# --------------------------------------------------------------------------

def test_assess_overfit_matrix_flags_likely_overfit():
    rep = assess_overfitting(returns_matrix=_overfit_matrix(600, 6, n_blocks=10, seed=12), n_splits=10)
    assert isinstance(rep, OverfittingReport)
    assert rep.verdict == "likely_overfit"
    assert rep.pbo is not None and rep.pbo > 0.5


def test_assess_dominant_matrix_not_flagged_overfit():
    rep = assess_overfitting(returns_matrix=_dominant_matrix(600, 8, seed=11), n_splits=10)
    assert rep.verdict != "likely_overfit"
    assert rep.pbo < 0.30


def test_assess_inconclusive_without_inputs():
    assert assess_overfitting().verdict == "inconclusive"


def test_assess_single_series_uses_deflated_sharpe():
    rep = assess_overfitting(returns=_series(0.001, 0.01, 500, seed=20), n_trials=50)
    assert rep.pbo is None
    assert rep.deflated_sharpe is not None and 0.0 <= rep.deflated_sharpe <= 1.0
    assert rep.verdict in {"robust", "caution", "inconclusive"}


def test_assess_report_to_dict_and_text():
    rep = assess_overfitting(returns_matrix=_noise_matrix(400, 6, seed=21), n_splits=8)
    d = rep.to_dict()
    assert {"pbo", "deflated_sharpe", "min_track_record_length", "verdict", "rationale"} <= set(d)
    assert isinstance(rep.to_text(), str) and rep.to_text()


def test_assess_custom_thresholds_flip_verdict():
    m = _noise_matrix(400, 6, seed=22)
    # An impossibly strict overfit threshold forces likely_overfit on ~0.5 PBO noise.
    rep = assess_overfitting(returns_matrix=m, n_splits=8, thresholds={"pbo_overfit": 0.0})
    assert rep.verdict == "likely_overfit"


# --------------------------------------------------------------------------
# overfitting_section — embeddable report section
# --------------------------------------------------------------------------

def test_overfitting_section_shape():
    rep = assess_overfitting(returns_matrix=_overfit_matrix(600, 6, n_blocks=10, seed=12), n_splits=10)
    section = overfitting_section(rep)
    assert section["title"] == "Backtest Honesty"
    item = section["main_content"][0]
    assert item["type"] == "table_html"
    assert "PBO" in item["data"] and "LIKELY OVERFIT" in item["data"].upper()


def test_overfitting_section_renders_via_html_builder(tmp_path):
    rep = assess_overfitting(returns_matrix=_noise_matrix(400, 6, seed=23), n_splits=8)
    out = tmp_path / "honesty.html"
    generate_html_report([overfitting_section(rep)], title="Honesty", filename=str(out))
    html = out.read_text(encoding="utf-8")
    assert "Backtest Honesty" in html
    assert rep.verdict.replace("_", " ") in html.lower()
