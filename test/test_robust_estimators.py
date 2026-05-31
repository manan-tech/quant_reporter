# test/test_robust_estimators.py
import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf as sk_ledoit_wolf

from quant_reporter.robust_estimators import ledoit_wolf_covariance
from conftest import make_synthetic_prices


def _returns(seed=0, n=300, cols=("AAA", "BBB", "CCC")):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    base = rng.normal(0, 0.01, (n, 1))
    data = {c: (base[:, 0] * (0.5 + 0.2 * i) + rng.normal(0, 0.008, n)) for i, c in enumerate(cols)}
    return pd.DataFrame(data, index=idx)


def test_lw_returns_expected_keys():
    out = ledoit_wolf_covariance(_returns())
    assert set(out) == {"cov_matrix", "shrinkage", "target", "sample_cov", "target_matrix"}


def test_lw_cov_is_symmetric_pd_and_labeled():
    r = _returns()
    out = ledoit_wolf_covariance(r)
    cov = out["cov_matrix"]
    assert list(cov.index) == list(r.columns) == list(cov.columns)
    assert np.allclose(cov.values, cov.values.T)
    assert (np.linalg.eigvalsh(cov.values) > 0).all()


def test_lw_shrinkage_in_unit_interval():
    out = ledoit_wolf_covariance(_returns())
    assert 0.0 <= out["shrinkage"] <= 1.0


def test_lw_annualization_matches_252_contract():
    r = _returns()
    ann = ledoit_wolf_covariance(r, periods_per_year=252)["cov_matrix"]
    daily = ledoit_wolf_covariance(r, periods_per_year=1)["cov_matrix"]
    assert np.allclose(ann.values, daily.values * 252)


def test_lw_identity_target_matches_sklearn():
    r = _returns()
    out = ledoit_wolf_covariance(r, target="identity", periods_per_year=1)
    sk_cov, sk_shrink = sk_ledoit_wolf(r.values)
    assert out["shrinkage"] == pytest.approx(sk_shrink, rel=1e-6)
    assert np.allclose(out["cov_matrix"].values, sk_cov, atol=1e-10)


def test_lw_explicit_delta_is_used():
    r = _returns()
    out = ledoit_wolf_covariance(r, delta=0.0, periods_per_year=1)
    # delta=0 => pure sample covariance (ddof handling aside), shrinkage reported as 0
    assert out["shrinkage"] == pytest.approx(0.0)
    assert np.allclose(out["cov_matrix"].values, out["sample_cov"].values)


def test_lw_constant_correlation_offdiag_uses_avg_corr():
    r = _returns()
    out = ledoit_wolf_covariance(r, target="constant_correlation", periods_per_year=1)
    F = out["target_matrix"].values
    S = out["sample_cov"].values
    # diagonal of target equals sample variances
    assert np.allclose(np.diag(F), np.diag(S))
    # off-diagonal target correlation is constant
    d = np.sqrt(np.diag(F))
    corr = F / np.outer(d, d)
    off = corr[~np.eye(len(d), dtype=bool)]
    assert np.allclose(off, off[0])
