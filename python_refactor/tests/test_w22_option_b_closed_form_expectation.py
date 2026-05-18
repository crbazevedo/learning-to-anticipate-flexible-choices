"""W22 Option B: tests for use_closed_form_expectation flag in
compute_oos_efhv.

True closed-form expected Ŝ via per-portfolio Black-Scholes-style
truncated-mean call (E[max(0, ROI-K)]) and put (E[max(0, K-risk)])
pricing on (ROI, risk) modeled as independent Gaussians, aggregated
by SUMMING per-portfolio E[HV_i] (no Pareto overlap correction →
acknowledged over-estimate).

Formulas (X ~ N(μ, σ²)):
  E[max(0, X - K)] = (μ - K) Φ((μ-K)/σ) + σ φ((μ-K)/σ)   [BS call]
  E[max(0, K - X)] = (K - μ) Φ((K-μ)/σ) + σ φ((K-μ)/σ)   [BS put]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as _stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.oos_evaluator import compute_oos_efhv


def _make_oos_returns(n_days: int = 100, n_assets: int = 5,
                       seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.normal(0.001, 0.01, (n_days, n_assets))
    return pd.DataFrame(arr, columns=[f"a{i}" for i in range(n_assets)])


def _make_pareto_weights(n_portfolios: int = 5, n_assets: int = 5,
                          seed: int = 42) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_portfolios):
        w = rng.dirichlet(np.ones(n_assets))
        weights.append(w)
    return weights


class TestFlagDefaultsFalse:
    """Default behavior: use_closed_form_expectation=False → MC bootstrap."""

    def test_default_uses_mc(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result = compute_oos_efhv(weights, oos, n_samples=10,
                                    rng=np.random.default_rng(42))
        assert len(result["efhv_samples"]) == 10  # MC produced 10 samples


class TestFlagOnDeterministic:
    """Closed-form expectation is deterministic given inputs."""

    def test_deterministic_across_rng_seeds(self):
        oos = _make_oos_returns()
        weights = _make_pareto_weights()
        result_a = compute_oos_efhv(weights, oos,
                                      use_closed_form_expectation=True,
                                      rng=np.random.default_rng(1))
        result_b = compute_oos_efhv(weights, oos,
                                      use_closed_form_expectation=True,
                                      rng=np.random.default_rng(999))
        # Identical Ŝ across different RNG seeds (deterministic)
        assert result_a["efhv_mean"] == pytest.approx(result_b["efhv_mean"], abs=0)
        # std=0 because deterministic
        assert result_a["efhv_std"] == 0.0
        assert len(result_a["efhv_samples"]) == 1


class TestManualBlackScholesMatch:
    """Sanity check: single-portfolio closed-form Ŝ equals manual BS calc."""

    def test_single_portfolio_matches_manual_formula(self):
        """One portfolio, hand-compute the BS truncated means, verify."""
        # 2-day OOS with controlled mean and variance.
        # Single asset, single portfolio with weight=1 on that asset.
        oos = pd.DataFrame(np.array([[0.01], [0.02], [0.015], [0.005]]),
                             columns=["a0"])
        weights = [np.array([1.0])]
        z_ref = (0.5, 0.0)  # risk_max=0.5, return_min=0.0

        result = compute_oos_efhv(weights, oos,
                                    use_closed_form_expectation=True,
                                    z_ref=z_ref)

        # Manual: μ̂ = mean of [0.01, 0.02, 0.015, 0.005] = 0.0125
        # Σ̂ = sample variance (ddof=1) = var([...]) ≈ 4.166e-5
        # n_days = 4
        # mu_i (portfolio mean) = 0.0125
        # sigma2_i = 4.166e-5 (variance is u^T Σ u with u=[1])
        # var_roi = sigma2_i / 4 = 1.0417e-5 → sd_roi ≈ 0.003227
        # var_risk = 2 * sigma2_i^2 / 3 ≈ 1.157e-9 → sd_risk ≈ 3.401e-5
        # call: d = (0.0125 - 0) / 0.003227 ≈ 3.873 → call ≈ 0.0125 * 1.0 + 0 ≈ 0.0125
        # put: d = (0.5 - 4.166e-5) / 3.401e-5 ≈ huge → put ≈ 0.4999 * 1.0 + 0 ≈ 0.5
        # E[HV] ≈ 0.0125 * 0.5 = 0.00625

        # Compute exact reference using numpy
        mu_hat = oos.values.mean(axis=0)
        Sigma_hat = np.cov(oos.values, rowvar=False, ddof=1)
        # For 1D, np.cov returns a 0-d scalar; coerce
        if Sigma_hat.ndim == 0:
            Sigma_hat = np.array([[float(Sigma_hat)]])
        w = weights[0]
        mu_i = float(mu_hat @ w)
        sigma2_i = float(w @ Sigma_hat @ w)
        var_roi = sigma2_i / 4
        var_risk = 2 * sigma2_i * sigma2_i / 3
        sd_roi = np.sqrt(var_roi)
        sd_risk = np.sqrt(var_risk)
        d_call = (mu_i - 0.0) / sd_roi
        d_put = (0.5 - sigma2_i) / sd_risk
        call = (mu_i - 0.0) * _stats.norm.cdf(d_call) + sd_roi * _stats.norm.pdf(d_call)
        put = (0.5 - sigma2_i) * _stats.norm.cdf(d_put) + sd_risk * _stats.norm.pdf(d_put)
        expected = call * put

        assert result["efhv_mean"] == pytest.approx(expected, rel=1e-12)


class TestInTheMoneyNonzero:
    """When portfolio has positive ROI and low risk, E[HV] > 0."""

    def test_inthemoney_returns_positive(self):
        oos = _make_oos_returns(n_days=50, seed=1)
        weights = _make_pareto_weights(n_portfolios=3, seed=1)
        result = compute_oos_efhv(weights, oos,
                                    use_closed_form_expectation=True,
                                    z_ref=(0.5, -0.5))  # generous z_ref
        assert result["efhv_mean"] > 0.0


class TestOutOfMoneyDegenerate:
    """When z_ref dominates all portfolios, E[HV] approaches 0."""

    def test_extreme_zref_makes_ehv_tiny(self):
        # Use a very TIGHT z_ref so no portfolio dominates it
        oos = _make_oos_returns(n_days=50, seed=1)
        weights = _make_pareto_weights(n_portfolios=3, seed=1)
        # z_ref = (0.0001, 1.0) means risk_max is very small AND
        # return_min is huge — portfolios with normal returns (~0.001)
        # can't dominate.
        result = compute_oos_efhv(weights, oos,
                                    use_closed_form_expectation=True,
                                    z_ref=(0.0001, 1.0))
        # Should be tiny (call term collapses near 0)
        assert result["efhv_mean"] < 1e-3
