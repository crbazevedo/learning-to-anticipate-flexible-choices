"""W22-NC27-deep regression tests for DirichletPosteriorPredictor.

Per W22 Inspection 3: this stateful Bayesian posterior should achieve the
2.8× accuracy gain over the exponential-smoothing DirichletPredictor that
Inspection 3 quantified on synthetic Dirichlet-generated data.

These tests verify:
1. Construction and prior initialization
2. observe() updates α correctly: α_{t+1} = α_t + concentration · obs
3. predict_mean() = α / Σα
4. predict_variance() matches the closed-form Dirichlet variance
5. Reset functionality
6. Accuracy claim: terminal L2 error to true Dirichlet mean is
   SIGNIFICANTLY BELOW the legacy DirichletPredictor on the
   Inspection-3 scenario (n=100 obs from Dirichlet(α=[5,3,2,1,1]))
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.anticipatory_learning import (
    DirichletPredictor,
    DirichletPosteriorPredictor,
)


def test_construction_uniform_prior():
    """alpha_prior=1.0 gives uniform Bayes-Laplace prior."""
    p = DirichletPosteriorPredictor(d=5, alpha_prior=1.0)
    np.testing.assert_allclose(p.alpha, np.ones(5))
    np.testing.assert_allclose(p.predict_mean(), np.ones(5) / 5.0)


def test_construction_jeffreys_prior():
    """alpha_prior=0.5 gives Jeffreys-style prior."""
    p = DirichletPosteriorPredictor(d=4, alpha_prior=0.5)
    np.testing.assert_allclose(p.alpha, 0.5 * np.ones(4))
    np.testing.assert_allclose(p.predict_mean(), np.ones(4) / 4.0)


def test_observe_updates_alpha():
    """One observation: α_1 = α_0 + obs."""
    p = DirichletPosteriorPredictor(d=3, alpha_prior=0.5)
    obs = np.array([0.5, 0.3, 0.2])
    p.observe(obs)
    np.testing.assert_allclose(p.alpha, np.array([1.0, 0.8, 0.7]))


def test_observe_concentration_increment():
    """concentration_increment scales the observation contribution."""
    p = DirichletPosteriorPredictor(d=3, alpha_prior=0.5)
    p.observe(np.array([0.5, 0.3, 0.2]), concentration_increment=2.0)
    np.testing.assert_allclose(p.alpha, np.array([1.5, 1.1, 0.9]))


def test_observe_and_predict_returns_posterior_mean():
    """observe_and_predict equals observe followed by predict_mean."""
    p1 = DirichletPosteriorPredictor(d=4)
    p2 = DirichletPosteriorPredictor(d=4)
    obs = np.array([0.4, 0.3, 0.2, 0.1])
    mean_via_combo = p1.observe_and_predict(obs)
    p2.observe(obs)
    mean_via_separate = p2.predict_mean()
    np.testing.assert_allclose(mean_via_combo, mean_via_separate)


def test_predict_variance_matches_dirichlet_formula():
    """Var(X_i) = α_i (Σα − α_i) / (Σα² (Σα + 1))."""
    p = DirichletPosteriorPredictor(d=3, alpha_prior=1.0)
    p.observe(np.array([1.0, 0.5, 0.5]))  # alpha = [2, 1.5, 1.5]
    alpha_sum = np.sum(p.alpha)
    expected_var = p.alpha * (alpha_sum - p.alpha) / (alpha_sum ** 2 * (alpha_sum + 1.0))
    np.testing.assert_allclose(p.predict_variance(), expected_var)


def test_reset_restores_prior():
    """reset() restores α to its prior state."""
    p = DirichletPosteriorPredictor(d=4, alpha_prior=0.5)
    for _ in range(10):
        p.observe(np.array([0.25, 0.25, 0.25, 0.25]))
    assert p.n_observations == 10
    p.reset()
    assert p.n_observations == 0
    np.testing.assert_allclose(p.alpha, 0.5 * np.ones(4))


def test_inspection_3_accuracy_gain_vs_dirichlet_predictor():
    """Per W22 Inspection 3: on 100 obs from Dirichlet(α=[5,3,2,1,1]),
    the stateful TRUE Dirichlet posterior achieves L2 error 0.032 vs
    the exponential-smoothing DirichletPredictor's 0.089 — 2.8× gain.

    Test: same scenario, assert DirichletPosteriorPredictor's terminal
    L2 error is at most 0.06 (loose gate; Inspection 3 saw 0.032 with
    a different RNG seed). Also assert it's STRICTLY better than the
    legacy DirichletPredictor.
    """
    rng = np.random.default_rng(42)
    alpha_true = np.array([5.0, 3.0, 2.0, 1.0, 1.0])
    true_mean = alpha_true / np.sum(alpha_true)
    n_obs = 100
    observations = rng.dirichlet(alpha_true, size=n_obs)

    # Legacy DirichletPredictor (exponential smoothing, rate=1.0)
    pred_legacy = observations[0].copy()
    for t in range(1, n_obs):
        pred_legacy = DirichletPredictor.dirichlet_mean_prediction_vec(
            pred_legacy, observations[t], anticipative_rate=1.0,
        )
    err_legacy = float(np.linalg.norm(pred_legacy - true_mean))

    # NC27-deep: stateful Bayesian posterior
    posterior = DirichletPosteriorPredictor(d=5, alpha_prior=0.5)
    for obs in observations:
        posterior.observe(obs)
    pred_posterior = posterior.predict_mean()
    err_posterior = float(np.linalg.norm(pred_posterior - true_mean))

    # Inspection 3 measured 2.8× gap; we conservatively assert posterior < legacy
    # AND posterior < 0.06 absolute error
    assert err_posterior < err_legacy, (
        f"Stateful posterior err={err_posterior:.4f} should be BELOW "
        f"legacy err={err_legacy:.4f}; Inspection-3 expectation broken."
    )
    assert err_posterior < 0.06, (
        f"Stateful posterior err={err_posterior:.4f} should be < 0.06; "
        f"Inspection 3 saw 0.032 with this scenario."
    )
    # Also assert at least a 2× gap (the Inspection-3 finding was 2.8×)
    assert err_legacy >= 2.0 * err_posterior, (
        f"Gap legacy/posterior = {err_legacy / err_posterior:.2f}× should be ≥ 2×; "
        f"Inspection 3 saw 2.8×."
    )


def test_dimension_mismatch_raises():
    p = DirichletPosteriorPredictor(d=4)
    import pytest
    with pytest.raises(ValueError):
        p.observe(np.array([0.5, 0.5]))  # dim 2 ≠ 4


def test_posterior_mean_is_on_simplex():
    """After any number of observations, predict_mean() is on the simplex."""
    rng = np.random.default_rng(1)
    p = DirichletPosteriorPredictor(d=5)
    for _ in range(20):
        obs = rng.dirichlet(np.ones(5))
        mean = p.observe_and_predict(obs)
        np.testing.assert_allclose(np.sum(mean), 1.0, atol=1e-12)
        assert np.all(mean >= 0.0)


def test_posterior_variance_decreases_with_observations():
    """Variance shrinks as posterior gets more confident."""
    p = DirichletPosteriorPredictor(d=4, alpha_prior=0.5)
    var_initial_max = float(np.max(p.predict_variance()))
    for _ in range(50):
        # Observe a near-uniform sample
        p.observe(np.array([0.25, 0.25, 0.25, 0.25]))
    var_after_max = float(np.max(p.predict_variance()))
    assert var_after_max < var_initial_max, (
        f"Posterior variance should shrink: initial max={var_initial_max:.6f}, "
        f"after 50 obs max={var_after_max:.6f}"
    )
