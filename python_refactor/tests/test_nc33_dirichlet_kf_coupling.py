"""W22-NC33 regression tests for Dirichlet ↔ KF coupling.

Per operator directive 2026-05-20 + W22-NEXT-STEPS-NC32-36.md Section C.

Tests:
1. Dirichlet posterior variance closed-form matches manual calculation
2. baseline_posterior_variance matches uniform-prior identity
3. scale_Q tightens with high concentration (low variance → scale = 1)
4. scale_Q grows with low concentration (high variance → scale > 1)
5. scale_Q capped at max_scale
6. residual → increment monotonic with magnitude
7. residual → increment clamped to [min, max]
8. coupled_predict_update_cycle returns all expected keys
9. coupled cycle: high residual → high concentration_increment + tighter posterior
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.dirichlet_kf_coupling import (
    baseline_posterior_variance,
    coupled_predict_update_cycle,
    dirichlet_posterior_precision,
    dirichlet_posterior_variance_max,
    kf_residual_to_dirichlet_concentration_increment,
    scale_Q_by_dirichlet_posterior,
)
from src.algorithms.anticipatory_learning import DirichletPosteriorPredictor


def test_dirichlet_posterior_variance_max_known_value():
    """For α = [1, 1, 1] (uniform Dirichlet on 3-simplex):
    α_sum = 3
    Var(X_i) = 1 · 2 / (9 · 4) = 2/36 = 1/18 ≈ 0.0556
    """
    alpha = np.array([1.0, 1.0, 1.0])
    var = dirichlet_posterior_variance_max(alpha)
    expected = 1.0 / 18.0
    np.testing.assert_allclose(var, expected, atol=1e-10)


def test_dirichlet_posterior_variance_max_asymmetric():
    """For α = [5, 3, 2]: α_sum=10
    Var(X_0) = 5·5/(100·11) = 25/1100 = 0.02273
    Var(X_1) = 3·7/(100·11) = 21/1100 = 0.01909
    Var(X_2) = 2·8/(100·11) = 16/1100 = 0.01455
    max = 0.02273
    """
    alpha = np.array([5.0, 3.0, 2.0])
    var = dirichlet_posterior_variance_max(alpha)
    np.testing.assert_allclose(var, 25.0/1100.0, atol=1e-10)


def test_dirichlet_posterior_precision_is_alpha_sum():
    alpha = np.array([2.5, 1.5, 1.0])
    assert dirichlet_posterior_precision(alpha) == 5.0


def test_baseline_posterior_variance_matches_uniform_dirichlet():
    """For d=3, alpha_prior=1.0: alpha_sum=3
    Var(X_i) = (d-1)/(d²(d·α+1)) = 2/(9·4) = 2/36 = 0.0556"""
    d = 3
    var = baseline_posterior_variance(d, alpha_prior=1.0)
    np.testing.assert_allclose(var, 2.0/36.0, atol=1e-10)


def test_scale_Q_at_uniform_prior_is_near_2():
    """At α = uniform-prior values, posterior variance ≈ baseline → scale ≈ 2."""
    d = 5
    alpha = np.ones(d) * 0.5  # Jeffreys prior
    Q_base = 1.0
    scaled = scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior=0.5)
    np.testing.assert_allclose(scaled, 2.0, atol=1e-10)


def test_scale_Q_with_high_concentration_returns_low_scale():
    """Tight posterior (large α_sum) → variance → 0 → scale → 1."""
    d = 5
    alpha = np.ones(d) * 100.0  # very tight
    Q_base = 1.0
    scaled = scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior=0.5)
    assert 1.0 <= scaled < 2.0  # should be very close to 1.0


def test_scale_Q_with_diffuse_posterior_can_exceed_2():
    """If concentration < uniform-prior baseline, variance > baseline → scale > 2."""
    d = 5
    # Posterior with α even smaller than prior → very diffuse
    alpha = np.ones(d) * 0.01  # almost no information
    Q_base = 1.0
    scaled = scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior=0.5)
    assert scaled > 2.0


def test_scale_Q_capped_at_max():
    """Force scale beyond max via alpha_prior=1.0 baseline (tighter than 0.5
    so the synthetic posterior with extreme alpha can exceed any max_scale)."""
    d = 5
    # alpha_prior=1.0 → baseline_var = 4/(25 * 6) = 0.0267
    # alpha = uniform near zero → posterior_var → ~0.16 (per scale formula)
    # scale = 1 + 0.16/0.0267 ≈ 7 → caps at max_scale=5
    alpha = np.ones(d) * 1e-10
    Q_base = 1.0
    scaled = scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior=1.0, max_scale=5.0)
    assert scaled == 5.0


def test_scale_Q_preserves_matrix_shape():
    d = 5
    alpha = np.ones(d) * 0.5
    Q_base = np.eye(4) * 0.01
    scaled = scale_Q_by_dirichlet_posterior(Q_base, alpha, alpha_prior=0.5)
    assert scaled.shape == (4, 4)
    np.testing.assert_allclose(scaled, Q_base * 2.0, atol=1e-10)


def test_residual_to_increment_monotonic():
    """Larger residual magnitude → larger concentration_increment."""
    small = kf_residual_to_dirichlet_concentration_increment(0.1, baseline_residual=1.0)
    medium = kf_residual_to_dirichlet_concentration_increment(0.5, baseline_residual=1.0)
    large = kf_residual_to_dirichlet_concentration_increment(2.0, baseline_residual=1.0)
    assert small < medium < large


def test_residual_to_increment_clamped():
    """Beyond max_increment, output is capped."""
    too_large = kf_residual_to_dirichlet_concentration_increment(
        100.0, baseline_residual=1.0, max_increment=5.0
    )
    assert too_large == 5.0
    too_small = kf_residual_to_dirichlet_concentration_increment(
        1e-8, baseline_residual=1.0, min_increment=0.1
    )
    assert too_small == 0.1


def test_residual_to_increment_vector_uses_l2_norm():
    """Vector residual → uses L2 norm."""
    residual_vec = np.array([3.0, 4.0])  # L2 = 5
    increment = kf_residual_to_dirichlet_concentration_increment(
        residual_vec, baseline_residual=5.0, max_increment=10.0,
    )
    np.testing.assert_allclose(increment, 1.0, atol=1e-10)


class _FakeKalmanState:
    def __init__(self, Q=None):
        self.Q = Q if Q is not None else 0.01 * np.eye(4)


def test_coupled_cycle_returns_all_keys():
    """coupled_predict_update_cycle returns the documented dict shape."""
    d = 4
    predictor = DirichletPosteriorPredictor(d, alpha_prior=0.5)
    kf = _FakeKalmanState()
    portfolio_obs = np.array([0.3, 0.3, 0.2, 0.2])
    result = coupled_predict_update_cycle(
        portfolio_obs=portfolio_obs,
        objective_obs=np.array([0.005, 0.002]),
        objective_pred=np.array([0.004, 0.003]),
        dirichlet_predictor=predictor,
        kalman_state=kf,
    )
    required = {
        "residual_magnitude",
        "concentration_increment",
        "posterior_mean",
        "effective_Q",
        "dirichlet_precision",
        "dirichlet_max_variance",
    }
    assert required.issubset(result.keys())


def test_coupled_cycle_high_residual_increases_increment():
    """Large residual → large concentration_increment → faster Dirichlet update.

    Use residual_baseline = 0.001 so portfolio-scale residuals span the
    [min, max] increment range non-trivially.
    """
    d = 4
    predictor_small = DirichletPosteriorPredictor(d, alpha_prior=0.5)
    predictor_large = DirichletPosteriorPredictor(d, alpha_prior=0.5)
    kf = _FakeKalmanState()
    portfolio_obs = np.array([0.4, 0.3, 0.2, 0.1])
    res_small = coupled_predict_update_cycle(
        portfolio_obs=portfolio_obs,
        objective_obs=np.array([0.005, 0.002]),
        objective_pred=np.array([0.0049, 0.0021]),  # ~1.4e-4 residual
        dirichlet_predictor=predictor_small,
        kalman_state=kf,
        residual_baseline=0.001,
    )
    res_large = coupled_predict_update_cycle(
        portfolio_obs=portfolio_obs,
        objective_obs=np.array([0.005, 0.002]),
        objective_pred=np.array([0.020, 0.020]),  # ~0.023 residual
        dirichlet_predictor=predictor_large,
        kalman_state=kf,
        residual_baseline=0.001,
    )
    assert res_large["concentration_increment"] > res_small["concentration_increment"]


def test_coupled_cycle_updates_posterior_mean_toward_obs():
    """After cycle, posterior_mean should be closer to portfolio_obs than uniform prior."""
    d = 4
    predictor = DirichletPosteriorPredictor(d, alpha_prior=0.5)
    kf = _FakeKalmanState()
    portfolio_obs = np.array([0.7, 0.1, 0.1, 0.1])  # concentrated
    uniform_prior = np.ones(d) / d
    result = coupled_predict_update_cycle(
        portfolio_obs=portfolio_obs,
        objective_obs=np.array([0.005, 0.002]),
        objective_pred=np.array([0.004, 0.003]),
        dirichlet_predictor=predictor,
        kalman_state=kf,
    )
    posterior = result["posterior_mean"]
    # Distance to obs should be < distance to uniform prior
    dist_to_obs = float(np.linalg.norm(posterior - portfolio_obs))
    dist_to_prior = float(np.linalg.norm(uniform_prior - portfolio_obs))
    assert dist_to_obs < dist_to_prior
