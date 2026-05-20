"""W22-NC27 regression tests for LogisticNormalPredictor.

Per W22 Inspection 3 / docs/W22-INSPECTIONS-SYNTHESIS.md Tier-1.

The Logistic-Normal predictor is a drop-in alternative to DirichletPredictor
(which is actually exponential smoothing — see Inspection 3). It operates
in Aitchison log-ratio coordinates and is selected via the
``W22_NC27_PREDICTOR`` env var. Default behavior is unchanged.

These tests verify:
1. Round-trip identity: _forward then _inverse recovers the input simplex
2. Predict-at-zero-rate identity: with rate=0, prediction == prev
3. Predict-at-full-rate: with rate=2.0 (0.5 prefactor → 1.0), prediction == current
4. Output is on the simplex (sums to 1, all ≥ 0)
5. Dispatcher default is DirichletPredictor; flip via env var switches it
6. HONEST SCAR: at production anticipative rates ∈ [1.0, 2.0], LN is a
   coordinate-system change, not a Bayesian fix. The 2.8× accuracy gain
   from Inspection 3 was for the TRUE Dirichlet posterior (stateful
   running-mean α += obs) which can't be implemented under DirichletPredictor's
   stateless-static-method interface. The accuracy test here only verifies
   that LN is COMPARABLE to Dirichlet (no large regression), not strictly
   better. A future NC27-deep would add DirichletPosteriorPredictor for the
   actual structural fix.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.anticipatory_learning import (
    DirichletPredictor,
    LogisticNormalPredictor,
    _get_active_predictor,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Ensure the W22_NC27_PREDICTOR env var is not set unless a test sets it."""
    monkeypatch.delenv("W22_NC27_PREDICTOR", raising=False)


def test_round_trip_uniform_simplex():
    """_forward then _inverse recovers a uniform simplex within numerical tolerance."""
    w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    y = LogisticNormalPredictor._forward(w)
    w_back = LogisticNormalPredictor._inverse(y)
    np.testing.assert_allclose(w_back, w, atol=1e-10)


def test_round_trip_sparse_simplex():
    """_forward then _inverse recovers a sparse simplex (with EPS clipping)."""
    w = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
    y = LogisticNormalPredictor._forward(w)
    w_back = LogisticNormalPredictor._inverse(y)
    # EPS-clipped components recover near-zero, not exactly zero.
    np.testing.assert_allclose(w_back[:3], w[:3], atol=1e-6)
    assert w_back[3] < 1e-6 and w_back[4] < 1e-6
    # Total mass conserved.
    np.testing.assert_allclose(np.sum(w_back), 1.0, atol=1e-10)


def test_predict_zero_rate_returns_prev():
    """At anticipative_rate=0.0, the prediction equals prev (no movement)."""
    prev = np.array([0.5, 0.3, 0.2])
    current = np.array([0.1, 0.1, 0.8])
    pred = LogisticNormalPredictor.dirichlet_mean_prediction_vec(prev, current, anticipative_rate=0.0)
    np.testing.assert_allclose(pred, prev, atol=1e-9)


def test_predict_full_rate_returns_current():
    """At anticipative_rate=2.0 (0.5 prefactor → effective 1.0), prediction equals current."""
    prev = np.array([0.5, 0.3, 0.2])
    current = np.array([0.1, 0.1, 0.8])
    pred = LogisticNormalPredictor.dirichlet_mean_prediction_vec(prev, current, anticipative_rate=2.0)
    np.testing.assert_allclose(pred, current, atol=1e-9)


def test_predict_output_on_simplex():
    """Predicted vector sums to 1 and is non-negative."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        prev = rng.dirichlet(np.ones(5))
        current = rng.dirichlet(np.ones(5))
        pred = LogisticNormalPredictor.dirichlet_mean_prediction_vec(
            prev, current, anticipative_rate=rng.uniform(0, 2)
        )
        np.testing.assert_allclose(np.sum(pred), 1.0, atol=1e-9)
        assert np.all(pred >= 0.0)


def test_map_update_output_on_simplex():
    """Map update output sums to 1 and is non-negative."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        pred = rng.dirichlet(np.ones(5))
        obs = rng.dirichlet(np.ones(5))
        concentration = rng.uniform(0.1, 10.0)
        updated = LogisticNormalPredictor.dirichlet_mean_map_update(pred, obs, concentration)
        np.testing.assert_allclose(np.sum(updated), 1.0, atol=1e-9)
        assert np.all(updated >= 0.0)


def test_dispatcher_default_is_dirichlet():
    """With no env var set, the dispatcher returns DirichletPredictor."""
    assert _get_active_predictor() is DirichletPredictor


def test_dispatcher_env_var_switches(monkeypatch):
    """W22_NC27_PREDICTOR=logistic_normal switches the dispatcher."""
    monkeypatch.setenv("W22_NC27_PREDICTOR", "logistic_normal")
    assert _get_active_predictor() is LogisticNormalPredictor


def test_dispatcher_unknown_value_falls_back(monkeypatch):
    """Unknown env values fall back to DirichletPredictor (safe default)."""
    monkeypatch.setenv("W22_NC27_PREDICTOR", "some_unknown_predictor")
    assert _get_active_predictor() is DirichletPredictor


def test_logistic_normal_comparable_to_dirichlet_no_regression():
    """No-regression check: on 100 obs from Dirichlet(α=[5,3,2,1,1]),
    Logistic-Normal predictor's terminal L2 error to the TRUE mean is
    within 2× DirichletPredictor's error.

    HONEST SCAR: per the NC27 module docstring, at production anticipative
    rates ∈ [1.0, 2.0] both predictors track recent observations — neither
    is a true running-mean estimator. The Inspection 3 2.8× gain was for
    the stateful TRUE Dirichlet posterior, not for a logistic-normal swap.

    This test guards against a LARGE regression (factor-of-2 worse) which
    would suggest a bug in the LN implementation. It does NOT claim LN is
    strictly better than the legacy DirichletPredictor at production rates.

    Method:
    - Generate observations ~ Dirichlet(α_true)
    - Drive each predictor with sequential observations using
      dirichlet_mean_prediction_vec at rate=1.0 (same rate for both)
    - Measure L2 distance from final predicted mean to true mean
    """
    rng = np.random.default_rng(42)
    alpha_true = np.array([5.0, 3.0, 2.0, 1.0, 1.0])
    true_mean = alpha_true / np.sum(alpha_true)
    n_obs = 100
    observations = rng.dirichlet(alpha_true, size=n_obs)

    pred_dirichlet = observations[0].copy()
    pred_ln = observations[0].copy()
    for t in range(1, n_obs):
        pred_dirichlet = DirichletPredictor.dirichlet_mean_prediction_vec(
            pred_dirichlet, observations[t], anticipative_rate=1.0,
        )
        pred_ln = LogisticNormalPredictor.dirichlet_mean_prediction_vec(
            pred_ln, observations[t], anticipative_rate=1.0,
        )

    err_dirichlet = float(np.linalg.norm(pred_dirichlet - true_mean))
    err_ln = float(np.linalg.norm(pred_ln - true_mean))
    # Guard against major regression (2× worse).
    assert err_ln <= err_dirichlet * 2.0, (
        f"Logistic-Normal err={err_ln:.4f} exceeded Dirichlet err={err_dirichlet:.4f} by >2x; "
        "the LN implementation is significantly degraded — likely a bug."
    )
    # Sanity: both should be O(0.1) for this regime (not catastrophic).
    assert err_dirichlet < 0.5
    assert err_ln < 0.5
