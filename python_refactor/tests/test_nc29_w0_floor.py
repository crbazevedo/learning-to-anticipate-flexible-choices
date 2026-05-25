"""W22-NC29 structural-fix tests: w_0 floor in multi-horizon anticipation.

Per operator directive 2026-05-19 — degenerate behavior shouldn't happen.
Pre-NC29: when Σλ > 1.0, the soft normalization set w_0 → 0 (the current
observation got ZERO weight). Post-NC29: w_0 is hard-capped at ≥
``W22_NC29_MIN_W0`` (default 0.2).

These tests verify:
1. With Σλ ≤ max, behavior unchanged (no normalization fires)
2. With Σλ > max, all λ_h scale proportionally to preserve w_0 ≥ MIN_W0
3. The min_w0 floor can be tuned via env var
4. Empty/edge cases handled
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Need to instantiate; the simplest path is to call the method directly
# via a minimal mock that bypasses the heavy parent __init__.
from src.algorithms.multi_horizon_anticipatory import MultiHorizonAnticipatoryLearning


def _make_instance():
    """Create a MultiHorizonAnticipatoryLearning without running full __init__."""
    return MultiHorizonAnticipatoryLearning.__new__(MultiHorizonAnticipatoryLearning)


def test_no_cap_when_lambda_sum_under_max():
    """Σλ = 0.6 ≤ 0.8 (default cap) → behavior unchanged."""
    inst = _make_instance()
    z_t = np.array([1.0, 1.0])
    z_pred = [np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    lambdas = [0.3, 0.3]  # Σ = 0.6
    result = inst.apply_anticipatory_learning_rule(z_t, z_pred, lambdas)
    # Expected: 0.4 * [1,1] + 0.3 * [2,2] + 0.3 * [3,3] = [0.4 + 0.6 + 0.9, ...] = [1.9, 1.9]
    np.testing.assert_allclose(result, [1.9, 1.9])


def test_cap_fires_when_lambda_sum_above_max():
    """Σλ = 1.5 > 0.8 → scale all λ by 0.8/1.5 to preserve w_0 = 0.2."""
    inst = _make_instance()
    z_t = np.array([1.0, 1.0])
    z_pred = [np.array([2.0, 2.0]), np.array([3.0, 3.0])]
    lambdas = [0.7, 0.8]  # Σ = 1.5
    result = inst.apply_anticipatory_learning_rule(z_t, z_pred, lambdas)
    # After cap: λ scaled by 0.8/1.5; new λ = [0.7*8/15, 0.8*8/15] = [0.3733..., 0.4266...]
    # Σ = 0.8; w_0 = 0.2
    # result = 0.2*1 + 0.3733*2 + 0.4266*3 = 0.2 + 0.7466 + 1.28 = 2.2266
    expected = 0.2 * 1.0 + (0.7 * 0.8 / 1.5) * 2.0 + (0.8 * 0.8 / 1.5) * 3.0
    np.testing.assert_allclose(result, [expected, expected], rtol=1e-10)


def test_w0_floor_never_violated():
    """Across many Σλ values, w_0 = 1 - Σλ_post is always ≥ 0.2."""
    inst = _make_instance()
    z_t = np.array([0.0, 0.0])
    z_pred = [np.array([1.0, 1.0])]  # single horizon
    for lam in np.linspace(0.0, 5.0, 30):
        result = inst.apply_anticipatory_learning_rule(z_t, z_pred, [lam])
        # Implied w_0 = 1 - (result[0] / 1.0) since z_pred[0] = [1,1]
        # result = w_0 * 0 + (1-w_0) * 1 = 1 - w_0
        implied_w0 = 1.0 - float(result[0])
        # MIN_W0 default = 0.2
        assert implied_w0 >= 0.2 - 1e-10, (
            f"w_0 floor violated: λ={lam:.3f}, implied w_0={implied_w0:.4f} (want ≥ 0.2)"
        )


def test_env_var_controls_floor(monkeypatch):
    """W22_NC29_MIN_W0 env var changes the floor."""
    inst = _make_instance()
    z_t = np.array([0.0, 0.0])
    z_pred = [np.array([1.0, 1.0])]
    # Set floor to 0.5
    monkeypatch.setenv("W22_NC29_MIN_W0", "0.5")
    result = inst.apply_anticipatory_learning_rule(z_t, z_pred, [10.0])
    implied_w0 = 1.0 - float(result[0])
    np.testing.assert_allclose(implied_w0, 0.5, atol=1e-10)


def test_zero_min_w0_recovers_original_behavior(monkeypatch):
    """With W22_NC29_MIN_W0=0.0, the cap is at Σλ ≤ 1.0 (original behavior)."""
    inst = _make_instance()
    z_t = np.array([0.0, 0.0])
    z_pred = [np.array([1.0, 1.0])]
    monkeypatch.setenv("W22_NC29_MIN_W0", "0.0")
    # Σλ = 5.0 > 1.0 → cap at 1.0 → w_0 = 0
    result = inst.apply_anticipatory_learning_rule(z_t, z_pred, [5.0])
    implied_w0 = 1.0 - float(result[0])
    np.testing.assert_allclose(implied_w0, 0.0, atol=1e-10)


def test_empty_predicted_states_returns_current():
    """No future horizons → identity (z_t returned unchanged)."""
    inst = _make_instance()
    z_t = np.array([1.0, 2.0])
    result = inst.apply_anticipatory_learning_rule(z_t, [], [])
    np.testing.assert_allclose(result, z_t)
    assert result is not z_t  # should be a copy
