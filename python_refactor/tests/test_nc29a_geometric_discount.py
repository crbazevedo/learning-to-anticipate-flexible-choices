"""W22-NC29a structural-fix tests: geometric γ^h discount in λ^H.

Per operator directive 2026-05-19 + W22 Inspection 5: replace the flat
(1/(H-1)) prefactor in calculate_multi_horizon_lambda_rates with a true
geometric discount γ^h. Default γ = 0.9; tunable via W22_NC29A_GAMMA.

These tests verify:
1. λ^H decays with h (when γ < 1) — fixes the "no real discount" degeneracy
2. γ default = 0.9 unless env var overrides
3. γ env var tunes the decay rate (γ=0.5 → faster decay; γ=0.99 → near-flat)
4. λ^H is still clamped to [0, 0.5] per the existing safety guard
5. With TIP saturated (entropy ≈ 0), λ^H_h decays cleanly as γ^h
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.multi_horizon_anticipatory import MultiHorizonAnticipatoryLearning


class _FakeTIP:
    """Minimal TIP calculator: returns a fixed TIP regardless of inputs."""

    def __init__(self, tip_value: float = 0.1):
        self.tip_value = tip_value

    def calculate_tip(self, *args, **kwargs):
        return self.tip_value

    @staticmethod
    def binary_entropy(p):
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    def reset_history(self):
        pass


def _make_instance_with_tip(tip_value: float = 0.1):
    """Build a MultiHorizonAnticipatoryLearning bypassing __init__ for a unit test."""
    inst = MultiHorizonAnticipatoryLearning.__new__(MultiHorizonAnticipatoryLearning)
    inst.tip_calculator = _FakeTIP(tip_value)
    inst.use_v2_anticipative_rate = False
    inst._lambda_trace_rows = []  # the calculate function appends here
    # _calculate_tip_for_horizon needs to be patched too (it builds predicted
    # solutions and calls tip_calculator.calculate_tip). For unit testing,
    # monkey-patch to return the fixed value directly.
    inst._calculate_tip_for_horizon = lambda _sol, _h: tip_value
    # Lambda^K returns (0.0, 'mocked') so the lambda_combined formula uses
    # only the λ^H arm (we're testing the discount shape, not the K-arm).
    inst._compute_lambda_k_with_branch = lambda *args, **kwargs: (0.0, "mocked")
    return inst


def test_lambda_h_decays_with_horizon_default_gamma():
    """With γ=0.9 (default), λ^H_h ∝ γ^h, so λ^H_h_(h+1) < λ^H_h_(h)."""
    inst = _make_instance_with_tip(tip_value=0.1)  # near-decisive prediction
    # Need a solution stub for the function signature
    sol = type("S", (), {})()
    sol.P = type("P", (), {})()
    sol.P.ROI = 0.0
    sol.P.risk = 0.0
    sol.P.kalman_state = None
    rates = inst.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=4)
    # rates is the COMBINED λ. Since λ^K = 0 (mocked), combined = 0.5 * λ^H.
    # So combined should still decay monotonically with h.
    assert len(rates) == 3  # h = 1, 2, 3
    assert rates[0] > rates[1] > rates[2], (
        f"λ should decay: {rates}"
    )


def test_gamma_env_var_controls_decay(monkeypatch):
    """Larger γ → slower decay; smaller γ → faster decay."""
    sol = type("S", (), {})()
    sol.P = type("P", (), {})()
    sol.P.ROI = 0.0
    sol.P.risk = 0.0
    sol.P.kalman_state = None

    # γ = 0.5: aggressive decay
    monkeypatch.setenv("W22_NC29A_GAMMA", "0.5")
    inst_fast = _make_instance_with_tip(tip_value=0.1)
    rates_fast = inst_fast.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=4)

    # γ = 0.99: near-flat
    monkeypatch.setenv("W22_NC29A_GAMMA", "0.99")
    inst_slow = _make_instance_with_tip(tip_value=0.1)
    rates_slow = inst_slow.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=4)

    # Decay ratio: rates_fast should fall faster than rates_slow.
    # rate[1]/rate[0] = γ — exactly γ ratio.
    ratio_fast = rates_fast[1] / rates_fast[0]
    ratio_slow = rates_slow[1] / rates_slow[0]
    assert ratio_fast < ratio_slow, (
        f"γ=0.5 should produce faster decay than γ=0.99: "
        f"fast ratio={ratio_fast:.4f}, slow ratio={ratio_slow:.4f}"
    )


def test_lambda_h_geometric_shape_exact():
    """With γ explicit and TIP fixed: λ^H_h ratio = γ exactly."""
    sol = type("S", (), {})()
    sol.P = type("P", (), {})()
    sol.P.ROI = 0.0
    sol.P.risk = 0.0
    sol.P.kalman_state = None

    import os
    os.environ["W22_NC29A_GAMMA"] = "0.7"
    try:
        inst = _make_instance_with_tip(tip_value=0.1)
        rates = inst.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=5)
        # rates[i] = 0.5 * (γ^(i+1) * (1 - entropy(0.1)))  (combined formula)
        # Pairwise ratio should equal γ
        for i in range(len(rates) - 1):
            ratio = rates[i + 1] / rates[i]
            np.testing.assert_allclose(ratio, 0.7, rtol=1e-6)
    finally:
        del os.environ["W22_NC29A_GAMMA"]


def test_lambda_h_clamped_below_0_5():
    """The [0, 0.5] clamp on λ^H is preserved post-NC29a."""
    sol = type("S", (), {})()
    sol.P = type("P", (), {})()
    sol.P.ROI = 0.0
    sol.P.risk = 0.0
    sol.P.kalman_state = None

    import os
    # γ=0.999 + decisive TIP (low entropy) → unclamped λ^H near 1.0 → must
    # be clamped to 0.5
    os.environ["W22_NC29A_GAMMA"] = "0.999"
    try:
        inst = _make_instance_with_tip(tip_value=0.001)  # very decisive
        rates = inst.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=4)
        # Combined = 0.5 * (λ^H_clamped + 0) = 0.5 * 0.5 = 0.25 max
        for r in rates:
            assert r <= 0.25 + 1e-10, f"λ_combined exceeded 0.25 cap (λ^H clamp): {r}"
    finally:
        del os.environ["W22_NC29A_GAMMA"]


def test_gamma_safety_bounds():
    """Invalid γ values are clamped to (0.01, 0.999)."""
    sol = type("S", (), {})()
    sol.P = type("P", (), {})()
    sol.P.ROI = 0.0
    sol.P.risk = 0.0
    sol.P.kalman_state = None

    import os
    # γ = 2.0 (out of range) → clamped to 0.999
    os.environ["W22_NC29A_GAMMA"] = "2.0"
    try:
        inst = _make_instance_with_tip(tip_value=0.1)
        rates = inst.calculate_multi_horizon_lambda_rates(sol, prediction_horizon=3)
        # ratio should be γ=0.999, not 2.0
        ratio = rates[1] / rates[0]
        assert ratio <= 1.0, f"γ should be clamped; got effective ratio {ratio}"
    finally:
        del os.environ["W22_NC29A_GAMMA"]
