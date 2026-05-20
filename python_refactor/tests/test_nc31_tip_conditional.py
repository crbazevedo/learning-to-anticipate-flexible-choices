"""W22-NC31 structural-fix tests: TIP conditional-mode per Defn 6.1.

Per W22 Inspection 1 + operator directive 2026-05-19. The TIP code samples
both current AND predicted as Gaussians, violating Defn 6.1's conditional
`| ẑ_t` (which requires current to be OBSERVED, not sampled).

Inspection 1 found empirically equivalent results (<1.5% delta across 5
scenarios) but the code is mathematically WRONG per definition. NC31 fixes
this with an opt-in env var:
  W22_NC31_TIP_CONDITIONAL=1 → Defn 6.1 correct (current fixed, predicted sampled)
  Default OFF preserves legacy behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.temporal_incomparability_probability import TemporalIncomparabilityCalculator


def test_default_is_legacy_joint_sampling():
    """Without env var: legacy joint sampling (samples both current and predicted)."""
    calc = TemporalIncomparabilityCalculator(monte_carlo_samples=200)
    # Equal means, equal cov: under both modes, TIP ≈ 0.5
    np.random.seed(0)
    tip = calc._calculate_tip_with_covariance(
        current_roi=0.001, current_risk=0.005,
        current_cov=np.diag([0.01, 0.01]),
        predicted_roi=0.001, predicted_risk=0.005,
        predicted_cov=np.diag([0.01, 0.01]),
    )
    assert 0.05 <= tip <= 0.95
    # Sanity check: around 0.5
    assert 0.3 < tip < 0.7


def test_conditional_mode_fixes_current(monkeypatch):
    """With env var: current is FIXED, only predicted sampled. Defn 6.1 correct."""
    monkeypatch.setenv("W22_NC31_TIP_CONDITIONAL", "1")
    calc = TemporalIncomparabilityCalculator(monte_carlo_samples=200)
    np.random.seed(0)
    tip = calc._calculate_tip_with_covariance(
        current_roi=0.001, current_risk=0.005,
        current_cov=np.diag([0.01, 0.01]),
        predicted_roi=0.001, predicted_risk=0.005,
        predicted_cov=np.diag([0.01, 0.01]),
    )
    assert 0.05 <= tip <= 0.95
    # At equal means + equal cov, TIP_conditional should also be ~0.5 (symmetry)
    assert 0.3 < tip < 0.7


def test_conditional_mode_empirically_close_to_joint(monkeypatch):
    """Per Inspection 1 finding: in close-means regime, conditional vs joint
    differ by <5% absolute. This is the test that locks in the empirical
    equivalence the inspection documented."""
    args = dict(
        current_roi=0.001, current_risk=0.005,
        current_cov=np.diag([0.01, 0.01]),
        predicted_roi=0.0012, predicted_risk=0.0048,  # close to current
        predicted_cov=np.diag([0.01, 0.01]),
    )
    calc = TemporalIncomparabilityCalculator(monte_carlo_samples=2000)
    np.random.seed(0)
    tip_joint = calc._calculate_tip_with_covariance(**args)

    monkeypatch.setenv("W22_NC31_TIP_CONDITIONAL", "1")
    np.random.seed(0)
    tip_conditional = calc._calculate_tip_with_covariance(**args)

    diff = abs(tip_joint - tip_conditional)
    assert diff < 0.05, (
        f"TIP_joint={tip_joint:.4f} and TIP_conditional={tip_conditional:.4f} "
        f"differ by {diff:.4f}; per Inspection 1, should be < 0.05 in close-means regime"
    )


def test_conditional_mode_deterministic_at_zero_predicted_variance(monkeypatch):
    """If predicted_cov ≈ 0, predicted_sample == predicted_mean every time.
    Then TIP depends entirely on the dominance between fixed-current and
    fixed-predicted: deterministic 0 or 1."""
    monkeypatch.setenv("W22_NC31_TIP_CONDITIONAL", "1")
    calc = TemporalIncomparabilityCalculator(
        monte_carlo_samples=100,
        clamp_range=None,  # disable clamp to see raw TIP
    )
    # Current dominates predicted: higher ROI, lower risk
    np.random.seed(0)
    tip = calc._calculate_tip_with_covariance(
        current_roi=0.005, current_risk=0.001,  # higher ROI, lower risk
        current_cov=np.diag([1e-12, 1e-12]),
        predicted_roi=0.001, predicted_risk=0.005,  # lower ROI, higher risk
        predicted_cov=np.diag([1e-12, 1e-12]),
    )
    # Current strictly dominates predicted → mutual non-dominance = 0 → TIP ≈ 0
    assert tip < 0.05, f"TIP should be ~0 when current dominates predicted; got {tip}"
