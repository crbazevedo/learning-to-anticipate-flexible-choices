"""W18-4 cross-check F: TIP saturation diagnosis tests.

Tests the three-implementation comparison + the structural saturation
finding. Closes operator check F + the W17-5 saturation question.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.run_tip import (
    tip_py_mc,
    tip_cpp_same_sign,
    tip_py_pareto_mixed,
)


class TestTIPImplementationsAgreeOnDisjointGaussians:
    """All three TIPs → 0 when distributions are well-separated."""

    def test_disjoint_means_tip_near_zero(self):
        cur_m = np.array([0.10, 0.05])
        pred_m = np.array([-0.10, 0.20])
        small_cov = np.eye(2) * 1e-4
        py_mc = tip_py_mc(cur_m, small_cov, pred_m, small_cov,
                          n_samples=2000, clamp=None)
        cpp_ss = tip_cpp_same_sign(cur_m, pred_m, small_cov, small_cov)
        py_pm = tip_py_pareto_mixed(cur_m, pred_m, small_cov, small_cov)
        # All should be near zero (means well-separated)
        assert py_mc < 0.05
        assert cpp_ss < 0.05
        assert py_pm < 0.05


class TestTIPSaturationOnProductionScale:
    """W17-5 finding reproduced: TIP saturates at ~0.5 on production inputs."""

    def test_production_scale_tip_saturates(self):
        """Tiny means + tiny covs (W17-5 production regime) → all impl → ~0.5."""
        cur_m = np.array([0.001, 0.003])
        pred_m = np.array([0.0015, 0.003])
        kf_cov = np.eye(2) * 1e-5
        py_mc = tip_py_mc(cur_m, kf_cov, pred_m, kf_cov,
                          n_samples=2000, clamp=None)
        cpp_ss = tip_cpp_same_sign(cur_m, pred_m, kf_cov, kf_cov)
        py_pm = tip_py_pareto_mixed(cur_m, pred_m, kf_cov, kf_cov)
        # All should saturate near 0.5 (max-uncertainty)
        assert abs(py_mc - 0.5) < 0.1, f"py_mc={py_mc} should be near 0.5"
        assert abs(cpp_ss - 0.5) < 0.1, f"cpp_same_sign={cpp_ss} should be near 0.5"
        assert abs(py_pm - 0.5) < 0.1, f"py_pareto_mixed={py_pm} should be near 0.5"


class TestTIPImplementationParityWithinTolerance:
    """Cross-impl parity within MC noise (~0.05) on diverse cases."""

    def test_cpp_and_python_mc_within_mc_noise(self):
        # Coincident distributions: known TIP ~0.5
        cur_m = np.array([0.05, 0.10])
        pred_m = np.array([0.05, 0.10])
        cov = np.eye(2) * 0.01
        py_mc = tip_py_mc(cur_m, cov, pred_m, cov,
                          n_samples=5000, clamp=None)
        cpp_ss = tip_cpp_same_sign(cur_m, pred_m, cov, cov)
        # Within MC noise + analytical-vs-MC small differences
        assert abs(py_mc - cpp_ss) < 0.05, f"py_mc={py_mc}, cpp_ss={cpp_ss}"


class TestSaturationIsStructural:
    """Documenting the W18-4 verdict: saturation = data property, not bug."""

    def test_separation_to_overlap_continuum(self):
        """As mean separation shrinks relative to cov, TIP → 0.5."""
        small_cov = np.eye(2) * 1e-4
        # Separation = 0.0 → TIP ≈ 0.5 (overlap)
        tip_overlap = tip_py_mc(
            np.array([0.05, 0.10]), small_cov,
            np.array([0.05, 0.10]), small_cov,
            n_samples=2000, clamp=None,
        )
        # Separation = 0.5 (large) → TIP → 0 (no overlap)
        tip_separated = tip_py_mc(
            np.array([0.05, 0.10]), small_cov,
            np.array([0.55, -0.40]), small_cov,
            n_samples=2000, clamp=None,
        )
        # Verify the gradient: overlap is much closer to 0.5 than separated
        assert abs(tip_overlap - 0.5) < 0.1
        assert tip_separated < 0.05
