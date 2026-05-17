"""W19-4: anticipative rate G — three-formula divergence tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cross_validation.run_anticipative_rate import (
    v1_linear_entropy,
    v1_alpha,
    v2_alpha,
    binary_entropy,
    python_lambda_h,
    python_lambda_eq716,
)


class TestLinearEntropyTent:
    """v1 linear_entropy: triangular tent."""

    def test_tent_peak_at_half(self):
        assert v1_linear_entropy(0.5) == 1.0

    def test_zero_at_extremes(self):
        assert v1_linear_entropy(0.0) == 0.0
        assert v1_linear_entropy(1.0) == 0.0

    def test_linear_to_peak(self):
        assert v1_linear_entropy(0.25) == 0.5
        assert v1_linear_entropy(0.75) == 0.5


class TestSaturationRegimeDivergence:
    """KEYSTONE finding: at TIP=0.5, the three formulas behave differently."""

    def test_v1_collapses_at_saturation(self):
        """v1 alpha = 0 at TIP=0.5 (max-entropy → tent peak → 1-1=0)."""
        assert v1_alpha(0.5) == 0.0

    def test_v2_maintains_half_at_saturation(self):
        """v2 alpha = 0.5 at TIP=0.5 (monotonic 1-p)."""
        assert v2_alpha(0.5) == 0.5

    def test_python_lambda_h_collapses_at_saturation(self):
        """Python λ^H = 0 at TIP=0.5 (Shannon → binary_entropy(0.5)=1 → 1-1=0)."""
        assert python_lambda_h(0.5) == 0.0

    def test_python_lambda_eq716_collapses_at_saturation(self):
        """Python λ Eq 7.16 = 0 at TIP=0.5 with λ^K=0."""
        assert python_lambda_eq716(0.5, lambda_k=0.0) == 0.0

    def test_v2_keeps_anticipation_active(self):
        """v2 alpha > 0 throughout TIP ∈ (0, 1)."""
        for tip in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert v2_alpha(tip) > 0.0, f"v2 alpha collapsed at TIP={tip}"


class TestMonotonicityVsSymmetry:
    """v2 monotonic decreasing; v1 + Python symmetric around TIP=0.5."""

    def test_v1_symmetric(self):
        for d in [0.1, 0.2, 0.3, 0.4]:
            np.testing.assert_allclose(v1_alpha(0.5 - d), v1_alpha(0.5 + d),
                                          atol=1e-15)

    def test_v2_monotonic_decreasing(self):
        prev = v2_alpha(0.0)
        for tip in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            curr = v2_alpha(tip)
            assert curr <= prev, f"v2 not monotonic at TIP={tip}"
            prev = curr

    def test_python_lambda_symmetric(self):
        for d in [0.1, 0.2, 0.3, 0.4]:
            np.testing.assert_allclose(python_lambda_h(0.5 - d),
                                          python_lambda_h(0.5 + d),
                                          atol=1e-15)


class TestBoundaryValues:
    """All formulas: alpha=1 at extremes (TIP=0 or TIP=1)."""

    def test_v1_alpha_max_at_extremes(self):
        assert v1_alpha(0.0) == 1.0
        assert v1_alpha(1.0) == 1.0

    def test_python_lambda_h_max_at_extremes(self):
        assert python_lambda_h(0.0) == 1.0
        assert python_lambda_h(1.0) == 1.0

    def test_v2_alpha_only_max_at_tip_zero(self):
        assert v2_alpha(0.0) == 1.0
        assert v2_alpha(1.0) == 0.0  # v2 is NOT symmetric
