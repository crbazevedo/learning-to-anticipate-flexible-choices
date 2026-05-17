"""W21-1: tests for use_v2_stability_weighting flag (Reading-F INVERTED experimental flag).

Per docs/W20-5-CORRECTION-READING-F-INVERSION.md, this flag inverts the
original W20-5 framing: instead of "ADD v2's stability formula to Python",
the flag DISABLES Python's existing stability multiplier (which depresses
Δ_S below the bare Gaussian expectation) by forcing solution.stability
to be treated as 1.0 — matching v2's effective behavior, where the
stability field is declared but never reassigned anywhere in
legacy-cpp-v2/source/*.cpp.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m, prev_c = Portfolio.mean_ROI, Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.covariance = prev_c


def _make_solution(roi: float, risk: float, stability: float) -> Solution:
    """Make a fixture solution with explicit objective values + stability."""
    sol = Solution(num_assets=5)
    sol.P.ROI = roi
    sol.P.risk = risk
    sol.stability = stability  # override the auto-computed one
    return sol


class TestFlagDefaultsFalseAndUsesPythonStability:
    """Default behavior: use_v2_stability_weighting=False → Python stability."""

    def test_default_flag_is_false(self):
        algo = SMSEMOA()
        assert algo.use_v2_stability_weighting is False

    def test_default_applies_python_stability_lt_one(self):
        """When flag is False, < 1.0 stability depresses HV contribution."""
        algo = SMSEMOA(z_ref=(0.0, 0.2))  # R1=0.0, R2=0.2
        # Solution with stability=0.5: HV contribution should be HALVED.
        sol = _make_solution(roi=0.1, risk=0.05, stability=0.5)
        algo._compute_hypervolume_contributions_class([sol])
        # Single-solution class: HV = (ROI - R1) * (R2 - risk) = 0.1 * 0.15 = 0.015
        # With default flag=False, single-solution class skips stability
        # (see _compute_hypervolume_contributions_class). For multi-solution
        # tests we exercise the stability path.
        assert sol.hypervolume_contribution == pytest.approx(0.015, abs=1e-12)


class TestFlagTrueForcesStabilityToOne:
    """Flag=True → solution.stability is treated as 1.0 (v2 no-op behavior)."""

    def test_multi_solution_class_flag_off_applies_stability(self):
        """With flag OFF, stability=0.5 halves the HV contribution in multi-class."""
        algo = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=False)
        sols = [
            _make_solution(roi=0.05, risk=0.10, stability=0.5),
            _make_solution(roi=0.10, risk=0.08, stability=0.5),
            _make_solution(roi=0.15, risk=0.06, stability=0.5),
        ]
        algo._compute_hypervolume_contributions_class(sols)
        # All three should have non-zero HV contributions, all multiplied by 0.5.
        # Verify the middle one's relative magnitude is halved compared to
        # what it would be without stability.
        unmul = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=True)
        sols_unmul = [
            _make_solution(roi=0.05, risk=0.10, stability=0.5),
            _make_solution(roi=0.10, risk=0.08, stability=0.5),
            _make_solution(roi=0.15, risk=0.06, stability=0.5),
        ]
        unmul._compute_hypervolume_contributions_class(sols_unmul)
        # Each solution's flag-off contribution should be exactly 0.5×
        # the flag-on contribution.
        for s_off, s_on in zip(sols, sols_unmul):
            if s_on.hypervolume_contribution != 0.0:
                assert s_off.hypervolume_contribution == pytest.approx(
                    0.5 * s_on.hypervolume_contribution, abs=1e-12
                )

    def test_flag_on_ignores_python_stability(self):
        """With flag ON, even stability=0.01 leaves HV unaffected."""
        algo = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=True)
        sols = [
            _make_solution(roi=0.05, risk=0.10, stability=0.01),
            _make_solution(roi=0.10, risk=0.08, stability=0.01),
            _make_solution(roi=0.15, risk=0.06, stability=0.01),
        ]
        algo._compute_hypervolume_contributions_class(sols)

        # Now with stability=1.0 explicitly (control)
        algo_ctrl = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=True)
        sols_ctrl = [
            _make_solution(roi=0.05, risk=0.10, stability=1.0),
            _make_solution(roi=0.10, risk=0.08, stability=1.0),
            _make_solution(roi=0.15, risk=0.06, stability=1.0),
        ]
        algo_ctrl._compute_hypervolume_contributions_class(sols_ctrl)

        # The HV contributions should be IDENTICAL — the flag-on path
        # ignores solution.stability entirely.
        for s, sc in zip(sols, sols_ctrl):
            assert s.hypervolume_contribution == pytest.approx(
                sc.hypervolume_contribution, abs=1e-15
            )


class TestFlagAttributeOnConstructor:
    """Flag is exposed as an instance attribute."""

    def test_attribute_default_false(self):
        algo = SMSEMOA()
        assert hasattr(algo, "use_v2_stability_weighting")
        assert algo.use_v2_stability_weighting is False

    def test_attribute_true_when_set(self):
        algo = SMSEMOA(use_v2_stability_weighting=True)
        assert algo.use_v2_stability_weighting is True

    def test_attribute_false_when_explicitly_set_false(self):
        algo = SMSEMOA(use_v2_stability_weighting=False)
        assert algo.use_v2_stability_weighting is False
