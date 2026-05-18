"""W22-EQ1 — Theorem 6.3.1 / Eq 6.41 stochastic Δ_S unit tests.

Pins ``SMSEMOA._compute_stochastic_hypervolume_contributions_class`` to
the EXACT closed-form expansion of Theorem 6.3.1 (thesis p.131).

Pre-fix the function used a heuristic
``(mean_dROI·var_drisk + mean_drisk·var_dROI) / (var_dROI + var_drisk)``
which is dimensionally wrong (a hypervolume rectangle cannot be a
weighted average of variances) and traces to NO equation in the thesis.

These tests construct hand-computed Δ_S using Eq 6.41 verbatim and
assert the implementation matches to 1e-12.

Thesis Eq 6.41 (verbatim, sign convention preserved):
    E[Δ_S(ẑ_t^(i))] = (m̂_{1,t}^(i) − m̂_{1,t}^(i−1))(m̂_{2,t}^(i+1) − m̂_{2,t}^(i))
                    + Cov(ẑ_{1,t}^(i)|m̂_{2,t}^(i),   ẑ_{2,t}^(i+1)|m̂_{1,t}^(i+1))
                    − Cov(ẑ_{1,t}^(i)|m̂_{2,t}^(i),   ẑ_{2,t}^(i)|m̂_{1,t}^(i))
                    − Cov(ẑ_{1,t}^(i−1)|m̂_{2,t}^(i−1), ẑ_{2,t}^(i+1)|m̂_{1,t}^(i+1))
                    + Cov(ẑ_{1,t}^(i−1)|m̂_{2,t}^(i−1), ẑ_{2,t}^(i)|m̂_{1,t}^(i))

In the Python refactor (no per-solution MC samples), the three
cross-pair covariances are 0; only ``Cov(a, d)`` (within-solution
self-term) is non-zero and equals the KF state ``P[0,1]`` of solution i.

The within-solution covariance is intentionally chosen ≠ 0 in the
fixtures so the Cov-term contribution to Δ_S is exercised and observable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.kalman_filter import KalmanParams
from src.algorithms.sms_emoa import SMSEMOA, StochasticParams
from src.algorithms.solution import Solution
from src.portfolio.portfolio import Portfolio


# ─── Test fixture infrastructure ────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """Each test gets a clean Portfolio class state."""
    prev = (
        Portfolio.mean_ROI,
        Portfolio.median_ROI,
        Portfolio.covariance,
        Portfolio.robust_covariance,
        Portfolio.use_thesis_eq74_risk,
    )
    Portfolio.mean_ROI = np.array([0.01, 0.02, 0.015])
    Portfolio.median_ROI = np.array([0.01, 0.02, 0.015])
    Portfolio.covariance = np.diag([0.04, 0.09, 0.0625])
    Portfolio.robust_covariance = np.diag([0.04, 0.09, 0.0625])
    Portfolio.use_thesis_eq74_risk = False
    try:
        yield
    finally:
        (
            Portfolio.mean_ROI,
            Portfolio.median_ROI,
            Portfolio.covariance,
            Portfolio.robust_covariance,
            Portfolio.use_thesis_eq74_risk,
        ) = prev


def _make_sol(roi: float, risk: float, var_roi: float, var_risk: float, cov_roi_risk: float) -> Solution:
    """Construct a Solution with a 4D Kalman state (ROI, risk, vROI, vRisk).

    The 2×2 top-left of the KF state covariance encodes the within-solution
    (ROI, risk) joint Gaussian per StochasticParams.
    """
    sol = Solution(num_assets=3)
    sol.P.investment = np.array([0.5, 0.3, 0.2])
    sol.P.ROI = roi
    sol.P.risk = risk
    # Disable stability multiplier (so Δ_S equals raw Eq 6.41 value).
    sol.stability = 1.0
    state = KalmanParams()
    state.F = np.eye(4)
    state.H = np.eye(2, 4)
    state.R = np.eye(2) * 0.01
    state.x = np.array([roi, risk, 0.0, 0.0])
    state.x_next = np.array([roi, risk, 0.0, 0.0])
    state.u = np.zeros(4)
    # 4×4 cov with non-trivial (ROI, risk) sub-block; velocity sub-block
    # is irrelevant for Δ_S (StochasticParams reads only P[0:2, 0:2]).
    P = np.zeros((4, 4))
    P[0, 0] = var_roi
    P[1, 1] = var_risk
    P[0, 1] = cov_roi_risk
    P[1, 0] = cov_roi_risk
    P[2, 2] = 0.001
    P[3, 3] = 0.001
    state.P = P
    state.P_next = P.copy()
    sol.P.kalman_state = state
    return sol


def _eq641_expected(
    *,
    roi_i: float,
    risk_i: float,
    left_roi_mean: float,
    right_risk_mean: float,
    self_cov_roi_risk: float,
) -> float:
    """Hand-compute Eq 6.41 with cross-pair Cov terms = 0.

    Returns:
        E[Δ_S] = (m_{1,i} − m_{1,i−1})(m_{2,i+1} − m_{2,i}) − Cov(a, d)

        where Cov(a, d) is the within-solution KF P[0,1] of solution i,
        and the three cross-pair covariances Cov(a,c), Cov(b,c), Cov(b,d)
        are all 0 (Python refactor lacks per-solution MC samples).
    """
    mean_product = (roi_i - left_roi_mean) * (right_risk_mean - risk_i)
    cov_a_d = self_cov_roi_risk
    return mean_product - cov_a_d


# ─── Tests ──────────────────────────────────────────────────────────────


class TestEq641SingleSolution:
    """|C| = 1 edge case: Δ_S = (ROI − R1)(R2 − risk) − Cov(ROI, risk).

    Matches C++ compute_stochastic_Delta_S_front_id single-solution
    branch (legacy-cpp-v2/source/asms_emoa.cpp:383-399).
    """

    def test_single_solution_eq641_matches_hand_computed(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol = _make_sol(roi=0.05, risk=0.10, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.012)

        algo._compute_stochastic_hypervolume_contributions_class([sol])

        # Hand-compute: (0.05 − 0.0)(0.2 − 0.10) − 0.012 = 0.005 − 0.012 = −0.007
        expected = (0.05 - 0.0) * (0.2 - 0.10) - 0.012
        assert sol.hypervolume_contribution == pytest.approx(expected, abs=1e-12), (
            f"Eq 6.41 single-solution: expected {expected}, got "
            f"{sol.hypervolume_contribution}"
        )

    def test_single_solution_zero_self_cov_reduces_to_deterministic_rectangle(self):
        """With Cov(ROI, risk) = 0, Δ_S = deterministic (ROI − R1)(R2 − risk)."""
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol = _make_sol(roi=0.04, risk=0.05, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.0)

        algo._compute_stochastic_hypervolume_contributions_class([sol])

        expected = (0.04 - 0.0) * (0.2 - 0.05)  # = 0.04 * 0.15 = 0.006
        assert sol.hypervolume_contribution == pytest.approx(expected, abs=1e-12)


class TestEq641TwoSolutionClass:
    """|C| = 2 edge case: each solution uses one boundary, one neighbor.

    - i=0: left ROI mean = R1, right risk mean = m̂_risk^(1)
    - i=1: left ROI mean = m̂_ROI^(0), right risk mean = R2
    """

    def test_two_solution_i0_eq641_matches_hand_computed(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol0 = _make_sol(roi=0.03, risk=0.04, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.008)
        sol1 = _make_sol(roi=0.07, risk=0.02, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.015)

        # Pre-sort: ROI ascending → [sol0, sol1] already in order.
        algo._compute_stochastic_hypervolume_contributions_class([sol0, sol1])

        # i=0: ROI=0.03, risk=0.04, left=R1=0.0, right_risk=risk(sol1)=0.02, self_cov=0.008
        expected_0 = _eq641_expected(
            roi_i=0.03,
            risk_i=0.04,
            left_roi_mean=0.0,
            right_risk_mean=0.02,
            self_cov_roi_risk=0.008,
        )
        # Hand-trace: (0.03 − 0.0)(0.02 − 0.04) − 0.008
        #          = 0.03 · (−0.02) − 0.008
        #          = −0.0006 − 0.008 = −0.0086
        assert expected_0 == pytest.approx(-0.0086, abs=1e-12)
        assert sol0.hypervolume_contribution == pytest.approx(expected_0, abs=1e-12)

    def test_two_solution_i1_eq641_matches_hand_computed(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol0 = _make_sol(roi=0.03, risk=0.04, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.008)
        sol1 = _make_sol(roi=0.07, risk=0.02, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.015)

        algo._compute_stochastic_hypervolume_contributions_class([sol0, sol1])

        # i=1: ROI=0.07, risk=0.02, left_ROI=ROI(sol0)=0.03, right=R2=0.2, self_cov=0.015
        expected_1 = _eq641_expected(
            roi_i=0.07,
            risk_i=0.02,
            left_roi_mean=0.03,
            right_risk_mean=0.2,
            self_cov_roi_risk=0.015,
        )
        # Hand-trace: (0.07 − 0.03)(0.20 − 0.02) − 0.015
        #          = 0.04 · 0.18 − 0.015
        #          = 0.0072 − 0.015 = −0.0078
        assert expected_1 == pytest.approx(-0.0078, abs=1e-12)
        assert sol1.hypervolume_contribution == pytest.approx(expected_1, abs=1e-12)


class TestEq641ThreeSolutionMiddle:
    """|C| ≥ 3: middle position exercises the FULL Eq 6.41 with both
    neighbors (no boundary fallback). This is the canonical Theorem
    6.3.1 case.
    """

    def test_three_solution_middle_eq641_matches_hand_computed(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol_left = _make_sol(roi=0.02, risk=0.08, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.010)
        sol_mid = _make_sol(roi=0.05, risk=0.05, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.020)
        sol_right = _make_sol(roi=0.09, risk=0.03, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.005)

        algo._compute_stochastic_hypervolume_contributions_class([sol_left, sol_mid, sol_right])

        # i=1 (middle): ROI=0.05, risk=0.05, left_ROI=0.02, right_risk=0.03, self_cov=0.020
        expected_mid = _eq641_expected(
            roi_i=0.05,
            risk_i=0.05,
            left_roi_mean=0.02,
            right_risk_mean=0.03,
            self_cov_roi_risk=0.020,
        )
        # Hand-trace: (0.05 − 0.02)(0.03 − 0.05) − 0.020
        #          = 0.03 · (−0.02) − 0.020
        #          = −0.0006 − 0.020 = −0.0206
        assert expected_mid == pytest.approx(-0.0206, abs=1e-12)
        assert sol_mid.hypervolume_contribution == pytest.approx(expected_mid, abs=1e-12)

    def test_three_solution_first_uses_R1_as_left_boundary(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol_left = _make_sol(roi=0.02, risk=0.08, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.010)
        sol_mid = _make_sol(roi=0.05, risk=0.05, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.020)
        sol_right = _make_sol(roi=0.09, risk=0.03, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.005)

        algo._compute_stochastic_hypervolume_contributions_class([sol_left, sol_mid, sol_right])

        # i=0: ROI=0.02, risk=0.08, left=R1=0.0, right_risk=risk(sol_mid)=0.05, self_cov=0.010
        expected_first = _eq641_expected(
            roi_i=0.02,
            risk_i=0.08,
            left_roi_mean=0.0,
            right_risk_mean=0.05,
            self_cov_roi_risk=0.010,
        )
        # Hand-trace: (0.02 − 0.0)(0.05 − 0.08) − 0.010
        #          = 0.02 · (−0.03) − 0.010
        #          = −0.0006 − 0.010 = −0.0106
        assert expected_first == pytest.approx(-0.0106, abs=1e-12)
        assert sol_left.hypervolume_contribution == pytest.approx(expected_first, abs=1e-12)

    def test_three_solution_last_uses_R2_as_right_boundary(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol_left = _make_sol(roi=0.02, risk=0.08, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.010)
        sol_mid = _make_sol(roi=0.05, risk=0.05, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.020)
        sol_right = _make_sol(roi=0.09, risk=0.03, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.005)

        algo._compute_stochastic_hypervolume_contributions_class([sol_left, sol_mid, sol_right])

        # i=2: ROI=0.09, risk=0.03, left_ROI=0.05, right=R2=0.2, self_cov=0.005
        expected_last = _eq641_expected(
            roi_i=0.09,
            risk_i=0.03,
            left_roi_mean=0.05,
            right_risk_mean=0.2,
            self_cov_roi_risk=0.005,
        )
        # Hand-trace: (0.09 − 0.05)(0.20 − 0.03) − 0.005
        #          = 0.04 · 0.17 − 0.005
        #          = 0.0068 − 0.005 = 0.0018
        assert expected_last == pytest.approx(0.0018, abs=1e-12)
        assert sol_right.hypervolume_contribution == pytest.approx(expected_last, abs=1e-12)


class TestEq641HeuristicRegression:
    """Pin: post-fix Δ_S MUST NOT equal the pre-fix heuristic
    ``(mean_dROI·var_drisk + mean_drisk·var_dROI) / (var_dROI + var_drisk)``.

    This guards against accidental reversion to the dimensionally-wrong
    formula. Use a 3-solution middle fixture where the two formulas
    differ by O(1) — there is no risk of a numerical coincidence.
    """

    def test_middle_solution_post_fix_does_not_match_pre_fix_heuristic(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2))
        sol_left = _make_sol(roi=0.02, risk=0.08, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.010)
        sol_mid = _make_sol(roi=0.05, risk=0.05, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.020)
        sol_right = _make_sol(roi=0.09, risk=0.03, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.005)

        algo._compute_stochastic_hypervolume_contributions_class([sol_left, sol_mid, sol_right])

        # Pre-fix heuristic for middle (per pre-fix code):
        #   mean_dROI = mean_ROI_self - mean_ROI_next = 0.05 - 0.09 = -0.04
        #   mean_drisk = mean_risk_prev - mean_risk_self = 0.08 - 0.05 = 0.03
        #   var_dROI = (var_ROI_self_cond) + (var_ROI_next_cond)
        #   var_drisk = (var_risk_prev_cond) + (var_risk_self_cond)
        #   pre_fix_dS = (mean_dROI·var_drisk + mean_drisk·var_dROI) / (var_dROI + var_drisk)
        stoch_self = StochasticParams(sol_mid)
        stoch_prev = StochasticParams(sol_left)
        stoch_next = StochasticParams(sol_right)
        mean_dROI = stoch_self.conditional_mean_ROI - stoch_next.conditional_mean_ROI
        mean_drisk = stoch_prev.conditional_mean_risk - stoch_self.conditional_mean_risk
        var_dROI = stoch_self.conditional_var_ROI + stoch_next.conditional_var_ROI
        var_drisk = stoch_prev.conditional_var_risk + stoch_self.conditional_var_risk
        pre_fix = (mean_dROI * var_drisk + mean_drisk * var_dROI) / (var_dROI + var_drisk)

        # Post-fix value (Eq 6.41) was hand-computed as -0.0206 in the
        # canonical test above. Assert the implementation is the post-fix
        # value AND is significantly different from the pre-fix heuristic.
        post_fix_expected = -0.0206
        assert sol_mid.hypervolume_contribution == pytest.approx(post_fix_expected, abs=1e-12)
        assert abs(sol_mid.hypervolume_contribution - pre_fix) > 1e-6, (
            f"Post-fix Δ_S ({sol_mid.hypervolume_contribution}) collapsed to "
            f"pre-fix heuristic ({pre_fix}) — Eq 6.41 fix was reverted."
        )


class TestEq641StabilityWeighting:
    """Δ_S × stability_factor wiring is preserved post-fix (existing
    behavior must remain — fix is only to the per-position formula).
    """

    def test_stability_multiplier_applied_post_fix(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=False)
        sol = _make_sol(roi=0.05, risk=0.10, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.012)
        sol.stability = 0.5  # depressing multiplier

        algo._compute_stochastic_hypervolume_contributions_class([sol])

        # Bare Δ_S = (0.05 − 0.0)(0.2 − 0.10) − 0.012 = 0.005 − 0.012 = −0.007
        # With stability 0.5: Δ_S × 0.5 = −0.0035
        expected = (-0.007) * 0.5
        assert sol.hypervolume_contribution == pytest.approx(expected, abs=1e-12)

    def test_v2_stability_weighting_overrides_to_1_post_fix(self):
        algo = SMSEMOA(z_ref=(0.0, 0.2), use_v2_stability_weighting=True)
        sol = _make_sol(roi=0.05, risk=0.10, var_roi=0.04, var_risk=0.09, cov_roi_risk=0.012)
        sol.stability = 0.5

        algo._compute_stochastic_hypervolume_contributions_class([sol])

        # v2 flag should force stability factor → 1.0, ignoring sol.stability
        expected = -0.007
        assert sol.hypervolume_contribution == pytest.approx(expected, abs=1e-12)
