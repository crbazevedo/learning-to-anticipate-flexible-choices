"""W22 V6 (Reading D): tests for use_v2_kf_lifecycle flag effect on
_evaluate_solution KF state evolution.

Per docs/W19-D-CARRY-1: v2's Kalman_filter() in legacy-cpp-v2/source/
kalman_filter.cpp calls Kalman_update() BEFORE Kalman_prediction() —
reversed vs Python's pre-W21-5 predict→update order. When the flag is
True, _evaluate_solution follows the update with a prediction, leaving
the KF state in v2-style PREDICTED mode at end of call.

Default behavior (flag=False) preserves pre-W21-5 update-only-per-call
lifecycle for backward compatibility.

Tests use call-counting wrappers to avoid the SMSEMOA full setup
complexity — we just need to assert WHETHER kalman_prediction gets
invoked after the update.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.sms_emoa import SMSEMOA
from src.algorithms.solution import Solution
from src.algorithms.kalman_filter import KalmanParams
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    prev_m = Portfolio.mean_ROI
    prev_med = Portfolio.median_ROI
    prev_c = Portfolio.covariance
    prev_rc = Portfolio.robust_covariance
    prev_r = Portfolio.use_thesis_eq74_risk
    # Solution.__init__ calls Portfolio.compute_efficiency which needs
    # both mean_ROI + median_ROI + covariance + robust_covariance. Set
    # all four so the fixture solutions construct cleanly.
    Portfolio.mean_ROI = np.array([0.01, 0.02, 0.015])
    Portfolio.median_ROI = np.array([0.01, 0.02, 0.015])  # same as mean for test simplicity
    Portfolio.covariance = np.diag([0.04, 0.09, 0.0625])
    Portfolio.robust_covariance = np.diag([0.04, 0.09, 0.0625])
    Portfolio.use_thesis_eq74_risk = False
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_m
        Portfolio.median_ROI = prev_med
        Portfolio.covariance = prev_c
        Portfolio.robust_covariance = prev_rc
        Portfolio.use_thesis_eq74_risk = prev_r


def _make_solution_with_kf(roi: float = 0.015, risk: float = 0.05) -> Solution:
    """Make a Solution with an initialized 4D Kalman state.

    Note: x and x_next are set with NON-ZERO velocities (last 2
    components) so that kalman_prediction (x_next = F @ x + u with
    F's velocity-projection coupling) produces a measurable change.
    Without non-zero velocities, F @ x = x for the first two
    components and the predict test can't distinguish flag-on from
    flag-off (both leave x_next == initial).
    """
    sol = Solution(num_assets=3)
    sol.P.investment = np.array([0.5, 0.3, 0.2])
    sol.P.ROI = roi
    sol.P.risk = risk
    state = KalmanParams()
    state.F = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    state.H = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    state.R = np.array([
        [0.01, 0.0],
        [0.0, 0.01],
    ])
    # NON-ZERO velocities so predict produces a measurable change
    state.x = np.array([roi, risk, 0.001, 0.0005])
    state.x_next = np.array([roi, risk, 0.001, 0.0005])
    state.P = np.eye(4) * 0.1
    state.P_next = np.eye(4) * 0.1
    state.u = np.zeros(4)
    sol.P.kalman_state = state
    return sol


# ─── Behavioral tests via call-counting ─────────────────────────────────


class TestFlagOffSkipsPrediction:
    """With flag=False (default), kalman_prediction is NOT called after
    kalman_update inside _evaluate_solution."""

    def test_flag_off_kalman_prediction_NOT_called(self):
        algo = SMSEMOA()
        assert algo.use_v2_kf_lifecycle is False

        sol = _make_solution_with_kf()

        # Patch the kalman_filter module's prediction function used inside
        # _evaluate_solution. The patch path matches the local import
        # inside the V6 branch.
        with patch("src.algorithms.kalman_filter.kalman_prediction") as mock_pred:
            try:
                algo._evaluate_solution(sol, data={})
            except Exception:
                # _evaluate_solution may fail at downstream steps; we
                # only care whether kalman_prediction was invoked DURING
                # the update path. Suppress + check count below.
                pass
            # With flag OFF: the V6 branch should NOT have invoked the
            # patched kalman_prediction.
            assert mock_pred.call_count == 0, (
                f"Flag OFF should NOT call kalman_prediction; got "
                f"{mock_pred.call_count} calls"
            )


class TestFlagOnInvokesPrediction:
    """With flag=True, kalman_prediction IS called once after the
    kalman_update inside _evaluate_solution."""

    def test_flag_on_kalman_prediction_called_exactly_once(self):
        algo = SMSEMOA(use_v2_kf_lifecycle=True)
        assert algo.use_v2_kf_lifecycle is True

        sol = _make_solution_with_kf()

        with patch("src.algorithms.kalman_filter.kalman_prediction") as mock_pred:
            try:
                algo._evaluate_solution(sol, data={})
            except Exception:
                pass
            # With flag ON: kalman_prediction should fire exactly once
            # (V6's added line, after the existing kalman_update call).
            assert mock_pred.call_count == 1, (
                f"Flag ON should call kalman_prediction exactly once; got "
                f"{mock_pred.call_count} calls"
            )
            # Argument check: it should be invoked on the solution's
            # kalman_state.
            args, _ = mock_pred.call_args
            assert args[0] is sol.P.kalman_state


class TestResidualRecordingPreserved:
    """The W17-2 residual recording (kalman_update returns squared
    residual + forwards to learner) must still fire when V6 flag is
    True. The follow-up prediction must NOT interfere."""

    def test_flag_on_residual_recording_preserved(self):
        from src.algorithms.anticipatory_learning import AnticipatoryLearning

        algo = SMSEMOA(use_v2_kf_lifecycle=True)
        learner = AnticipatoryLearning(window_size=3)
        algo.set_learning(learner)

        sol = _make_solution_with_kf()

        buf_before = len(list(getattr(learner, "_kf_residual_window", [])))
        try:
            algo._evaluate_solution(sol, data={})
        except Exception:
            pass
        buf_after = len(list(getattr(learner, "_kf_residual_window", [])))

        assert buf_after == buf_before + 1, (
            f"Residual recording broke when flag=True; buffer went from "
            f"{buf_before} → {buf_after}"
        )


class TestStateBehavioralDelta:
    """Functional check: with all things equal, flag-on and flag-off
    produce DIFFERENT kalman_state.x_next at end of _evaluate_solution
    (because the predict updates x_next; without predict x_next was
    set by initialization and stays equal to initial x_next)."""

    def test_flag_on_vs_off_produce_different_x_next(self):
        # Flag OFF: x_next unchanged from initialization (no predict)
        algo_off = SMSEMOA(use_v2_kf_lifecycle=False)
        sol_off = _make_solution_with_kf()
        x_next_initial = sol_off.P.kalman_state.x_next.copy()
        try:
            algo_off._evaluate_solution(sol_off, data={})
        except Exception:
            pass
        x_next_off_final = sol_off.P.kalman_state.x_next.copy()

        # Flag ON: x_next mutated by kalman_prediction post-update
        algo_on = SMSEMOA(use_v2_kf_lifecycle=True)
        sol_on = _make_solution_with_kf()
        try:
            algo_on._evaluate_solution(sol_on, data={})
        except Exception:
            pass
        x_next_on_final = sol_on.P.kalman_state.x_next.copy()

        # Flag OFF preserved initial x_next
        assert np.allclose(x_next_off_final, x_next_initial, atol=1e-12), (
            "Flag OFF altered x_next unexpectedly"
        )
        # Flag ON changed x_next
        assert not np.allclose(x_next_on_final, x_next_initial, atol=1e-9), (
            "Flag ON did NOT change x_next (kalman_prediction did not "
            "fire or had no effect)"
        )
