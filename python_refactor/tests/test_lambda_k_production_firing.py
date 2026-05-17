"""
W17-2: regression tests for record_kf_residual production wiring
(W16-1-CARRY-1 closure).

Verifies that:
  1. kalman_update returns the squared-residual scalar (W17-2 addition)
  2. After K periods of Kalman updates, the residual buffer is populated
     (truncated to K)
  3. _compute_lambda_k_with_branch returns 'kperiod_sum' when the buffer
     is populated (vs 'warmup_traditional' before, 'k0_zero' for K=0)
  4. The trace row's lambda_k_branch column reflects the actual branch
"""
from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.anticipatory_learning import AnticipatoryLearning
from src.algorithms.solution import Solution
from src.algorithms.kalman_filter import (
    KalmanParams,
    initialize_kalman_matrices,
    kalman_update,
)
from src.portfolio.portfolio import Portfolio


@pytest.fixture(autouse=True)
def _reset_portfolio_class_state():
    """Reset Portfolio class variables between tests."""
    prev_mean = Portfolio.mean_ROI
    prev_cov = Portfolio.covariance
    Portfolio.mean_ROI = None
    Portfolio.covariance = None
    try:
        yield
    finally:
        Portfolio.mean_ROI = prev_mean
        Portfolio.covariance = prev_cov


def _make_solution(num_assets: int = 5) -> Solution:
    sol = Solution(num_assets=num_assets)
    sol.P.ROI = 0.05
    sol.P.risk = 0.10
    sol.prediction_error = 0.5
    sol.alpha = 0.5
    return sol


def _make_kalman_params() -> KalmanParams:
    """Build a minimal valid KalmanParams for unit testing."""
    F, H, R = initialize_kalman_matrices()
    params = KalmanParams()
    params.F = F
    params.H = H
    params.R = R
    params.x = np.array([0.05, 0.10, 0.0, 0.0])
    params.x_next = np.array([0.05, 0.10, 0.0, 0.0])
    params.P = np.eye(4) * 0.01
    params.P_next = np.eye(4) * 0.01
    return params


class TestKalmanUpdateReturnsResidual:
    """W17-2: kalman_update returns innovation^T @ innovation scalar."""

    def test_returns_float(self):
        params = _make_kalman_params()
        measurement = np.array([0.07, 0.12])
        residual_sq = kalman_update(params, measurement)
        assert isinstance(residual_sq, float)

    def test_residual_matches_innovation(self):
        """Returned scalar equals (Z - H@x_next)^T @ (Z - H@x_next)."""
        params = _make_kalman_params()
        measurement = np.array([0.07, 0.12])
        expected_innov = measurement - params.H @ params.x_next
        expected_sq = float(expected_innov @ expected_innov)
        residual_sq = kalman_update(params, measurement)
        assert residual_sq == pytest.approx(expected_sq, abs=1e-12)

    def test_zero_residual_when_perfect_prediction(self):
        """If measurement equals predicted observation, residual is 0."""
        params = _make_kalman_params()
        # H @ x_next = [0.05, 0.10] → measurement = [0.05, 0.10] → residual=0
        measurement = np.array([0.05, 0.10])
        residual_sq = kalman_update(params, measurement)
        assert residual_sq == pytest.approx(0.0, abs=1e-12)


class TestRecordKFResidualPopulatesBuffer:
    """W17-2: K-period residual buffer fills as Kalman updates fire."""

    def test_buffer_grows_to_k(self):
        K = 3
        al = AnticipatoryLearning(window_size=K)
        params = _make_kalman_params()
        # 4 periods of updates
        for measurement_val in [0.06, 0.07, 0.08, 0.09]:
            residual_sq = kalman_update(params, np.array([measurement_val, 0.10]))
            al.record_kf_residual(residual_sq)
        # Buffer truncated to K=3
        assert len(al._kf_residual_window) == K
        # All entries are non-negative floats
        assert all(isinstance(r, float) and r >= 0.0 for r in al._kf_residual_window)


class TestLambdaKBranches:
    """W17-2: _compute_lambda_k_with_branch returns correct branch label."""

    def test_k0_returns_zero_zero_branch(self):
        al = AnticipatoryLearning(window_size=0)
        sol = _make_solution()
        val, branch = al._compute_lambda_k_with_branch(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        assert val == 0.0
        assert branch == "k0_zero"

    def test_warmup_branch_when_buffer_empty(self):
        """K>0 + empty buffer → warmup_traditional branch."""
        al = AnticipatoryLearning(window_size=3)
        sol = _make_solution()
        val, branch = al._compute_lambda_k_with_branch(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        assert branch == "warmup_traditional"
        assert 0.0 <= val <= 0.5

    def test_kperiod_branch_when_buffer_populated(self):
        """K>0 + populated buffer → kperiod_sum branch (the W17-2 fix)."""
        al = AnticipatoryLearning(window_size=3)
        for _ in range(3):
            al.record_kf_residual(1.5)
        sol = _make_solution()
        val, branch = al._compute_lambda_k_with_branch(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        assert branch == "kperiod_sum"
        assert val > 0.0
        assert val <= 0.5


class TestLambdaTraceRecordsBranch:
    """W17-2: trace row + CSV include lambda_k_branch column."""

    def test_trace_row_has_branch_column(self):
        al = AnticipatoryLearning(window_size=2)
        for _ in range(2):
            al.record_kf_residual(1.0)
        sol = _make_solution()
        al.compute_anticipatory_learning_rate(
            sol, 0.0, 1.0, 0.0, 1.0, current_time=5,
        )
        row = al.get_lambda_trace()[-1]
        assert "lambda_k_branch" in row
        assert row["lambda_k_branch"] == "kperiod_sum"

    def test_csv_header_includes_branch(self):
        assert "lambda_k_branch" in AnticipatoryLearning.LAMBDA_TRACE_CSV_HEADER
