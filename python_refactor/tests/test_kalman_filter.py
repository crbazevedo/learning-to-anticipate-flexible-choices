"""
Tests for Kalman Filter implementation.
"""

import pytest
import numpy as np
from src.algorithms.kalman_filter import (
    KalmanParams, kalman_prediction, kalman_update, kalman_filter,
    initialize_kalman_matrices, create_kalman_params,
    update_measurement_noise, get_portfolio_state, get_portfolio_prediction,
    get_error_covariance, get_prediction_error_covariance
)


class TestKalmanParams:
    """Test KalmanParams dataclass."""
    
    def test_kalman_params_initialization(self):
        """Test KalmanParams initialization with default values."""
        params = KalmanParams()
        
        assert params.x is not None
        assert params.x_next is not None
        assert params.u is not None
        assert params.P is not None
        assert params.P_next is not None
        
        assert params.x.shape == (4,)
        assert params.x_next.shape == (4,)
        assert params.u.shape == (4,)
        assert params.P.shape == (4, 4)
        assert params.P_next.shape == (4, 4)
    
    def test_kalman_params_custom_initialization(self):
        """Test KalmanParams initialization with custom values."""
        x = np.array([1.0, 0.1, 0.5, 0.05])
        P = np.eye(4) * 0.01
        
        params = KalmanParams(x=x, P=P)
        
        np.testing.assert_array_equal(params.x, x)
        np.testing.assert_array_equal(params.P, P)


class TestKalmanMatrices:
    """Test Kalman filter matrix initialization."""
    
    def test_initialize_kalman_matrices(self):
        """Test initialization of Kalman filter matrices.

        State-vector ordering per paper Eq (11) with m=2:
            x = [ROI, risk, ROI_velocity, risk_velocity]
                [ 0,   1,        2,             3      ]
        """
        F, H, R = initialize_kalman_matrices()

        # Check state transition matrix F (4x4) — paper-canonical
        assert F.shape == (4, 4)
        assert F[0, 0] == 1.0  # ROI_next = ROI + ROI_velocity
        assert F[0, 2] == 1.0  # ROI_velocity is at index 2 under paper Eq (11)
        assert F[1, 1] == 1.0  # risk_next = risk + risk_velocity
        assert F[1, 3] == 1.0  # risk_velocity is at index 3
        assert F[0, 1] == 0.0  # ROI does NOT mix with risk under constant-velocity dynamics
        assert F[1, 0] == 0.0
        # Identity on velocity rows
        assert F[2, 2] == 1.0  # ROI_velocity_next = ROI_velocity
        assert F[3, 3] == 1.0  # risk_velocity_next = risk_velocity

        # Check measurement matrix H (2x4) — observe ROI from x[0], risk from x[1]
        assert H.shape == (2, 4)
        assert H[0, 0] == 1.0  # Observe ROI
        assert H[1, 1] == 1.0  # Observe risk (NOT x[2] — that's ROI_velocity)
        
        # Check measurement noise covariance R (2x2)
        assert R.shape == (2, 2)
        assert R[0, 0] > 0  # ROI measurement noise
        assert R[1, 1] > 0  # Risk measurement noise


class TestKalmanPrediction:
    """Test Kalman filter prediction step."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_kalman_prediction(self):
        """Test Kalman filter prediction step."""
        # Store initial values
        initial_x = self.params.x.copy()
        initial_P = self.params.P.copy()
        
        # Perform prediction
        kalman_prediction(self.params)
        
        # Check that prediction follows state transition model
        expected_x_next = self.params.F @ initial_x + self.params.u
        np.testing.assert_array_almost_equal(self.params.x_next, expected_x_next)
        
        # Check that covariance prediction is correct
        expected_P_next = self.params.F @ initial_P @ self.params.F.T
        np.testing.assert_array_almost_equal(self.params.P_next, expected_P_next)
        
        # Check that covariance was updated (should be different)
        assert not np.array_equal(self.params.P_next, initial_P)


class TestKalmanUpdate:
    """Test Kalman filter update step."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        # Perform initial prediction
        kalman_prediction(self.params)
    
    def test_kalman_update(self):
        """Test Kalman filter update step."""
        # Store prediction values
        x_next = self.params.x_next.copy()
        P_next = self.params.P_next.copy()
        
        # Create measurement
        measurement = np.array([0.12, 0.06])  # [ROI, risk]
        
        # Perform update
        kalman_update(self.params, measurement)
        
        # Check that state was updated
        assert not np.array_equal(self.params.x, x_next)
        assert not np.array_equal(self.params.P, P_next)
        
        # Check that state is reasonable (paper Eq 11 ordering: x[1] = risk)
        assert 0.0 <= self.params.x[0] <= 1.0  # ROI should be positive
        assert 0.0 <= self.params.x[1] <= 1.0  # Risk should be positive


class TestKalmanFilter:
    """Test complete Kalman filter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_kalman_filter_complete(self):
        """Test complete Kalman filter step."""
        # Store initial values
        initial_x = self.params.x.copy()
        initial_P = self.params.P.copy()
        
        # Create measurement
        measurement = np.array([0.12, 0.06])
        
        # Perform complete filter step
        kalman_filter(self.params, measurement)
        
        # Check that both prediction and update occurred
        assert not np.array_equal(self.params.x, initial_x)
        assert not np.array_equal(self.params.P, initial_P)
        
        # Check that next state was computed
        assert self.params.x_next is not None
        assert self.params.P_next is not None


class TestKalmanUtilities:
    """Test Kalman filter utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
    
    def test_create_kalman_params(self):
        """Test creation of Kalman parameters (paper Eq 11 ordering)."""
        params = create_kalman_params(initial_roi=0.15, initial_risk=0.08)

        # Check initial state — paper-canonical [ROI, risk, ROI_vel, risk_vel]
        assert params.x[0] == 0.15  # ROI
        assert params.x[1] == 0.08  # risk (NOT x[2] — that's ROI_velocity)
        assert params.x[2] == 0.0   # ROI velocity
        assert params.x[3] == 0.0   # risk velocity
        
        # Check that matrices are set
        assert params.F is not None
        assert params.H is not None
        assert params.R is not None
    
    def test_update_measurement_noise(self):
        """Test updating measurement noise covariance."""
        roi_variance = 0.02
        risk_variance = 0.03
        covariance = 0.005
        
        update_measurement_noise(self.params, roi_variance, risk_variance, covariance)
        
        expected_R = np.array([
            [roi_variance, covariance],
            [covariance, risk_variance]
        ])
        
        np.testing.assert_array_equal(self.params.R, expected_R)
    
    def test_get_portfolio_state(self):
        """Test extracting portfolio state from Kalman filter (Eq 11 ordering)."""
        roi, risk = get_portfolio_state(self.params)

        assert roi == self.params.x[0]
        assert risk == self.params.x[1]  # paper-canonical: risk is x[1]

    def test_get_portfolio_prediction(self):
        """Test extracting portfolio prediction (Eq 11 ordering)."""
        # Perform prediction first
        kalman_prediction(self.params)

        roi_pred, risk_pred = get_portfolio_prediction(self.params)

        assert roi_pred == self.params.x_next[0]
        assert risk_pred == self.params.x_next[1]  # paper-canonical: risk is [1]
    
    def test_get_error_covariance(self):
        """Test getting error covariance matrix."""
        covar = get_error_covariance(self.params)
        
        np.testing.assert_array_equal(covar, self.params.P)
        assert covar is not self.params.P  # Should be a copy
    
    def test_get_prediction_error_covariance(self):
        """Test getting prediction error covariance matrix."""
        # Perform prediction first
        kalman_prediction(self.params)
        
        covar = get_prediction_error_covariance(self.params)
        
        np.testing.assert_array_equal(covar, self.params.P_next)
        assert covar is not self.params.P_next  # Should be a copy


class TestKalmanFilterIntegration:
    """Test Kalman filter integration scenarios."""
    
    def test_multiple_filter_steps(self):
        """Test multiple Kalman filter steps."""
        params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        
        measurements = [
            np.array([0.12, 0.06]),
            np.array([0.11, 0.07]),
            np.array([0.13, 0.05])
        ]
        
        for measurement in measurements:
            kalman_filter(params, measurement)

            # Check that state is reasonable (paper Eq 11 ordering)
            assert 0.0 <= params.x[0] <= 1.0  # ROI = x[0]
            assert 0.0 <= params.x[1] <= 1.0  # risk = x[1]
            assert params.P[0, 0] > 0  # ROI variance
            assert params.P[1, 1] > 0  # risk variance (was P[2,2] under interleaved)
    
    def test_kalman_filter_convergence(self):
        """Test that Kalman filter converges with consistent measurements."""
        params = create_kalman_params(initial_roi=0.1, initial_risk=0.05)
        
        # Consistent measurement
        measurement = np.array([0.12, 0.06])
        
        # Multiple filter steps
        for _ in range(10):
            kalman_filter(params, measurement)
        
        # Check convergence — measurement is [ROI, risk]; paper Eq (11)
        # ordering puts those at x[0] and x[1].
        np.testing.assert_array_almost_equal(
            params.x[[0, 1]], measurement, decimal=2
        )


class TestPaperEq11Canonical:
    """Tests asserting paper Eq (11) state-vector canonical ordering.

    Reference: Azevedo, C. R. B., & Von Zuben, F. J. (2015). Learning to
    Anticipate Flexible Choices in Multiple Criteria Decision-Making
    Under Uncertainty. IEEE Transactions on Cybernetics. §IV-A:

        z_t^+ = (z_{1,t} ... z_{m,t}  ż_{1,t} ... ż_{m,t})^T    (11)

    For m=2 (objectives = ROI, risk):

        x = [ROI, risk, ROI_velocity, risk_velocity]
            [ 0,   1,        2,             3      ]

    These tests pin the convention so future refactors can't silently
    revert to the interleaved [ROI, ROI_vel, risk, risk_vel] layout
    that previously caused sms_emoa.py:499 to read ROI_velocity as
    "future_risk" (the W1-1 keystone bug).
    """

    def test_kalman_state_vector_matches_paper_eq11_canonical_ordering(self):
        """Predict one step with known velocities; assert each component lands at the paper-canonical index.

        Setup: x_0 = [ROI=1.0, risk=0.5, ROI_vel=0.1, risk_vel=-0.05]
        After kalman_prediction (constant-velocity F), x_1 should be:
          - x_1[0] = ROI + ROI_vel       = 1.1
          - x_1[1] = risk + risk_vel     = 0.45
          - x_1[2] = ROI_vel preserved   = 0.1
          - x_1[3] = risk_vel preserved  = -0.05

        Crucially, x_1[1] (NOT x_1[2]) is the "future risk" — sms_emoa.py
        before W1-1 read x_next[2] under sms_emoa's own paper-canonical
        F matrix, silently treating ROI_velocity as risk.
        """
        params = create_kalman_params(initial_roi=1.0, initial_risk=0.5)
        # Inject known velocities at paper-canonical indices 2 and 3.
        params.x[2] = 0.1    # ROI_velocity
        params.x[3] = -0.05  # risk_velocity
        params.u = np.zeros(4)  # no exogenous control

        kalman_prediction(params)

        # Future ROI = ROI + ROI_velocity
        np.testing.assert_allclose(params.x_next[0], 1.1, atol=1e-9)
        # Future risk = risk + risk_velocity — THIS is the index that
        # sms_emoa.py:499 was previously misreading as x_next[2].
        np.testing.assert_allclose(params.x_next[1], 0.45, atol=1e-9)
        # Velocities preserved under constant-velocity dynamics.
        np.testing.assert_allclose(params.x_next[2], 0.1, atol=1e-9)
        np.testing.assert_allclose(params.x_next[3], -0.05, atol=1e-9)

    def test_state_vector_index_2_is_roi_velocity_not_risk(self):
        """Pin: x[2] is ROI_velocity, NOT risk. Regression guard for W1-1."""
        params = create_kalman_params(initial_roi=0.42, initial_risk=0.17)
        # x[0]=0.42 (ROI), x[1]=0.17 (risk), x[2]=0 (ROI_vel), x[3]=0 (risk_vel)
        assert params.x[0] == 0.42, "x[0] must be ROI per paper Eq (11)"
        assert params.x[1] == 0.17, "x[1] must be risk per paper Eq (11)"
        assert params.x[2] == 0.0, "x[2] must be ROI_velocity per paper Eq (11)"
        assert params.x[3] == 0.0, "x[3] must be risk_velocity per paper Eq (11)"

    def test_get_portfolio_state_reads_paper_canonical_indices(self):
        """get_portfolio_state must return (x[0], x[1]) — ROI and risk per Eq (11)."""
        params = create_kalman_params(initial_roi=0.33, initial_risk=0.22)
        roi, risk = get_portfolio_state(params)
        assert roi == 0.33
        assert risk == 0.22  # NOT 0.0 (which would be ROI_velocity at x[2])

    def test_get_portfolio_prediction_reads_paper_canonical_indices(self):
        """get_portfolio_prediction must return (x_next[0], x_next[1])."""
        params = create_kalman_params(initial_roi=0.5, initial_risk=0.3)
        params.x[2] = 0.05    # ROI_velocity
        params.x[3] = -0.02   # risk_velocity
        params.u = np.zeros(4)
        kalman_prediction(params)
        roi_pred, risk_pred = get_portfolio_prediction(params)
        np.testing.assert_allclose(roi_pred, 0.55, atol=1e-9)
        np.testing.assert_allclose(risk_pred, 0.28, atol=1e-9)
