"""
Kalman Filter implementation for portfolio optimization.

This module implements the Kalman filter for state estimation of portfolio
ROI and risk, including prediction and update steps.

State-vector ordering. The canonical ordering follows the paper:

    Azevedo, C. R. B., & Von Zuben, F. J. (2015). Learning to Anticipate
    Flexible Choices in Multiple Criteria Decision-Making Under
    Uncertainty. IEEE Transactions on Cybernetics. §IV-A Eq (11):

        z_t^+ = (z_{1,t} ... z_{m,t}  ż_{1,t} ... ż_{m,t})^T

For the m=2 portfolio case (objectives = ROI, risk) this is:

        x = [ROI, risk, ROI_velocity, risk_velocity]

That is: all m objectives first, then all m velocities. This module
encodes that convention throughout — F, x init, and the get_portfolio_*
readers. Callers MUST honour the same convention to avoid silent
miswiring (a class of bug that previously affected sms_emoa.py:499
under W1-1 of the dfg-harness wave plan).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class KalmanParams:
    """
    Kalman filter parameters equivalent to C++ Kalman_params struct.
    
    Attributes:
        F: State transition matrix (static)
        H: Measurement matrix (static) 
        R: Measurement noise covariance matrix (static)
        x: Current state vector
        x_next: Next state vector
        u: Control input vector
        P: Current error covariance matrix
        P_next: Next error covariance matrix
    """
    # Static matrices (shared across all instances)
    F: Optional[np.ndarray] = None  # State transition matrix
    H: Optional[np.ndarray] = None  # Measurement matrix
    R: Optional[np.ndarray] = None  # Measurement noise covariance
    
    # Instance variables
    x: Optional[np.ndarray] = None      # Current state vector
    x_next: Optional[np.ndarray] = None # Next state vector
    u: Optional[np.ndarray] = None      # Control input vector
    P: Optional[np.ndarray] = None      # Current error covariance matrix
    P_next: Optional[np.ndarray] = None # Next error covariance matrix
    
    def __post_init__(self):
        """Initialize with default values if not provided."""
        if self.x is None:
            # Paper Eq (11) ordering: [ROI, risk, ROI_velocity, risk_velocity]
            self.x = np.zeros(4)
        if self.x_next is None:
            self.x_next = np.zeros(4)
        if self.u is None:
            self.u = np.zeros(4)
        if self.P is None:
            self.P = np.eye(4) * 0.1  # Initial covariance
        if self.P_next is None:
            self.P_next = np.eye(4) * 0.1


def kalman_prediction(params: KalmanParams) -> None:
    """
    Perform Kalman filter prediction step.
    
    Args:
        params: Kalman filter parameters
    """
    # State prediction: x_next = F * x + u
    params.x_next = params.F @ params.x + params.u
    
    # Covariance prediction: P_next = F * P * F^T
    params.P_next = params.F @ params.P @ params.F.T


def kalman_update(params: KalmanParams, measurement: np.ndarray) -> float:
    """
    Perform Kalman filter update step.

    Args:
        params: Kalman filter parameters
        measurement: Measurement vector [ROI, risk]

    Returns:
        Squared-residual scalar (innovation^T @ innovation) — W17-2
        addition. Caller (typically AnticipatoryLearning) accumulates
        this per-period to feed λ^K per thesis Eq 6.9. Pre-W17-2 the
        update returned None and the residual was lost; per W16-1-CARRY-1
        retro this is why λ^K stayed in warm-up fallback throughout the
        W16-5 smoke. Return value is additive (callers that ignore it
        are unaffected).
    """
    # Create measurement vector Z
    Z = np.array([measurement[0], measurement[1]])  # [ROI, risk]

    # Innovation: y = Z - H * x_next
    y = Z - params.H @ params.x_next

    # Innovation covariance: S = H * P_next * H^T + R
    S = params.H @ params.P_next @ params.H.T + params.R

    # Kalman gain: K = P_next * H^T * S^(-1)
    K = params.P_next @ params.H.T @ np.linalg.inv(S)

    # State update: x = x_next + K * y
    params.x = params.x_next + K @ y

    # Covariance update: P = (I - K * H) * P_next
    I = np.eye(params.F.shape[0])
    params.P = (I - K @ params.H) @ params.P_next

    # W17-2: return squared-residual scalar for λ^K (Eq 6.9).
    # innovation^T @ innovation = sum of squared per-objective residuals.
    return float(y @ y)


def kalman_filter(params: KalmanParams, measurement: np.ndarray) -> float:
    """
    Perform complete Kalman filter step (prediction + update).

    Args:
        params: Kalman filter parameters
        measurement: Measurement vector [ROI, risk]

    Returns:
        Squared-residual from the update step (W17-2; forwards
        kalman_update's return value).
    """
    kalman_prediction(params)
    return kalman_update(params, measurement)


def initialize_kalman_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize Kalman filter matrices following paper Eq (11) ordering.

    State vector layout (paper Eq 11 with m=2):
        x = [ROI, risk, ROI_velocity, risk_velocity]
            [ 0,   1,        2,             3      ]

    Returns:
        Tuple of (F, H, R) matrices
    """
    # State transition matrix F (4x4) — constant-velocity dynamics.
    # Indices: [0]=ROI, [1]=risk, [2]=ROI_vel, [3]=risk_vel.
    F = np.array([
        [1.0, 0.0, 1.0, 0.0],  # ROI_next = ROI + ROI_velocity
        [0.0, 1.0, 0.0, 1.0],  # risk_next = risk + risk_velocity
        [0.0, 0.0, 1.0, 0.0],  # ROI_velocity_next = ROI_velocity
        [0.0, 0.0, 0.0, 1.0]   # risk_velocity_next = risk_velocity
    ])

    # Measurement matrix H (2x4)
    # We observe ROI (x[0]) and risk (x[1]); velocities are latent.
    H = np.array([
        [1.0, 0.0, 0.0, 0.0],  # Observe ROI from x[0]
        [0.0, 1.0, 0.0, 0.0]   # Observe risk from x[1]
    ])
    
    # Measurement noise covariance R (2x2).
    # W22 R-INCONSISTENCY FIX (2026-05-18):
    # Pre-W22, this function set R off-diagonal = 0.005, while
    # sms_emoa._initialize_kalman_state set R off-diagonal = 0.0.
    # As evolution proceeds, offspring (going through Solution.__init__
    # → Portfolio.initialize_kalman_filter → create_kalman_params →
    # initialize_kalman_matrices) had a DIFFERENT noise model than the
    # initial population. Kalman gain calculation diverged across the
    # population, affecting StochasticParams-driven selection and
    # anticipation differently than intended.
    #
    # Fix: harmonize to the sms_emoa._initialize_kalman_state value
    # (zero off-diagonal). Both code paths now produce identical R.
    R = np.array([
        [0.01, 0.0],  # ROI measurement noise (off-diagonal zero per W22 fix)
        [0.0, 0.01]   # Risk measurement noise
    ])

    return F, H, R


def create_kalman_params(initial_roi: float = 0.0, initial_risk: float = 0.0) -> KalmanParams:
    """
    Create and initialize Kalman filter parameters.
    
    Args:
        initial_roi: Initial ROI value
        initial_risk: Initial risk value
        
    Returns:
        Initialized KalmanParams object
    """
    # Initialize matrices
    F, H, R = initialize_kalman_matrices()
    
    # Create instance with matrices
    params = KalmanParams(F=F, H=H, R=R)

    # Initialize state vector (paper Eq 11 ordering):
    # [ROI, risk, ROI_velocity, risk_velocity]
    params.x = np.array([initial_roi, initial_risk, 0.0, 0.0])
    params.x_next = params.x.copy()
    
    # Initialize control input (zero for now)
    params.u = np.zeros(4)
    
    # Initialize covariance matrices
    params.P = np.eye(4) * 0.1
    params.P_next = params.P.copy()
    
    return params


def update_measurement_noise(params: KalmanParams, roi_variance: float, risk_variance: float, 
                           covariance: float = 0.0) -> None:
    """
    Update measurement noise covariance matrix.
    
    Args:
        params: Kalman filter parameters
        roi_variance: ROI measurement variance
        risk_variance: Risk measurement variance
        covariance: ROI-risk measurement covariance
    """
    params.R = np.array([
        [roi_variance, covariance],
        [covariance, risk_variance]
    ])


def get_portfolio_state(params: KalmanParams) -> tuple[float, float]:
    """
    Extract current portfolio ROI and risk from Kalman state.

    Per paper Eq (11) ordering: x = [ROI, risk, ROI_vel, risk_vel].

    Args:
        params: Kalman filter parameters

    Returns:
        Tuple of (ROI, risk)
    """
    return params.x[0], params.x[1]  # ROI = x[0], risk = x[1]


def get_portfolio_prediction(params: KalmanParams) -> tuple[float, float]:
    """
    Extract predicted portfolio ROI and risk from Kalman state.

    Per paper Eq (11) ordering: x_next = [ROI, risk, ROI_vel, risk_vel].

    Args:
        params: Kalman filter parameters

    Returns:
        Tuple of (predicted_ROI, predicted_risk)
    """
    return params.x_next[0], params.x_next[1]  # ROI = [0], risk = [1]


def get_error_covariance(params: KalmanParams) -> np.ndarray:
    """
    Get current error covariance matrix.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Error covariance matrix
    """
    return params.P.copy()


def get_prediction_error_covariance(params: KalmanParams) -> np.ndarray:
    """
    Get prediction error covariance matrix.
    
    Args:
        params: Kalman filter parameters
        
    Returns:
        Prediction error covariance matrix
    """
    return params.P_next.copy() 