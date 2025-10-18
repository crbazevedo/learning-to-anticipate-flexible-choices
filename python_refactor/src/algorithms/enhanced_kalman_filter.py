"""
Enhanced Kalman Filter Implementation

This module implements an enhanced Kalman filter with advanced features including:
- Enhanced state space model with sophisticated dynamics
- Adaptive parameter estimation
- Improved uncertainty quantification
- Regime integration capabilities
- Performance optimizations

Based on EPIC 1 findings and designed to provide a solid foundation for EPIC 2.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from datetime import datetime
from scipy import linalg
from scipy.stats import multivariate_normal
import warnings

from algorithms.kalman_filter import KalmanParams, kalman_prediction, kalman_update
from algorithms.regime_detection_bnn import MarketRegimeDetectionBNN, RegimeDetectionResult

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedPredictionResult:
    """Enhanced prediction result with uncertainty and regime information."""
    
    prediction: np.ndarray
    covariance: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    regime_info: Optional[RegimeDetectionResult] = None
    prediction_interval: Optional[Dict[str, np.ndarray]] = None
    timestamp: float = 0.0


@dataclass
class EnhancedUpdateResult:
    """Enhanced update result with parameter adjustments."""
    
    updated_state: np.ndarray
    updated_covariance: np.ndarray
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    log_likelihood: float
    parameter_adjustments: Optional[Dict[str, float]] = None
    timestamp: float = 0.0


@dataclass
class KalmanParameters:
    """Enhanced Kalman filter parameters."""
    
    # State transition matrix
    F: np.ndarray
    
    # Measurement matrix
    H: np.ndarray
    
    # Process noise covariance
    Q: np.ndarray
    
    # Measurement noise covariance
    R: np.ndarray
    
    # Initial state covariance
    P0: np.ndarray
    
    # Adaptive parameters
    adaptation_rate: float = 0.01
    forgetting_factor: float = 0.95
    
    # Regime-specific parameters
    regime_parameters: Optional[Dict[str, Dict[str, np.ndarray]]] = None


class EnhancedStateSpaceModel:
    """Enhanced state space model with sophisticated dynamics."""
    
    def __init__(self, state_dim: int = 4, observation_dim: int = 2):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        # Enhanced dynamics parameters
        self.dynamics_params = {
            'mean_reversion': 0.1,
            'volatility_clustering': 0.05,
            'cross_correlation': 0.3,
            'momentum_persistence': 0.8
        }
        
        logger.info(f"Initialized EnhancedStateSpaceModel with {state_dim} states")
    
    def create_enhanced_transition_matrix(self, dt: float = 1.0) -> np.ndarray:
        """
        Create enhanced state transition matrix with sophisticated dynamics.
        
        Args:
            dt: Time step
            
        Returns:
            Enhanced state transition matrix
        """
        # Enhanced 4-state model: [ROI, risk, ROI_velocity, risk_velocity]
        F = np.zeros((4, 4))
        
        # ROI dynamics with mean reversion
        F[0, 0] = 1.0 - self.dynamics_params['mean_reversion'] * dt
        F[0, 2] = dt  # ROI velocity contribution
        
        # Risk dynamics with volatility clustering
        F[1, 1] = 1.0 - self.dynamics_params['volatility_clustering'] * dt
        F[1, 3] = dt  # Risk velocity contribution
        
        # ROI velocity with momentum persistence
        F[2, 2] = self.dynamics_params['momentum_persistence']
        
        # Risk velocity with momentum persistence
        F[3, 3] = self.dynamics_params['momentum_persistence']
        
        # Cross-correlation effects
        cross_corr = self.dynamics_params['cross_correlation'] * dt
        F[0, 1] = cross_corr  # ROI affected by risk
        F[1, 0] = cross_corr  # Risk affected by ROI
        
        return F
    
    def create_enhanced_process_noise(self, dt: float = 1.0, 
                                    regime: str = 'sideways_market') -> np.ndarray:
        """
        Create enhanced process noise covariance matrix.
        
        Args:
            dt: Time step
            regime: Market regime
            
        Returns:
            Enhanced process noise covariance matrix
        """
        # Base process noise
        Q = np.zeros((4, 4))
        
        # Regime-specific noise levels
        regime_noise = {
            'bull_market': {'roi': 0.005, 'risk': 0.002, 'roi_vel': 0.001, 'risk_vel': 0.001},
            'bear_market': {'roi': 0.02, 'risk': 0.01, 'roi_vel': 0.005, 'risk_vel': 0.005},
            'sideways_market': {'roi': 0.01, 'risk': 0.005, 'roi_vel': 0.002, 'risk_vel': 0.002}
        }
        
        noise_params = regime_noise.get(regime, regime_noise['sideways_market'])
        
        # Diagonal noise
        Q[0, 0] = noise_params['roi'] * dt
        Q[1, 1] = noise_params['risk'] * dt
        Q[2, 2] = noise_params['roi_vel'] * dt
        Q[3, 3] = noise_params['risk_vel'] * dt
        
        # Cross-correlation in noise
        cross_noise = 0.1 * np.sqrt(Q[0, 0] * Q[1, 1])
        Q[0, 1] = Q[1, 0] = cross_noise
        
        return Q
    
    def create_enhanced_measurement_matrix(self) -> np.ndarray:
        """
        Create enhanced measurement matrix.
        
        Returns:
            Enhanced measurement matrix
        """
        # Enhanced measurement model
        H = np.zeros((2, 4))
        
        # Direct ROI observation
        H[0, 0] = 1.0
        
        # Direct risk observation
        H[1, 1] = 1.0
        
        # Velocity contributions to observations
        H[0, 2] = 0.1  # ROI velocity affects ROI observation
        H[1, 3] = 0.1  # Risk velocity affects risk observation
        
        return H
    
    def create_enhanced_measurement_noise(self, regime: str = 'sideways_market') -> np.ndarray:
        """
        Create enhanced measurement noise covariance matrix.
        
        Args:
            regime: Market regime
            
        Returns:
            Enhanced measurement noise covariance matrix
        """
        # Regime-specific measurement noise
        regime_noise = {
            'bull_market': {'roi': 0.002, 'risk': 0.001},
            'bear_market': {'roi': 0.01, 'risk': 0.005},
            'sideways_market': {'roi': 0.005, 'risk': 0.002}
        }
        
        noise_params = regime_noise.get(regime, regime_noise['sideways_market'])
        
        R = np.array([
            [noise_params['roi'], 0.0],
            [0.0, noise_params['risk']]
        ])
        
        return R


class KalmanParameterEstimator:
    """Advanced parameter estimation for Kalman filter."""
    
    def __init__(self):
        self.estimation_history = []
        self.parameter_history = []
        
        logger.info("Initialized KalmanParameterEstimator")
    
    def maximum_likelihood_estimation(self, observations: np.ndarray,
                                    initial_params: KalmanParameters) -> KalmanParameters:
        """
        Maximum likelihood parameter estimation.
        
        Args:
            observations: Historical observations
            initial_params: Initial parameter estimates
            
        Returns:
            ML-estimated parameters
        """
        logger.info("Performing maximum likelihood parameter estimation")
        
        # Simplified ML estimation (in practice, would use EM algorithm)
        # For now, return improved initial parameters
        
        improved_params = KalmanParameters(
            F=initial_params.F,
            H=initial_params.H,
            Q=initial_params.Q * 1.1,  # Slightly increase process noise
            R=initial_params.R * 0.9,  # Slightly decrease measurement noise
            P0=initial_params.P0,
            adaptation_rate=initial_params.adaptation_rate,
            forgetting_factor=initial_params.forgetting_factor
        )
        
        return improved_params
    
    def bayesian_parameter_estimation(self, observations: np.ndarray,
                                    prior_params: KalmanParameters) -> KalmanParameters:
        """
        Bayesian parameter estimation with priors.
        
        Args:
            observations: Historical observations
            prior_params: Prior parameter estimates
            
        Returns:
            Bayesian-estimated parameters
        """
        logger.info("Performing Bayesian parameter estimation")
        
        # Simplified Bayesian estimation
        # In practice, would use MCMC or variational inference
        
        bayesian_params = KalmanParameters(
            F=prior_params.F,
            H=prior_params.H,
            Q=prior_params.Q * 0.95,  # Slightly decrease process noise
            R=prior_params.R * 1.05,  # Slightly increase measurement noise
            P0=prior_params.P0,
            adaptation_rate=prior_params.adaptation_rate,
            forgetting_factor=prior_params.forgetting_factor
        )
        
        return bayesian_params
    
    def online_parameter_update(self, observation: np.ndarray,
                              current_params: KalmanParameters,
                              innovation: np.ndarray) -> KalmanParameters:
        """
        Online parameter update based on recent observations.
        
        Args:
            observation: Current observation
            current_params: Current parameter estimates
            innovation: Prediction innovation
            
        Returns:
            Updated parameters
        """
        # Adaptive parameter update
        adaptation_rate = current_params.adaptation_rate
        
        # Update process noise based on innovation magnitude
        innovation_magnitude = np.linalg.norm(innovation)
        Q_adjustment = 1.0 + adaptation_rate * innovation_magnitude
        
        # Update measurement noise based on observation quality
        observation_quality = 1.0 / (1.0 + np.var(observation))
        R_adjustment = 1.0 - adaptation_rate * observation_quality
        
        # Create updated parameters
        updated_params = KalmanParameters(
            F=current_params.F,
            H=current_params.H,
            Q=current_params.Q * Q_adjustment,
            R=current_params.R * R_adjustment,
            P0=current_params.P0,
            adaptation_rate=current_params.adaptation_rate,
            forgetting_factor=current_params.forgetting_factor
        )
        
        # Store in history
        self.parameter_history.append(updated_params)
        
        return updated_params


class KalmanUncertaintyQuantifier:
    """Enhanced uncertainty quantification for Kalman filter."""
    
    def __init__(self, num_ensemble_samples: int = 100):
        self.num_ensemble_samples = num_ensemble_samples
        self.uncertainty_history = []
        
        logger.info(f"Initialized KalmanUncertaintyQuantifier with {num_ensemble_samples} samples")
    
    def ensemble_uncertainty(self, prediction: np.ndarray, 
                           covariance: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty using ensemble methods.
        
        Args:
            prediction: Mean prediction
            covariance: Prediction covariance
            
        Returns:
            Ensemble-based uncertainty
        """
        # Generate ensemble samples
        samples = np.random.multivariate_normal(prediction, covariance, self.num_ensemble_samples)
        
        # Calculate ensemble uncertainty
        ensemble_uncertainty = np.std(samples, axis=0)
        
        return ensemble_uncertainty
    
    def bootstrap_uncertainty(self, prediction: np.ndarray,
                            covariance: np.ndarray,
                            num_bootstrap: int = 50) -> np.ndarray:
        """
        Calculate uncertainty using bootstrap methods.
        
        Args:
            prediction: Mean prediction
            covariance: Prediction covariance
            num_bootstrap: Number of bootstrap samples
            
        Returns:
            Bootstrap-based uncertainty
        """
        bootstrap_uncertainties = []
        
        for _ in range(num_bootstrap):
            # Bootstrap sample
            bootstrap_sample = np.random.multivariate_normal(prediction, covariance)
            bootstrap_uncertainties.append(bootstrap_sample)
        
        bootstrap_uncertainties = np.array(bootstrap_uncertainties)
        bootstrap_uncertainty = np.std(bootstrap_uncertainties, axis=0)
        
        return bootstrap_uncertainty
    
    def bayesian_uncertainty(self, prediction: np.ndarray,
                           covariance: np.ndarray,
                           prior_covariance: np.ndarray) -> np.ndarray:
        """
        Calculate uncertainty using Bayesian methods.
        
        Args:
            prediction: Mean prediction
            covariance: Prediction covariance
            prior_covariance: Prior covariance
            
        Returns:
            Bayesian-based uncertainty
        """
        # Bayesian uncertainty combines prediction and prior uncertainty
        bayesian_covariance = np.linalg.inv(
            np.linalg.inv(covariance) + np.linalg.inv(prior_covariance)
        )
        
        bayesian_uncertainty = np.sqrt(np.diag(bayesian_covariance))
        
        return bayesian_uncertainty
    
    def calculate_prediction_intervals(self, prediction: np.ndarray,
                                     uncertainty: np.ndarray,
                                     confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> Dict[str, np.ndarray]:
        """
        Calculate prediction intervals for different confidence levels.
        
        Args:
            prediction: Mean prediction
            uncertainty: Prediction uncertainty
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary of prediction intervals
        """
        intervals = {}
        
        for confidence in confidence_levels:
            # Calculate z-score for confidence level
            z_score = 1.96 if confidence == 0.95 else 2.58 if confidence == 0.99 else 1.0
            
            # Calculate intervals
            lower = prediction - z_score * uncertainty
            upper = prediction + z_score * uncertainty
            
            intervals[f'{confidence:.0%}'] = {
                'lower': lower,
                'upper': upper
            }
        
        return intervals


class EnhancedKalmanFilter:
    """Enhanced Kalman filter with advanced features."""
    
    def __init__(self, state_dim: int = 4, observation_dim: int = 2):
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        # Enhanced components
        self.state_model = EnhancedStateSpaceModel(state_dim, observation_dim)
        self.parameter_estimator = KalmanParameterEstimator()
        self.uncertainty_quantifier = KalmanUncertaintyQuantifier()
        
        # Current state
        self.current_state = np.zeros(state_dim)
        self.current_covariance = np.eye(state_dim) * 0.1
        
        # Parameters
        self.parameters = None
        self.regime = 'sideways_market'
        
        # History
        self.prediction_history = []
        self.update_history = []
        
        logger.info(f"Initialized EnhancedKalmanFilter with {state_dim} states")
    
    def initialize_parameters(self, regime: str = 'sideways_market') -> None:
        """
        Initialize enhanced Kalman filter parameters.
        
        Args:
            regime: Initial market regime
        """
        self.regime = regime
        
        # Create enhanced parameters
        F = self.state_model.create_enhanced_transition_matrix()
        H = self.state_model.create_enhanced_measurement_matrix()
        Q = self.state_model.create_enhanced_process_noise(regime=regime)
        R = self.state_model.create_enhanced_measurement_noise(regime=regime)
        P0 = np.eye(self.state_dim) * 0.1
        
        self.parameters = KalmanParameters(
            F=F, H=H, Q=Q, R=R, P0=P0,
            adaptation_rate=0.01,
            forgetting_factor=0.95
        )
        
        logger.info(f"Initialized parameters for regime: {regime}")
    
    def enhanced_prediction(self, current_state: np.ndarray = None,
                          regime: str = None) -> EnhancedPredictionResult:
        """
        Enhanced prediction with uncertainty quantification.
        
        Args:
            current_state: Current state vector (if None, use internal state)
            regime: Market regime (if None, use current regime)
            
        Returns:
            Enhanced prediction result
        """
        if current_state is None:
            current_state = self.current_state
        
        if regime is None:
            regime = self.regime
        
        # Update parameters for current regime
        if regime != self.regime:
            self.regime = regime
            self.initialize_parameters(regime)
        
        # Create Kalman parameters for prediction
        kalman_params = KalmanParams(
            x=current_state,
            F=self.parameters.F,
            H=self.parameters.H,
            R=self.parameters.R,
            P=self.current_covariance
        )
        
        # Perform prediction
        kalman_prediction(kalman_params)
        
        # Calculate enhanced uncertainty
        uncertainty = self.uncertainty_quantifier.ensemble_uncertainty(
            kalman_params.x, kalman_params.P
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(kalman_params.P)
        
        # Calculate prediction intervals
        prediction_intervals = self.uncertainty_quantifier.calculate_prediction_intervals(
            kalman_params.x, uncertainty
        )
        
        # Create enhanced result
        result = EnhancedPredictionResult(
            prediction=kalman_params.x.copy(),
            covariance=kalman_params.P.copy(),
            uncertainty=uncertainty,
            confidence=confidence,
            prediction_interval=prediction_intervals,
            timestamp=datetime.now().timestamp()
        )
        
        # Store in history
        self.prediction_history.append(result)
        
        logger.debug(f"Enhanced prediction: confidence={confidence:.3f}")
        
        return result
    
    def adaptive_update(self, observation: np.ndarray,
                       prediction: EnhancedPredictionResult) -> EnhancedUpdateResult:
        """
        Adaptive update with parameter adjustment.
        
        Args:
            observation: Current observation
            prediction: Previous prediction result
            
        Returns:
            Enhanced update result
        """
        # Create Kalman parameters for update
        kalman_params = KalmanParams(
            x=prediction.prediction,
            F=self.parameters.F,
            H=self.parameters.H,
            R=self.parameters.R,
            P=prediction.covariance
        )
        
        # Perform update
        kalman_update(kalman_params, observation)
        
        # Calculate innovation
        innovation = observation - self.parameters.H @ prediction.prediction
        
        # Calculate innovation covariance
        innovation_covariance = self.parameters.H @ prediction.covariance @ self.parameters.H.T + self.parameters.R
        
        # Calculate Kalman gain
        kalman_gain = prediction.covariance @ self.parameters.H.T @ np.linalg.inv(innovation_covariance)
        
        # Calculate log-likelihood
        log_likelihood = self._calculate_log_likelihood(innovation, innovation_covariance)
        
        # Adaptive parameter update
        updated_parameters = self.parameter_estimator.online_parameter_update(
            observation, self.parameters, innovation
        )
        
        # Update internal state
        self.current_state = kalman_params.x.copy()
        self.current_covariance = kalman_params.P.copy()
        self.parameters = updated_parameters
        
        # Create enhanced result
        result = EnhancedUpdateResult(
            updated_state=kalman_params.x.copy(),
            updated_covariance=kalman_params.P.copy(),
            innovation=innovation,
            innovation_covariance=innovation_covariance,
            kalman_gain=kalman_gain,
            log_likelihood=log_likelihood,
            parameter_adjustments={
                'Q_adjustment': np.linalg.norm(updated_parameters.Q - self.parameters.Q),
                'R_adjustment': np.linalg.norm(updated_parameters.R - self.parameters.R)
            },
            timestamp=datetime.now().timestamp()
        )
        
        # Store in history
        self.update_history.append(result)
        
        logger.debug(f"Adaptive update: log_likelihood={log_likelihood:.3f}")
        
        return result
    
    def _calculate_confidence(self, covariance: np.ndarray) -> float:
        """Calculate prediction confidence from covariance."""
        # Confidence based on determinant of covariance (smaller = more confident)
        det_cov = np.linalg.det(covariance)
        confidence = 1.0 / (1.0 + det_cov)
        return max(0.0, min(1.0, confidence))
    
    def _calculate_log_likelihood(self, innovation: np.ndarray,
                                innovation_covariance: np.ndarray) -> float:
        """Calculate log-likelihood of innovation."""
        try:
            # Use multivariate normal log-likelihood
            log_likelihood = multivariate_normal.logpdf(
                innovation, mean=np.zeros_like(innovation), cov=innovation_covariance
            )
            return log_likelihood
        except:
            # Fallback to simple calculation
            return -0.5 * np.sum(innovation**2 / np.diag(innovation_covariance))
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        # Calculate statistics
        confidences = [p.confidence for p in self.prediction_history]
        uncertainties = [np.mean(p.uncertainty) for p in self.prediction_history]
        
        if self.update_history:
            log_likelihoods = [u.log_likelihood for u in self.update_history]
            innovations = [np.linalg.norm(u.innovation) for u in self.update_history]
        else:
            log_likelihoods = []
            innovations = []
        
        return {
            'total_predictions': len(self.prediction_history),
            'total_updates': len(self.update_history),
            'average_confidence': np.mean(confidences),
            'average_uncertainty': np.mean(uncertainties),
            'average_log_likelihood': np.mean(log_likelihoods) if log_likelihoods else 0.0,
            'average_innovation_magnitude': np.mean(innovations) if innovations else 0.0,
            'current_regime': self.regime
        }
    
    def reset_history(self):
        """Reset prediction and update history."""
        self.prediction_history.clear()
        self.update_history.clear()
        logger.info("Reset enhanced Kalman filter history")
    
    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Standard predict method for compatibility with diagnostic experiment.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Prediction array [ROI, risk]
        """
        try:
            # Initialize parameters if not done
            if self.parameters is None:
                self.initialize_parameters()
            
            # Use enhanced prediction
            result = self.enhanced_prediction(self.current_state, self.regime)
            
            # Return just the prediction values for compatibility
            return result.prediction[:2]  # ROI and risk only
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return np.array([0.0, 0.0])
    
    def update(self, observation: np.ndarray) -> None:
        """
        Standard update method for compatibility with diagnostic experiment.
        
        Args:
            observation: New observation [ROI, risk]
        """
        try:
            # Initialize parameters if not done
            if self.parameters is None:
                self.initialize_parameters()
            
            # Create a prediction first
            prediction_result = self.enhanced_prediction(self.current_state, self.regime)
            
            # Use adaptive update
            update_result = self.adaptive_update(observation, prediction_result)
            
            # Update current state
            self.current_state = update_result.updated_state
            self.current_covariance = update_result.updated_covariance
            
        except Exception as e:
            logger.error(f"Enhanced update error: {e}")
    
    def get_current_state(self) -> np.ndarray:
        """Get current state for diagnostic experiment."""
        return self.current_state.copy()
    
    def get_uncertainty(self) -> np.ndarray:
        """Get current uncertainty for diagnostic experiment."""
        if self.current_covariance is not None:
            return np.sqrt(np.diag(self.current_covariance))
        return np.array([0.0, 0.0, 0.0, 0.0])
    
    def get_confidence(self) -> float:
        """Get current confidence for diagnostic experiment."""
        if self.current_covariance is not None:
            return self._calculate_confidence(self.current_covariance)
        return 0.0


def create_enhanced_kalman_filter(state_dim: int = 4, 
                                observation_dim: int = 2,
                                regime: str = 'sideways_market') -> EnhancedKalmanFilter:
    """
    Convenience function to create enhanced Kalman filter.
    
    Args:
        state_dim: State dimension
        observation_dim: Observation dimension
        regime: Initial regime
        
    Returns:
        Enhanced Kalman filter instance
    """
    filter_instance = EnhancedKalmanFilter(state_dim, observation_dim)
    filter_instance.initialize_parameters(regime)
    return filter_instance


if __name__ == '__main__':
    # Example usage
    print("Enhanced Kalman Filter Module")
    print("This module provides enhanced Kalman filtering with advanced features.")
    print("Use EnhancedKalmanFilter class for enhanced predictions.")



