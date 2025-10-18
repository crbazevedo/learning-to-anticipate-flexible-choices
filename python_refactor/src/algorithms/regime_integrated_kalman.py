"""
Regime-Integrated Kalman Filter

This module implements a Kalman filter with full regime integration,
combining the enhanced Kalman filter with regime detection from EPIC 1.
It provides regime-specific models, smooth transitions, and regime-aware predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

from algorithms.enhanced_kalman_filter import (
    EnhancedKalmanFilter, EnhancedPredictionResult, EnhancedUpdateResult,
    KalmanParameters, EnhancedStateSpaceModel
)
from algorithms.regime_detection_bnn import MarketRegimeDetectionBNN, RegimeDetectionResult

logger = logging.getLogger(__name__)


@dataclass
class RegimeAwareResult:
    """Result with full regime awareness."""
    
    prediction: np.ndarray
    covariance: np.ndarray
    uncertainty: np.ndarray
    confidence: float
    regime_info: RegimeDetectionResult
    regime_specific_prediction: np.ndarray
    regime_transition_probability: float
    multi_regime_predictions: Dict[str, np.ndarray]
    timestamp: float


class RegimeSpecificKalmanModel:
    """Regime-specific Kalman filter model."""
    
    def __init__(self, regime: str, state_dim: int = 4, observation_dim: int = 2):
        self.regime = regime
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        
        # Regime-specific parameters
        self.regime_params = self._create_regime_parameters(regime)
        
        # Model performance tracking
        self.performance_history = []
        self.prediction_accuracy = []
        
        logger.info(f"Initialized RegimeSpecificKalmanModel for {regime}")
    
    def _create_regime_parameters(self, regime: str) -> Dict[str, Any]:
        """Create regime-specific parameters."""
        
        if regime == 'bull_market':
            return {
                'mean_reversion': 0.05,  # Lower mean reversion in bull markets
                'volatility_clustering': 0.02,  # Lower volatility clustering
                'cross_correlation': 0.2,  # Lower cross-correlation
                'momentum_persistence': 0.9,  # Higher momentum persistence
                'process_noise_scale': 0.5,  # Lower process noise
                'measurement_noise_scale': 0.3,  # Lower measurement noise
                'adaptation_rate': 0.005,  # Slower adaptation
                'confidence_boost': 0.1  # Higher confidence
            }
        elif regime == 'bear_market':
            return {
                'mean_reversion': 0.2,  # Higher mean reversion in bear markets
                'volatility_clustering': 0.1,  # Higher volatility clustering
                'cross_correlation': 0.6,  # Higher cross-correlation
                'momentum_persistence': 0.6,  # Lower momentum persistence
                'process_noise_scale': 2.0,  # Higher process noise
                'measurement_noise_scale': 1.5,  # Higher measurement noise
                'adaptation_rate': 0.02,  # Faster adaptation
                'confidence_boost': -0.1  # Lower confidence
            }
        else:  # sideways_market
            return {
                'mean_reversion': 0.1,  # Moderate mean reversion
                'volatility_clustering': 0.05,  # Moderate volatility clustering
                'cross_correlation': 0.3,  # Moderate cross-correlation
                'momentum_persistence': 0.8,  # Moderate momentum persistence
                'process_noise_scale': 1.0,  # Standard process noise
                'measurement_noise_scale': 1.0,  # Standard measurement noise
                'adaptation_rate': 0.01,  # Standard adaptation
                'confidence_boost': 0.0  # No confidence boost
            }
    
    def create_regime_specific_parameters(self) -> KalmanParameters:
        """Create regime-specific Kalman parameters."""
        
        # Create enhanced state space model with regime parameters
        state_model = EnhancedStateSpaceModel(self.state_dim, self.observation_dim)
        state_model.dynamics_params.update(self.regime_params)
        
        # Create regime-specific matrices
        F = state_model.create_enhanced_transition_matrix()
        H = state_model.create_enhanced_measurement_matrix()
        Q = state_model.create_enhanced_process_noise(regime=self.regime)
        R = state_model.create_enhanced_measurement_noise(regime=self.regime)
        P0 = np.eye(self.state_dim) * 0.1
        
        # Scale matrices by regime parameters
        Q *= self.regime_params['process_noise_scale']
        R *= self.regime_params['measurement_noise_scale']
        
        return KalmanParameters(
            F=F, H=H, Q=Q, R=R, P0=P0,
            adaptation_rate=self.regime_params['adaptation_rate'],
            forgetting_factor=0.95
        )
    
    def predict(self, current_state: np.ndarray, 
               regime_info: RegimeDetectionResult) -> EnhancedPredictionResult:
        """Make prediction with regime-specific model."""
        
        # Create regime-specific parameters
        parameters = self.create_regime_specific_parameters()
        
        # Create enhanced Kalman filter with regime parameters
        kalman_filter = EnhancedKalmanFilter(self.state_dim, self.observation_dim)
        kalman_filter.parameters = parameters
        kalman_filter.current_state = current_state
        kalman_filter.current_covariance = parameters.P0
        
        # Make prediction
        prediction = kalman_filter.enhanced_prediction(current_state, self.regime)
        
        # Adjust confidence based on regime
        prediction.confidence += self.regime_params['confidence_boost']
        prediction.confidence = max(0.0, min(1.0, prediction.confidence))
        
        # Store performance
        self.performance_history.append(prediction)
        
        return prediction
    
    def update_performance(self, observation: np.ndarray, 
                          prediction: EnhancedPredictionResult) -> None:
        """Update model performance based on observation."""
        
        # Calculate prediction error
        prediction_error = np.linalg.norm(observation - prediction.prediction[:2])
        self.prediction_accuracy.append(prediction_error)
        
        # Keep only recent history
        if len(self.prediction_accuracy) > 100:
            self.prediction_accuracy = self.prediction_accuracy[-100:]
    
    def get_regime_performance(self) -> Dict[str, Any]:
        """Get regime-specific performance statistics."""
        
        if not self.prediction_accuracy:
            return {'error': 'No performance data available'}
        
        return {
            'regime': self.regime,
            'average_prediction_error': np.mean(self.prediction_accuracy),
            'prediction_error_std': np.std(self.prediction_accuracy),
            'total_predictions': len(self.prediction_accuracy),
            'recent_performance': np.mean(self.prediction_accuracy[-10:]) if len(self.prediction_accuracy) >= 10 else np.mean(self.prediction_accuracy)
        }


class RegimeIntegratedKalmanFilter:
    """Kalman filter with full regime integration."""
    
    def __init__(self, regime_detector: MarketRegimeDetectionBNN):
        self.regime_detector = regime_detector
        
        # Regime-specific models
        self.regime_models = {
            'bull_market': RegimeSpecificKalmanModel('bull_market'),
            'bear_market': RegimeSpecificKalmanModel('bear_market'),
            'sideways_market': RegimeSpecificKalmanModel('sideways_market')
        }
        
        # Current state
        self.current_state = np.zeros(4)
        self.current_covariance = np.eye(4) * 0.1
        
        # Regime tracking
        self.current_regime = 'sideways_market'
        self.regime_history = []
        self.regime_transition_history = []
        
        # Performance tracking
        self.prediction_history = []
        self.update_history = []
        
        logger.info("Initialized RegimeIntegratedKalmanFilter")
    
    def regime_aware_prediction(self, current_state: np.ndarray,
                              market_features: np.ndarray) -> RegimeAwareResult:
        """
        Prediction with full regime awareness.
        
        Args:
            current_state: Current state vector
            market_features: Market features for regime detection
            
        Returns:
            Regime-aware prediction result
        """
        # Detect current regime
        regime_info = self.regime_detector.detect_regime(market_features)
        detected_regime = regime_info.predicted_regime
        
        # Handle regime transitions
        if detected_regime != self.current_regime:
            self._handle_regime_transition(detected_regime, regime_info)
        
        # Get regime-specific prediction
        regime_model = self.regime_models[self.current_regime]
        regime_prediction = regime_model.predict(current_state, regime_info)
        
        # Calculate regime transition probability
        transition_probability = self._calculate_transition_probability(regime_info)
        
        # Get multi-regime predictions
        multi_regime_predictions = self._get_multi_regime_predictions(current_state, regime_info)
        
        # Create regime-aware result
        result = RegimeAwareResult(
            prediction=regime_prediction.prediction,
            covariance=regime_prediction.covariance,
            uncertainty=regime_prediction.uncertainty,
            confidence=regime_prediction.confidence,
            regime_info=regime_info,
            regime_specific_prediction=regime_prediction.prediction,
            regime_transition_probability=transition_probability,
            multi_regime_predictions=multi_regime_predictions,
            timestamp=datetime.now().timestamp()
        )
        
        # Store in history
        self.prediction_history.append(result)
        self.regime_history.append(regime_info)
        
        # Update current state
        self.current_state = current_state.copy()
        
        logger.debug(f"Regime-aware prediction: {self.current_regime} (confidence: {result.confidence:.3f})")
        
        return result
    
    def regime_aware_update(self, observation: np.ndarray,
                          prediction: RegimeAwareResult) -> EnhancedUpdateResult:
        """
        Update with regime awareness.
        
        Args:
            observation: Current observation
            prediction: Previous regime-aware prediction
            
        Returns:
            Enhanced update result
        """
        # Get current regime model
        regime_model = self.regime_models[self.current_regime]
        
        # Create enhanced Kalman filter for update
        kalman_filter = EnhancedKalmanFilter()
        kalman_filter.parameters = regime_model.create_regime_specific_parameters()
        kalman_filter.current_state = prediction.prediction
        kalman_filter.current_covariance = prediction.covariance
        
        # Perform update
        update_result = kalman_filter.adaptive_update(observation, 
                                                    EnhancedPredictionResult(
                                                        prediction=prediction.prediction,
                                                        covariance=prediction.covariance,
                                                        uncertainty=prediction.uncertainty,
                                                        confidence=prediction.confidence
                                                    ))
        
        # Update regime model performance
        regime_model.update_performance(observation, 
                                      EnhancedPredictionResult(
                                          prediction=prediction.prediction,
                                          covariance=prediction.covariance,
                                          uncertainty=prediction.uncertainty,
                                          confidence=prediction.confidence
                                      ))
        
        # Update current state
        self.current_state = update_result.updated_state.copy()
        self.current_covariance = update_result.updated_covariance.copy()
        
        # Store in history
        self.update_history.append(update_result)
        
        logger.debug(f"Regime-aware update: {self.current_regime} (log_likelihood: {update_result.log_likelihood:.3f})")
        
        return update_result
    
    def _handle_regime_transition(self, new_regime: str, regime_info: RegimeDetectionResult) -> None:
        """Handle transition to new regime."""
        
        old_regime = self.current_regime
        self.current_regime = new_regime
        
        # Record transition
        transition = {
            'from_regime': old_regime,
            'to_regime': new_regime,
            'confidence': regime_info.confidence,
            'timestamp': datetime.now().timestamp()
        }
        self.regime_transition_history.append(transition)
        
        logger.info(f"Regime transition: {old_regime} -> {new_regime} (confidence: {regime_info.confidence:.3f})")
    
    def _calculate_transition_probability(self, regime_info: RegimeDetectionResult) -> float:
        """Calculate probability of regime transition."""
        
        if not self.regime_history:
            return 0.0
        
        # Calculate transition probability based on recent regime stability
        recent_regimes = [r.predicted_regime for r in self.regime_history[-10:]]
        regime_stability = len(set(recent_regimes)) / len(recent_regimes)
        
        # Higher stability = lower transition probability
        transition_probability = 1.0 - regime_stability
        
        return transition_probability
    
    def _get_multi_regime_predictions(self, current_state: np.ndarray,
                                    regime_info: RegimeDetectionResult) -> Dict[str, np.ndarray]:
        """Get predictions from all regime models."""
        
        multi_regime_predictions = {}
        
        for regime_name, regime_model in self.regime_models.items():
            try:
                prediction = regime_model.predict(current_state, regime_info)
                multi_regime_predictions[regime_name] = prediction.prediction
            except Exception as e:
                logger.warning(f"Failed to get prediction for regime {regime_name}: {e}")
                multi_regime_predictions[regime_name] = current_state.copy()
        
        return multi_regime_predictions
    
    def predict_multiple_horizons(self, current_state: np.ndarray,
                                market_features: np.ndarray,
                                horizons: List[int]) -> Dict[int, RegimeAwareResult]:
        """
        Predict for multiple horizons with regime awareness.
        
        Args:
            current_state: Current state vector
            market_features: Market features for regime detection
            horizons: List of prediction horizons
            
        Returns:
            Dictionary mapping horizon to prediction result
        """
        results = {}
        current_state_copy = current_state.copy()
        
        for horizon in horizons:
            # Get regime-aware prediction
            result = self.regime_aware_prediction(current_state_copy, market_features)
            results[horizon] = result
            
            # Update state for next horizon
            current_state_copy = result.prediction.copy()
        
        return results
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics."""
        
        if not self.regime_history:
            return {'error': 'No regime history available'}
        
        # Regime distribution
        regime_counts = {}
        for regime in ['bull_market', 'bear_market', 'sideways_market']:
            regime_counts[regime] = sum(1 for r in self.regime_history 
                                      if r.predicted_regime == regime)
        
        # Regime transitions
        transitions = []
        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i-1].predicted_regime
            curr_regime = self.regime_history[i].predicted_regime
            if prev_regime != curr_regime:
                transitions.append(f"{prev_regime} -> {curr_regime}")
        
        # Regime performance
        regime_performance = {}
        for regime_name, regime_model in self.regime_models.items():
            regime_performance[regime_name] = regime_model.get_regime_performance()
        
        # Overall statistics
        confidences = [r.confidence for r in self.regime_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'total_updates': len(self.update_history),
            'regime_distribution': regime_counts,
            'regime_transitions': transitions,
            'transition_count': len(transitions),
            'average_confidence': np.mean(confidences),
            'current_regime': self.current_regime,
            'regime_stability': 1.0 - (len(transitions) / len(self.regime_history)) if self.regime_history else 0.0,
            'regime_performance': regime_performance
        }
    
    def reset_history(self):
        """Reset all history."""
        self.prediction_history.clear()
        self.update_history.clear()
        self.regime_history.clear()
        self.regime_transition_history.clear()
        
        # Reset regime models
        for regime_model in self.regime_models.values():
            regime_model.performance_history.clear()
            regime_model.prediction_accuracy.clear()
        
        logger.info("Reset regime-integrated Kalman filter history")
    
    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Standard predict method for compatibility with diagnostic experiment.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Prediction array [ROI, risk]
        """
        try:
            # Use current state and create dummy market features
            market_features = np.array([0.0, 0.0, 0.0, 0.0])  # Dummy features
            
            # Use regime-aware prediction
            result = self.regime_aware_prediction(self.current_state, market_features)
            
            # Return just the prediction values for compatibility
            return result.prediction[:2]  # ROI and risk only
            
        except Exception as e:
            logger.error(f"Regime-integrated prediction error: {e}")
            return np.array([0.0, 0.0])
    
    def update(self, observation: np.ndarray) -> None:
        """
        Standard update method for compatibility with diagnostic experiment.
        
        Args:
            observation: New observation [ROI, risk]
        """
        try:
            # Create dummy market features from observation
            market_features = np.array([observation[0], observation[1], 0.0, 0.0])
            
            # First get a prediction
            prediction_result = self.regime_aware_prediction(self.current_state, market_features)
            
            # Then use regime-aware update
            result = self.regime_aware_update(observation, prediction_result)
            
            # Update current state
            self.current_state = result.updated_state
            self.current_covariance = result.updated_covariance
            
        except Exception as e:
            logger.error(f"Regime-integrated update error: {e}")
    
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
        if self.regime_history:
            return self.regime_history[-1].confidence
        return 0.0


def create_regime_integrated_kalman(regime_detector: MarketRegimeDetectionBNN) -> RegimeIntegratedKalmanFilter:
    """
    Convenience function to create regime-integrated Kalman filter.
    
    Args:
        regime_detector: Market regime detection BNN
        
    Returns:
        Regime-integrated Kalman filter instance
    """
    return RegimeIntegratedKalmanFilter(regime_detector)


if __name__ == '__main__':
    # Example usage
    print("Regime-Integrated Kalman Filter Module")
    print("This module provides regime-aware Kalman filtering.")
    print("Use RegimeIntegratedKalmanFilter class for regime-aware predictions.")



