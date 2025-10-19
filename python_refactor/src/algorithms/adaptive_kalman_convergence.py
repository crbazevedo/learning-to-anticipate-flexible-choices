#!/usr/bin/env python3
"""
Adaptive Kalman Convergence - Enhanced Convergence Mechanisms

This module implements sophisticated Kalman filter convergence mechanisms with:
1. Adaptive convergence criteria
2. Convergence monitoring
3. Initial parameter estimation
4. Convergence acceleration techniques
5. Dynamic convergence adjustment

Key improvements:
1. Adaptive convergence criteria based on system state
2. Real-time convergence monitoring and alerts
3. Improved initial parameter estimation
4. Convergence acceleration techniques
5. Dynamic convergence adjustment based on performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveKalmanConvergence:
    """
    Adaptive Kalman convergence system with enhanced convergence mechanisms.
    
    This class implements sophisticated Kalman filter convergence mechanisms:
    - Adaptive convergence criteria
    - Convergence monitoring
    - Initial parameter estimation
    - Convergence acceleration techniques
    - Dynamic convergence adjustment
    """
    
    def __init__(self, state_dim: int = 4, observation_dim: int = 2, 
                 convergence_threshold: float = 0.95):
        """
        Initialize adaptive Kalman convergence system.
        
        Args:
            state_dim: Dimension of the state vector
            observation_dim: Dimension of the observation vector
            convergence_threshold: Minimum convergence threshold
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.convergence_threshold = convergence_threshold
        
        # Convergence tracking
        self.convergence_history = deque(maxlen=50)
        self.innovation_history = deque(maxlen=50)
        self.covariance_history = deque(maxlen=50)
        self.state_history = deque(maxlen=50)
        
        # Adaptive convergence criteria
        self.adaptive_criteria = {
            'innovation_threshold': 0.01,
            'covariance_threshold': 0.001,
            'state_change_threshold': 0.005,
            'convergence_window': 10
        }
        
        # Convergence acceleration parameters
        self.acceleration_params = {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'adaptive_step_size': True,
            'convergence_boost': 1.2
        }
        
        # Initial parameter estimation
        self.initial_params = {
            'process_noise': 0.01,
            'measurement_noise': 0.005,
            'initial_covariance': 0.1,
            'initial_state': None
        }
        
        # Convergence monitoring
        self.convergence_metrics = {
            'current_convergence': 0.0,
            'convergence_rate': 0.0,
            'convergence_stability': 0.0,
            'convergence_quality': 0.0
        }
        
        # Dynamic adjustment
        self.adjustment_enabled = True
        self.monitoring_enabled = True
        self.acceleration_enabled = True
        
        logger.info(f"Initialized AdaptiveKalmanConvergence with state_dim={state_dim}, observation_dim={observation_dim}")
    
    def check_convergence(self, innovation: np.ndarray, covariance: np.ndarray, 
                         state: np.ndarray, previous_state: np.ndarray = None) -> Dict[str, Any]:
        """
        Check Kalman filter convergence using adaptive criteria.
        
        Args:
            innovation: Innovation vector
            covariance: Covariance matrix
            state: Current state vector
            previous_state: Previous state vector
            
        Returns:
            Dictionary with convergence results
        """
        try:
            # Calculate convergence metrics
            innovation_norm = np.linalg.norm(innovation)
            covariance_trace = np.trace(covariance)
            state_change = np.linalg.norm(state - previous_state) if previous_state is not None else 0.0
            
            # Store history
            self.innovation_history.append(innovation_norm)
            self.covariance_history.append(covariance_trace)
            self.state_history.append(state.copy())
            
            # Calculate adaptive convergence score
            convergence_score = self._calculate_adaptive_convergence_score(
                innovation_norm, covariance_trace, state_change
            )
            
            # Update convergence history
            self.convergence_history.append(convergence_score)
            
            # Check convergence status
            is_converged = self._check_convergence_status(convergence_score)
            
            # Calculate convergence metrics
            convergence_metrics = self._calculate_convergence_metrics()
            
            # Apply convergence acceleration if needed
            if self.acceleration_enabled and not is_converged:
                acceleration_factor = self._calculate_acceleration_factor(convergence_score)
            else:
                acceleration_factor = 1.0
            
            return {
                'is_converged': is_converged,
                'convergence_score': convergence_score,
                'innovation_norm': innovation_norm,
                'covariance_trace': covariance_trace,
                'state_change': state_change,
                'convergence_metrics': convergence_metrics,
                'acceleration_factor': acceleration_factor,
                'adaptive_criteria': self.adaptive_criteria
            }
            
        except Exception as e:
            logger.error(f"Error checking convergence: {e}")
            return {
                'is_converged': False,
                'convergence_score': 0.0,
                'innovation_norm': 0.0,
                'covariance_trace': 0.0,
                'state_change': 0.0,
                'convergence_metrics': {},
                'acceleration_factor': 1.0,
                'adaptive_criteria': self.adaptive_criteria
            }
    
    def _calculate_adaptive_convergence_score(self, innovation_norm: float, 
                                            covariance_trace: float, 
                                            state_change: float) -> float:
        """Calculate adaptive convergence score."""
        
        # Normalize metrics
        innovation_score = max(0, 1.0 - innovation_norm / self.adaptive_criteria['innovation_threshold'])
        covariance_score = max(0, 1.0 - covariance_trace / self.adaptive_criteria['covariance_threshold'])
        state_score = max(0, 1.0 - state_change / self.adaptive_criteria['state_change_threshold'])
        
        # Weighted combination
        weights = [0.4, 0.3, 0.3]  # Innovation, covariance, state change
        scores = [innovation_score, covariance_score, state_score]
        
        convergence_score = np.average(scores, weights=weights)
        
        # Apply historical consistency
        if len(self.convergence_history) >= 3:
            recent_scores = list(self.convergence_history)[-3:]
            consistency = 1.0 - np.std(recent_scores)
            convergence_score = 0.7 * convergence_score + 0.3 * consistency
        
        return min(1.0, max(0.0, convergence_score))
    
    def _check_convergence_status(self, convergence_score: float) -> bool:
        """Check if the system has converged."""
        
        # Basic threshold check
        if convergence_score >= self.convergence_threshold:
            return True
        
        # Adaptive threshold based on history
        if len(self.convergence_history) >= self.adaptive_criteria['convergence_window']:
            recent_scores = list(self.convergence_history)[-self.adaptive_criteria['convergence_window']:]
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            # Adaptive threshold
            adaptive_threshold = self.convergence_threshold - 0.1 * std_score
            
            if mean_score >= adaptive_threshold and std_score < 0.05:
                return True
        
        return False
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate detailed convergence metrics."""
        
        if len(self.convergence_history) < 2:
            return {
                'current_convergence': 0.0,
                'convergence_rate': 0.0,
                'convergence_stability': 0.0,
                'convergence_quality': 0.0
            }
        
        # Current convergence
        current_convergence = self.convergence_history[-1]
        
        # Convergence rate (improvement over time)
        if len(self.convergence_history) >= 5:
            recent_scores = list(self.convergence_history)[-5:]
            convergence_rate = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        else:
            convergence_rate = 0.0
        
        # Convergence stability (consistency)
        if len(self.convergence_history) >= 3:
            recent_scores = list(self.convergence_history)[-3:]
            convergence_stability = 1.0 - np.std(recent_scores)
        else:
            convergence_stability = 0.0
        
        # Convergence quality (overall assessment)
        convergence_quality = 0.4 * current_convergence + 0.3 * convergence_rate + 0.3 * convergence_stability
        
        return {
            'current_convergence': current_convergence,
            'convergence_rate': convergence_rate,
            'convergence_stability': convergence_stability,
            'convergence_quality': convergence_quality
        }
    
    def _calculate_acceleration_factor(self, convergence_score: float) -> float:
        """Calculate convergence acceleration factor."""
        
        if convergence_score < 0.5:
            # Low convergence - apply strong acceleration
            return self.acceleration_params['convergence_boost'] * 1.5
        elif convergence_score < 0.8:
            # Medium convergence - apply moderate acceleration
            return self.acceleration_params['convergence_boost']
        else:
            # High convergence - apply light acceleration
            return 1.0 + (self.acceleration_params['convergence_boost'] - 1.0) * 0.5
    
    def estimate_initial_parameters(self, historical_data: np.ndarray) -> Dict[str, Any]:
        """
        Estimate initial parameters for Kalman filter.
        
        Args:
            historical_data: Historical data for parameter estimation
            
        Returns:
            Dictionary with estimated initial parameters
        """
        try:
            if len(historical_data) < 10:
                return self.initial_params
            
            # Estimate process noise from data volatility
            if historical_data.shape[1] >= 2:
                roi_data = historical_data[:, 0]
                risk_data = historical_data[:, 1]
                
                # Calculate volatility
                roi_volatility = np.std(roi_data)
                risk_volatility = np.std(risk_data)
                
                # Estimate process noise
                process_noise = min(0.1, max(0.001, (roi_volatility + risk_volatility) / 2))
                
                # Estimate measurement noise
                measurement_noise = min(0.01, max(0.001, process_noise * 0.5))
                
                # Estimate initial covariance
                initial_covariance = min(0.5, max(0.01, process_noise * 10))
                
                # Estimate initial state
                initial_state = np.array([
                    np.mean(roi_data[-5:]),  # Recent ROI average
                    np.mean(risk_data[-5:]),  # Recent risk average
                    0.0,  # ROI velocity
                    0.0   # Risk velocity
                ])
                
                return {
                    'process_noise': process_noise,
                    'measurement_noise': measurement_noise,
                    'initial_covariance': initial_covariance,
                    'initial_state': initial_state
                }
            else:
                return self.initial_params
                
        except Exception as e:
            logger.error(f"Error estimating initial parameters: {e}")
            return self.initial_params
    
    def update_adaptive_criteria(self, performance_metric: float):
        """Update adaptive convergence criteria based on performance."""
        
        if not self.adjustment_enabled:
            return
        
        # Adjust criteria based on performance
        if performance_metric > 0.8:
            # High performance - tighten criteria
            self.adaptive_criteria['innovation_threshold'] *= 0.95
            self.adaptive_criteria['covariance_threshold'] *= 0.95
            self.adaptive_criteria['state_change_threshold'] *= 0.95
        elif performance_metric < 0.6:
            # Low performance - loosen criteria
            self.adaptive_criteria['innovation_threshold'] *= 1.05
            self.adaptive_criteria['covariance_threshold'] *= 1.05
            self.adaptive_criteria['state_change_threshold'] *= 1.05
        
        # Ensure bounds
        self.adaptive_criteria['innovation_threshold'] = max(0.001, min(0.1, self.adaptive_criteria['innovation_threshold']))
        self.adaptive_criteria['covariance_threshold'] = max(0.0001, min(0.01, self.adaptive_criteria['covariance_threshold']))
        self.adaptive_criteria['state_change_threshold'] = max(0.001, min(0.05, self.adaptive_criteria['state_change_threshold']))
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """Get current convergence status and metrics."""
        
        return {
            'convergence_type': 'AdaptiveKalmanConvergence',
            'convergence_threshold': self.convergence_threshold,
            'adaptive_criteria': self.adaptive_criteria,
            'acceleration_params': self.acceleration_params,
            'convergence_history_length': len(self.convergence_history),
            'innovation_history_length': len(self.innovation_history),
            'covariance_history_length': len(self.covariance_history),
            'state_history_length': len(self.state_history),
            'adjustment_enabled': self.adjustment_enabled,
            'monitoring_enabled': self.monitoring_enabled,
            'acceleration_enabled': self.acceleration_enabled
        }
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get detailed convergence metrics."""
        
        if len(self.convergence_history) == 0:
            return {
                'current_convergence': 0.0,
                'convergence_rate': 0.0,
                'convergence_stability': 0.0,
                'convergence_quality': 0.0,
                'is_converged': False
            }
        
        # Calculate current metrics
        current_convergence = self.convergence_history[-1]
        convergence_metrics = self._calculate_convergence_metrics()
        is_converged = self._check_convergence_status(current_convergence)
        
        return {
            'current_convergence': current_convergence,
            'convergence_rate': convergence_metrics['convergence_rate'],
            'convergence_stability': convergence_metrics['convergence_stability'],
            'convergence_quality': convergence_metrics['convergence_quality'],
            'is_converged': is_converged,
            'convergence_history': list(self.convergence_history),
            'innovation_history': list(self.innovation_history),
            'covariance_history': list(self.covariance_history)
        }
