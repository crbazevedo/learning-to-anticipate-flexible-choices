#!/usr/bin/env python3
"""
Adaptive Parameter Stability - Enhanced Parameter Management

This module implements sophisticated parameter stability mechanisms with:
1. Adaptive parameter bounds
2. Parameter smoothing mechanisms
3. Parameter validation
4. Parameter history analysis
5. Dynamic parameter adjustment

Key improvements:
1. Adaptive parameter bounds to prevent extreme values
2. Parameter smoothing to reduce oscillations
3. Parameter validation to ensure consistency
4. Parameter history analysis for trend detection
5. Dynamic parameter adjustment based on performance
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

class AdaptiveParameterStability:
    """
    Adaptive parameter stability system with enhanced parameter management.
    
    This class implements sophisticated parameter stability mechanisms:
    - Adaptive parameter bounds
    - Parameter smoothing mechanisms
    - Parameter validation
    - Parameter history analysis
    - Dynamic parameter adjustment
    """
    
    def __init__(self, window_size: int = 20, stability_threshold: float = 0.85):
        """
        Initialize adaptive parameter stability system.
        
        Args:
            window_size: Size of the sliding window for parameter analysis
            stability_threshold: Minimum stability threshold for parameters
        """
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        
        # Parameter history tracking
        self.parameter_history = {
            'learning_rates': deque(maxlen=window_size),
            'confidence_levels': deque(maxlen=window_size),
            'adaptation_rates': deque(maxlen=window_size),
            'forgetting_factors': deque(maxlen=window_size)
        }
        
        # Parameter bounds (adaptive)
        self.parameter_bounds = {
            'learning_rates': {'min': 0.001, 'max': 0.1, 'default': 0.01},
            'confidence_levels': {'min': 0.5, 'max': 0.99, 'default': 0.95},
            'adaptation_rates': {'min': 0.001, 'max': 0.05, 'default': 0.01},
            'forgetting_factors': {'min': 0.8, 'max': 0.99, 'default': 0.95}
        }
        
        # Parameter smoothing weights
        self.smoothing_weights = {
            'learning_rates': 0.3,
            'confidence_levels': 0.2,
            'adaptation_rates': 0.3,
            'forgetting_factors': 0.2
        }
        
        # Parameter validation rules
        self.validation_rules = {
            'learning_rates': self._validate_learning_rate,
            'confidence_levels': self._validate_confidence_level,
            'adaptation_rates': self._validate_adaptation_rate,
            'forgetting_factors': self._validate_forgetting_factor
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=window_size)
        self.stability_scores = deque(maxlen=window_size)
        
        # Adaptive mechanisms
        self.adaptation_enabled = True
        self.smoothing_enabled = True
        self.validation_enabled = True
        
        logger.info(f"Initialized AdaptiveParameterStability with window_size={window_size}")
    
    def update_parameters(self, new_parameters: Dict[str, float], 
                         performance_metric: float = None) -> Dict[str, float]:
        """
        Update parameters with stability mechanisms.
        
        Args:
            new_parameters: New parameter values
            performance_metric: Current performance metric
            
        Returns:
            Stabilized parameter values
        """
        try:
            # Track performance
            if performance_metric is not None:
                self.performance_history.append(performance_metric)
            
            # Apply parameter stability mechanisms
            stabilized_parameters = {}
            
            for param_name, param_value in new_parameters.items():
                if param_name in self.parameter_bounds:
                    # Apply adaptive bounds
                    bounded_value = self._apply_adaptive_bounds(param_name, param_value)
                    
                    # Apply smoothing
                    if self.smoothing_enabled:
                        smoothed_value = self._apply_parameter_smoothing(param_name, bounded_value)
                    else:
                        smoothed_value = bounded_value
                    
                    # Apply validation
                    if self.validation_enabled:
                        validated_value = self._validate_parameter(param_name, smoothed_value)
                    else:
                        validated_value = smoothed_value
                    
                    stabilized_parameters[param_name] = validated_value
                    
                    # Update parameter history
                    self.parameter_history[param_name].append(validated_value)
                else:
                    # Unknown parameter - use as is
                    stabilized_parameters[param_name] = param_value
            
            # Calculate stability score
            stability_score = self._calculate_stability_score()
            self.stability_scores.append(stability_score)
            
            # Apply adaptive adjustments if enabled
            if self.adaptation_enabled:
                stabilized_parameters = self._apply_adaptive_adjustments(stabilized_parameters)
            
            return stabilized_parameters
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return new_parameters
    
    def _apply_adaptive_bounds(self, param_name: str, param_value: float) -> float:
        """Apply adaptive bounds to parameter value."""
        bounds = self.parameter_bounds[param_name]
        
        # Get current bounds
        min_val = bounds['min']
        max_val = bounds['max']
        
        # Apply adaptive bounds based on history
        if len(self.parameter_history[param_name]) > 5:
            recent_values = list(self.parameter_history[param_name])[-5:]
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Adjust bounds based on recent behavior
            if std_value < 0.01:  # Low volatility - tighten bounds
                min_val = max(min_val, mean_value - 0.01)
                max_val = min(max_val, mean_value + 0.01)
            elif std_value > 0.05:  # High volatility - loosen bounds
                min_val = max(min_val, mean_value - 0.05)
                max_val = min(max_val, mean_value + 0.05)
        
        # Apply bounds
        return max(min_val, min(max_val, param_value))
    
    def _apply_parameter_smoothing(self, param_name: str, param_value: float) -> float:
        """Apply parameter smoothing to reduce oscillations."""
        if len(self.parameter_history[param_name]) == 0:
            return param_value
        
        # Get smoothing weight
        smoothing_weight = self.smoothing_weights[param_name]
        
        # Get previous value
        previous_value = self.parameter_history[param_name][-1]
        
        # Apply exponential smoothing
        smoothed_value = (smoothing_weight * param_value + 
                         (1 - smoothing_weight) * previous_value)
        
        return smoothed_value
    
    def _validate_parameter(self, param_name: str, param_value: float) -> float:
        """Validate parameter value using validation rules."""
        if param_name in self.validation_rules:
            return self.validation_rules[param_name](param_value)
        else:
            return param_value
    
    def _validate_learning_rate(self, value: float) -> float:
        """Validate learning rate parameter."""
        if value <= 0:
            return self.parameter_bounds['learning_rates']['default']
        if value > 0.1:
            return 0.1
        if value < 0.001:
            return 0.001
        return value
    
    def _validate_confidence_level(self, value: float) -> float:
        """Validate confidence level parameter."""
        if value < 0.5:
            return 0.5
        if value > 0.99:
            return 0.99
        return value
    
    def _validate_adaptation_rate(self, value: float) -> float:
        """Validate adaptation rate parameter."""
        if value <= 0:
            return self.parameter_bounds['adaptation_rates']['default']
        if value > 0.05:
            return 0.05
        if value < 0.001:
            return 0.001
        return value
    
    def _validate_forgetting_factor(self, value: float) -> float:
        """Validate forgetting factor parameter."""
        if value < 0.8:
            return 0.8
        if value > 0.99:
            return 0.99
        return value
    
    def _calculate_stability_score(self) -> float:
        """Calculate overall parameter stability score."""
        if len(self.parameter_history['learning_rates']) < 3:
            return 0.5
        
        stability_scores = []
        
        for param_name in self.parameter_history:
            if len(self.parameter_history[param_name]) >= 3:
                recent_values = list(self.parameter_history[param_name])[-3:]
                volatility = np.std(recent_values)
                mean_value = np.mean(recent_values)
                
                # Calculate stability score (lower volatility = higher stability)
                if mean_value > 0:
                    stability_score = 1.0 - min(volatility / mean_value, 1.0)
                else:
                    stability_score = 1.0 - min(volatility, 1.0)
                
                stability_scores.append(stability_score)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _apply_adaptive_adjustments(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Apply adaptive adjustments based on performance and stability."""
        if len(self.performance_history) < 3 or len(self.stability_scores) < 3:
            return parameters
        
        # Get recent performance and stability
        recent_performance = list(self.performance_history)[-3:]
        recent_stability = list(self.stability_scores)[-3:]
        
        # Calculate performance trend
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        # Calculate stability trend
        stability_trend = np.polyfit(range(len(recent_stability)), recent_stability, 1)[0]
        
        # Apply adjustments based on trends
        adjusted_parameters = parameters.copy()
        
        if performance_trend < 0 and stability_trend < 0:
            # Both performance and stability declining - increase learning rates
            if 'learning_rates' in adjusted_parameters:
                adjusted_parameters['learning_rates'] *= 1.1
            if 'adaptation_rates' in adjusted_parameters:
                adjusted_parameters['adaptation_rates'] *= 1.1
        
        elif performance_trend > 0 and stability_trend > 0:
            # Both performance and stability improving - maintain current rates
            pass
        
        elif performance_trend > 0 and stability_trend < 0:
            # Performance improving but stability declining - reduce learning rates
            if 'learning_rates' in adjusted_parameters:
                adjusted_parameters['learning_rates'] *= 0.9
            if 'adaptation_rates' in adjusted_parameters:
                adjusted_parameters['adaptation_rates'] *= 0.9
        
        elif performance_trend < 0 and stability_trend > 0:
            # Performance declining but stability improving - increase learning rates
            if 'learning_rates' in adjusted_parameters:
                adjusted_parameters['learning_rates'] *= 1.05
            if 'adaptation_rates' in adjusted_parameters:
                adjusted_parameters['adaptation_rates'] *= 1.05
        
        return adjusted_parameters
    
    def get_parameter_stability_metrics(self) -> Dict[str, Any]:
        """Get current parameter stability metrics."""
        metrics = {}
        
        for param_name in self.parameter_history:
            if len(self.parameter_history[param_name]) > 0:
                recent_values = list(self.parameter_history[param_name])
                
                metrics[param_name] = {
                    'current_value': recent_values[-1],
                    'mean_value': np.mean(recent_values),
                    'std_value': np.std(recent_values),
                    'min_value': np.min(recent_values),
                    'max_value': np.max(recent_values),
                    'volatility': np.std(recent_values) / (np.mean(recent_values) + 1e-8),
                    'stability_score': 1.0 - min(np.std(recent_values) / (np.mean(recent_values) + 1e-8), 1.0)
                }
        
        # Overall stability metrics
        if len(self.stability_scores) > 0:
            metrics['overall'] = {
                'current_stability': self.stability_scores[-1],
                'mean_stability': np.mean(self.stability_scores),
                'stability_trend': np.polyfit(range(len(self.stability_scores)), list(self.stability_scores), 1)[0] if len(self.stability_scores) > 1 else 0.0
            }
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'parameter_stability_type': 'AdaptiveParameterStability',
            'window_size': self.window_size,
            'stability_threshold': self.stability_threshold,
            'adaptation_enabled': self.adaptation_enabled,
            'smoothing_enabled': self.smoothing_enabled,
            'validation_enabled': self.validation_enabled,
            'parameter_history_length': {name: len(history) for name, history in self.parameter_history.items()},
            'performance_history_length': len(self.performance_history),
            'stability_scores_length': len(self.stability_scores)
        }
