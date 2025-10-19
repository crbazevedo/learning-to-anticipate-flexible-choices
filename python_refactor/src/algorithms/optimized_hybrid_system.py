#!/usr/bin/env python3
"""
Optimized Hybrid Online Learning System

This module implements the optimized hybrid online learning system with
tuned parameters from the parameter tuning experiment.

Optimal Parameters:
- Adaptive Mirror Descent: lr=0.01, beta=0.99, epsilon=1e-06
- UCB Confidence: confidence=0.9, alpha=1.0
- Learning Rates: regime=0.005, parameter=0.02, kalman=0.005
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Import our online learning components
from .contextual_bandits_regime_detection import ContextualBanditUCB, RegimeDetectionSystem
from .adaptive_mirror_descent import AdaptiveMirrorDescent, ParameterAdaptationSystem
from .online_newton_step import OnlineNewtonStep, KalmanOptimizationSystem

logger = logging.getLogger(__name__)

@dataclass
class OptimizedSystemResult:
    """Result of optimized system update"""
    regime: str
    regime_confidence: float
    adapted_parameters: Dict[str, float]
    kalman_matrices: Dict[str, np.ndarray]
    regret_bounds: Dict[str, float]
    optimization_metrics: Dict[str, float]
    timestamp: float

class OptimizedHybridOnlineLearningSystem:
    """
    Optimized hybrid online learning system with tuned parameters.
    
    Uses optimal parameters from parameter tuning experiment:
    - Adaptive Mirror Descent: lr=0.01, beta=0.99, epsilon=1e-06
    - UCB Confidence: confidence=0.9, alpha=1.0
    - Learning Rates: regime=0.005, parameter=0.02, kalman=0.005
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize optimized hybrid online learning system.
        
        Args:
            window_size: Window size for feature extraction
        """
        self.window_size = window_size
        
        # Optimal parameters from tuning experiment
        self.optimal_parameters = {
            'adaptive_mirror_descent': {
                'learning_rate': 0.01,
                'beta': 0.99,
                'epsilon': 1e-06
            },
            'ucb_confidence': {
                'confidence': 0.9,
                'alpha': 1.0
            },
            'learning_rates': {
                'regime_detection': 0.005,
                'parameter_adaptation': 0.02,
                'kalman_optimization': 0.005
            }
        }
        
        # Initialize components with optimal parameters
        self.regime_system = RegimeDetectionSystem(
            window_size=window_size,
            confidence=self.optimal_parameters['ucb_confidence']['confidence']
        )
        
        self.param_system = ParameterAdaptationSystem(
            learning_rate=self.optimal_parameters['learning_rates']['parameter_adaptation'],
            beta=self.optimal_parameters['adaptive_mirror_descent']['beta']
        )
        
        self.kalman_system = KalmanOptimizationSystem(
            learning_rate=self.optimal_parameters['learning_rates']['kalman_optimization'],
            regularization=0.001
        )
        
        # Apply optimal parameters
        self._apply_optimal_parameters()
        
        # Performance tracking
        self.update_history = []
        self.performance_history = []
        self.optimization_metrics = {
            'regime_detection_improvement': 0.0,
            'parameter_adaptation_improvement': 0.0,
            'kalman_optimization_improvement': 0.0,
            'overall_improvement': 0.0
        }
        
        logger.info("Initialized OptimizedHybridOnlineLearningSystem with tuned parameters")
    
    def _apply_optimal_parameters(self):
        """Apply optimal parameters to all components"""
        
        # Apply UCB parameters
        self.regime_system.bandit.confidence = self.optimal_parameters['ucb_confidence']['confidence']
        self.regime_system.bandit.alpha = self.optimal_parameters['ucb_confidence']['alpha']
        
        # Apply Adaptive Mirror Descent parameters
        self.param_system.optimizer.eta = self.optimal_parameters['adaptive_mirror_descent']['learning_rate']
        self.param_system.optimizer.beta = self.optimal_parameters['adaptive_mirror_descent']['beta']
        self.param_system.optimizer.epsilon = self.optimal_parameters['adaptive_mirror_descent']['epsilon']
        
        # Apply learning rates
        self.regime_system.bandit.learning_rate = self.optimal_parameters['learning_rates']['regime_detection']
        self.kalman_system.optimizer.eta = self.optimal_parameters['learning_rates']['kalman_optimization']
    
    def process_financial_data(self, data: np.ndarray, 
                              observation: np.ndarray = None,
                              reward: float = None) -> OptimizedSystemResult:
        """
        Process financial data through the optimized hybrid system.
        
        Args:
            data: Financial data (time x 2) [ROI, risk]
            observation: Current observation (optional)
            reward: Reward signal (optional)
            
        Returns:
            OptimizedSystemResult with all system outputs
        """
        # 1. Regime Detection
        regime_result = self.regime_system.process_data(data)
        regime = regime_result.regime
        regime_confidence = regime_result.confidence
        
        # 2. Parameter Adaptation
        if observation is not None:
            # Extract features for parameter adaptation
            features = self._extract_adaptation_features(data, observation)
            adapted_params = self.param_system.adapt_parameters(
                features, observation[0], regime  # Use ROI as target
            )
        else:
            # Use default parameters
            adapted_params = self.param_system.optimizer.get_parameter_adaptation(regime)
        
        # 3. Kalman Optimization
        if observation is not None:
            # Use current state and observation for Kalman optimization
            current_state = np.array([data[-1, 0], data[-1, 1]])  # Latest ROI, risk
            kalman_matrices = self.kalman_system.optimize_kalman_parameters(
                current_state, observation
            )
        else:
            # Use current matrices
            kalman_matrices = self.kalman_system.optimizer.get_kalman_matrices()
        
        # 4. Update with reward if provided
        if reward is not None:
            self.regime_system.update_with_reward(reward)
        
        # 5. Compute regret bounds
        regret_bounds = {
            'regime_detection': self.regime_system.bandit.get_regret_bound(len(data)),
            'parameter_adaptation': self.param_system.optimizer.get_regret_bound(),
            'kalman_optimization': self.kalman_system.optimizer.get_regret_bound()
        }
        
        # 6. Compute optimization metrics
        optimization_metrics = self._compute_optimization_metrics(regret_bounds)
        
        # 7. Create result
        result = OptimizedSystemResult(
            regime=regime,
            regime_confidence=regime_confidence,
            adapted_parameters=adapted_params,
            kalman_matrices=kalman_matrices,
            regret_bounds=regret_bounds,
            optimization_metrics=optimization_metrics,
            timestamp=datetime.now().timestamp()
        )
        
        # 8. Store history
        self.update_history.append(result)
        if reward is not None:
            self.performance_history.append(reward)
        
        return result
    
    def _extract_adaptation_features(self, data: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Extract features for parameter adaptation.
        
        Args:
            data: Financial data
            observation: Current observation
            
        Returns:
            Feature vector for parameter adaptation
        """
        # Extract financial features
        roi_data = data[:, 0]
        risk_data = data[:, 1]
        
        # Compute features
        roi_trend = np.mean(roi_data[-self.window_size:])
        roi_volatility = np.std(roi_data[-self.window_size:])
        risk_trend = np.mean(risk_data[-self.window_size:])
        risk_volatility = np.std(risk_data[-self.window_size:])
        
        # Observation features
        obs_roi = observation[0]
        obs_risk = observation[1]
        
        # Combine features
        features = np.array([
            roi_trend, roi_volatility, risk_trend, risk_volatility,
            obs_roi, obs_risk
        ])
        
        return features
    
    def _compute_optimization_metrics(self, regret_bounds: Dict) -> Dict:
        """Compute optimization metrics"""
        
        # Baseline regret bounds (from original system)
        baseline_regret = {
            'regime_detection': 65.1356,
            'parameter_adaptation': 0.0644,
            'kalman_optimization': 84.8529
        }
        
        # Compute improvements
        improvements = {}
        for component, current_regret in regret_bounds.items():
            baseline = baseline_regret.get(component, current_regret)
            if baseline > 0:
                improvement = (baseline - current_regret) / baseline * 100
                improvements[f'{component}_improvement'] = improvement
            else:
                improvements[f'{component}_improvement'] = 0.0
        
        # Overall improvement
        overall_improvement = np.mean(list(improvements.values()))
        improvements['overall_improvement'] = overall_improvement
        
        return improvements
    
    def get_system_status(self) -> Dict:
        """
        Get complete system status.
        
        Returns:
            Dictionary with system status
        """
        # Get status from each component
        regime_status = self.regime_system.get_system_status()
        param_status = self.param_system.get_system_status()
        kalman_status = self.kalman_system.get_system_status()
        
        # Overall system metrics
        total_updates = len(self.update_history)
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0.0
        
        # Regret bounds summary
        if self.update_history:
            latest_result = self.update_history[-1]
            regret_summary = {
                'regime_detection': latest_result.regret_bounds['regime_detection'],
                'parameter_adaptation': latest_result.regret_bounds['parameter_adaptation'],
                'kalman_optimization': latest_result.regret_bounds['kalman_optimization'],
                'total_regret': sum(latest_result.regret_bounds.values())
            }
        else:
            regret_summary = {
                'regime_detection': 0.0,
                'parameter_adaptation': 0.0,
                'kalman_optimization': 0.0,
                'total_regret': 0.0
            }
        
        return {
            'system_overview': {
                'total_updates': total_updates,
                'average_performance': avg_performance,
                'confidence_level': self.optimal_parameters['ucb_confidence']['confidence'],
                'window_size': self.window_size,
                'optimization_status': 'OPTIMIZED'
            },
            'regime_detection': regime_status,
            'parameter_adaptation': param_status,
            'kalman_optimization': kalman_status,
            'regret_bounds': regret_summary,
            'optimization_metrics': self.optimization_metrics,
            'optimal_parameters': self.optimal_parameters
        }
    
    def get_optimization_summary(self) -> Dict:
        """
        Get optimization summary.
        
        Returns:
            Dictionary with optimization summary
        """
        if not self.update_history:
            return {
                'optimization_status': 'NO_DATA',
                'improvements': {},
                'optimal_parameters': self.optimal_parameters
            }
        
        # Get latest optimization metrics
        latest_result = self.update_history[-1]
        improvements = latest_result.optimization_metrics
        
        # Calculate average improvements
        avg_improvements = {
            'regime_detection': np.mean([r.optimization_metrics.get('regime_detection_improvement', 0) 
                                       for r in self.update_history]),
            'parameter_adaptation': np.mean([r.optimization_metrics.get('parameter_adaptation_improvement', 0) 
                                          for r in self.update_history]),
            'kalman_optimization': np.mean([r.optimization_metrics.get('kalman_optimization_improvement', 0) 
                                         for r in self.update_history]),
            'overall': np.mean([r.optimization_metrics.get('overall_improvement', 0) 
                              for r in self.update_history])
        }
        
        return {
            'optimization_status': 'OPTIMIZED',
            'improvements': avg_improvements,
            'optimal_parameters': self.optimal_parameters,
            'total_optimizations': len(self.update_history)
        }
    
    def reset_system(self) -> None:
        """Reset the entire system."""
        self.regime_system.reset_system()
        self.param_system.reset_system()
        self.kalman_system.reset_system()
        self.update_history = []
        self.performance_history = []
        self.optimization_metrics = {
            'regime_detection_improvement': 0.0,
            'parameter_adaptation_improvement': 0.0,
            'kalman_optimization_improvement': 0.0,
            'overall_improvement': 0.0
        }
        
        # Reapply optimal parameters
        self._apply_optimal_parameters()
        
        logger.info("Reset OptimizedHybridOnlineLearningSystem to initial state")

def main():
    """Test the optimized hybrid online learning system."""
    import matplotlib.pyplot as plt
    
    # Generate test data
    np.random.seed(42)
    n_periods = 100
    
    # Generate financial data with regime changes
    data = []
    observations = []
    rewards = []
    
    # Bull market (periods 0-30)
    for i in range(30):
        roi = 0.02 + np.random.normal(0, 0.01)
        risk = 0.10 + np.random.normal(0, 0.02)
        data.append([roi, risk])
        observations.append([roi, risk])
        rewards.append(1.0)  # Positive reward for bull market
    
    # Bear market (periods 30-60)
    for i in range(30):
        roi = -0.01 + np.random.normal(0, 0.02)
        risk = 0.20 + np.random.normal(0, 0.03)
        data.append([roi, risk])
        observations.append([roi, risk])
        rewards.append(0.0)  # Neutral reward for bear market
    
    # Sideways market (periods 60-100)
    for i in range(40):
        roi = 0.005 + np.random.normal(0, 0.015)
        risk = 0.15 + np.random.normal(0, 0.025)
        data.append([roi, risk])
        observations.append([roi, risk])
        rewards.append(0.5)  # Moderate reward for sideways market
    
    data = np.array(data)
    observations = np.array(observations)
    rewards = np.array(rewards)
    
    # Initialize optimized system
    optimized_system = OptimizedHybridOnlineLearningSystem(window_size=10)
    
    # Process data
    results = []
    for i in range(len(data)):
        window_data = data[:i+1]
        observation = observations[i] if i < len(observations) else None
        reward = rewards[i] if i < len(rewards) else None
        
        result = optimized_system.process_financial_data(window_data, observation, reward)
        results.append(result)
    
    # Print results
    print("Optimized Hybrid Online Learning System Results:")
    print("=" * 60)
    
    # System status
    status = optimized_system.get_system_status()
    print(f"Total Updates: {status['system_overview']['total_updates']}")
    print(f"Average Performance: {status['system_overview']['average_performance']:.4f}")
    print(f"Optimization Status: {status['system_overview']['optimization_status']}")
    
    # Optimization summary
    optimization_summary = optimized_system.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Status: {optimization_summary['optimization_status']}")
    print(f"Improvements: {optimization_summary['improvements']}")
    
    # Regret bounds
    regret_bounds = status['regret_bounds']
    print(f"\nRegret Bounds:")
    print(f"  Regime Detection: {regret_bounds['regime_detection']:.4f}")
    print(f"  Parameter Adaptation: {regret_bounds['parameter_adaptation']:.4f}")
    print(f"  Kalman Optimization: {regret_bounds['kalman_optimization']:.4f}")
    print(f"  Total Regret: {regret_bounds['total_regret']:.4f}")
    
    # Optimal parameters
    optimal_params = status['optimal_parameters']
    print(f"\nOptimal Parameters:")
    for component, params in optimal_params.items():
        print(f"  {component}: {params}")

if __name__ == "__main__":
    main()
