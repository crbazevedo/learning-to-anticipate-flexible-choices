#!/usr/bin/env python3
"""
Hybrid Online Learning System

Integrates contextual bandits, adaptive mirror descent, and online Newton step
into a complete system with theoretical guarantees for financial applications.

Components:
- Contextual Bandits: O(√(dT log T)) regret for regime detection
- Adaptive Mirror Descent: O(√(G²T)) regret for parameter adaptation
- Online Newton Step: O(d log T) regret for Kalman optimization
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
class HybridSystemResult:
    """Result of hybrid system update"""
    regime: str
    regime_confidence: float
    adapted_parameters: Dict[str, float]
    kalman_matrices: Dict[str, np.ndarray]
    regret_bounds: Dict[str, float]
    timestamp: float

class HybridOnlineLearningSystem:
    """
    Complete hybrid online learning system with theoretical guarantees.
    
    Integrates:
    1. Contextual Bandits for regime detection
    2. Adaptive Mirror Descent for parameter adaptation
    3. Online Newton Step for Kalman optimization
    """
    
    def __init__(self, learning_rates: Dict[str, float] = None,
                 confidence: float = 0.95,
                 window_size: int = 10):
        """
        Initialize hybrid online learning system.
        
        Args:
            learning_rates: Learning rates for each component
            confidence: Confidence level for UCB
            window_size: Window size for feature extraction
        """
        # Default learning rates
        if learning_rates is None:
            learning_rates = {
                'regime_detection': 0.01,
                'parameter_adaptation': 0.01,
                'kalman_optimization': 0.01
            }
        
        self.learning_rates = learning_rates
        self.confidence = confidence
        self.window_size = window_size
        
        # Initialize components
        self.regime_system = RegimeDetectionSystem(
            window_size=window_size,
            confidence=confidence
        )
        
        self.param_system = ParameterAdaptationSystem(
            learning_rate=learning_rates['parameter_adaptation'],
            beta=0.9
        )
        
        self.kalman_system = KalmanOptimizationSystem(
            learning_rate=learning_rates['kalman_optimization'],
            regularization=0.001
        )
        
        # Performance tracking
        self.update_history = []
        self.performance_history = []
        
        logger.info("Initialized HybridOnlineLearningSystem with all components")
    
    def process_financial_data(self, data: np.ndarray, 
                              observation: np.ndarray = None,
                              reward: float = None) -> HybridSystemResult:
        """
        Process financial data through the complete hybrid system.
        
        Args:
            data: Financial data (time x 2) [ROI, risk]
            observation: Current observation (optional)
            reward: Reward signal (optional)
            
        Returns:
            HybridSystemResult with all system outputs
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
        
        # 6. Create result
        result = HybridSystemResult(
            regime=regime,
            regime_confidence=regime_confidence,
            adapted_parameters=adapted_params,
            kalman_matrices=kalman_matrices,
            regret_bounds=regret_bounds,
            timestamp=datetime.now().timestamp()
        )
        
        # 7. Store history
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
                'confidence_level': self.confidence,
                'window_size': self.window_size
            },
            'regime_detection': regime_status,
            'parameter_adaptation': param_status,
            'kalman_optimization': kalman_status,
            'regret_bounds': regret_summary
        }
    
    def get_current_configuration(self) -> Dict:
        """
        Get current system configuration.
        
        Returns:
            Dictionary with current configuration
        """
        if not self.update_history:
            return {
                'regime': 'unknown',
                'parameters': {},
                'kalman_matrices': {},
                'regret_bounds': {}
            }
        
        latest_result = self.update_history[-1]
        
        return {
            'regime': latest_result.regime,
            'regime_confidence': latest_result.regime_confidence,
            'parameters': latest_result.adapted_parameters,
            'kalman_matrices': latest_result.kalman_matrices,
            'regret_bounds': latest_result.regret_bounds
        }
    
    def reset_system(self) -> None:
        """Reset the entire system."""
        self.regime_system.reset_system()
        self.param_system.reset_system()
        self.kalman_system.reset_system()
        self.update_history = []
        self.performance_history = []
        
        logger.info("Reset HybridOnlineLearningSystem to initial state")

class HybridSystemExperiment:
    """
    Experiment class for testing the hybrid online learning system.
    """
    
    def __init__(self, system: HybridOnlineLearningSystem):
        """
        Initialize experiment.
        
        Args:
            system: Hybrid online learning system
        """
        self.system = system
        self.experiment_results = []
    
    def run_experiment(self, data: np.ndarray, 
                      observations: np.ndarray = None,
                      rewards: np.ndarray = None,
                      experiment_name: str = "hybrid_experiment") -> Dict:
        """
        Run experiment with the hybrid system.
        
        Args:
            data: Financial data (time x 2)
            observations: Observation data (optional)
            rewards: Reward data (optional)
            experiment_name: Name of the experiment
            
        Returns:
            Experiment results
        """
        results = {
            'experiment_name': experiment_name,
            'data_shape': data.shape,
            'system_updates': [],
            'performance_metrics': {},
            'regret_analysis': {}
        }
        
        # Process data through system
        for i in range(len(data)):
            # Get data window
            window_data = data[:i+1]
            
            # Get observation and reward if available
            observation = observations[i] if observations is not None else None
            reward = rewards[i] if rewards is not None else None
            
            # Process through system
            result = self.system.process_financial_data(
                window_data, observation, reward
            )
            
            # Store result
            results['system_updates'].append({
                'timestep': i,
                'regime': result.regime,
                'regime_confidence': result.regime_confidence,
                'regret_bounds': result.regret_bounds
            })
        
        # Compute performance metrics
        if self.system.performance_history:
            results['performance_metrics'] = {
                'total_rewards': len(self.system.performance_history),
                'average_reward': np.mean(self.system.performance_history),
                'reward_std': np.std(self.system.performance_history),
                'cumulative_reward': np.sum(self.system.performance_history)
            }
        
        # Regret analysis
        if self.system.update_history:
            latest_result = self.system.update_history[-1]
            results['regret_analysis'] = {
                'final_regret_bounds': latest_result.regret_bounds,
                'total_regret': sum(latest_result.regret_bounds.values()),
                'regime_detection_accuracy': self._compute_regime_accuracy()
            }
        
        # Store experiment results
        self.experiment_results.append(results)
        
        return results
    
    def _compute_regime_accuracy(self) -> float:
        """Compute regime detection accuracy."""
        if len(self.system.update_history) < 2:
            return 0.0
        
        # Simple accuracy based on regime consistency
        regimes = [result.regime for result in self.system.update_history]
        regime_changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        
        if len(regimes) > 1:
            stability = 1.0 - (regime_changes / (len(regimes) - 1))
            return max(0.0, stability)
        
        return 0.0
    
    def get_experiment_summary(self) -> Dict:
        """
        Get summary of all experiments.
        
        Returns:
            Dictionary with experiment summary
        """
        if not self.experiment_results:
            return {'experiments': 0, 'summary': {}}
        
        # Aggregate results
        total_experiments = len(self.experiment_results)
        avg_performance = np.mean([
            exp['performance_metrics'].get('average_reward', 0.0)
            for exp in self.experiment_results
        ])
        
        avg_regret = np.mean([
            exp['regret_analysis'].get('total_regret', 0.0)
            for exp in self.experiment_results
        ])
        
        return {
            'experiments': total_experiments,
            'summary': {
                'average_performance': avg_performance,
                'average_regret': avg_regret,
                'system_status': self.system.get_system_status()
            }
        }

def main():
    """Test the hybrid online learning system."""
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
    
    # Initialize hybrid system
    hybrid_system = HybridOnlineLearningSystem(
        learning_rates={
            'regime_detection': 0.01,
            'parameter_adaptation': 0.01,
            'kalman_optimization': 0.01
        },
        confidence=0.95,
        window_size=10
    )
    
    # Run experiment
    experiment = HybridSystemExperiment(hybrid_system)
    results = experiment.run_experiment(
        data, observations, rewards, "hybrid_online_learning_test"
    )
    
    # Print results
    print("Hybrid Online Learning System Results:")
    print("=" * 60)
    
    # System status
    status = hybrid_system.get_system_status()
    print(f"Total Updates: {status['system_overview']['total_updates']}")
    print(f"Average Performance: {status['system_overview']['average_performance']:.4f}")
    
    # Regret bounds
    regret_bounds = status['regret_bounds']
    print(f"\nRegret Bounds:")
    print(f"  Regime Detection: {regret_bounds['regime_detection']:.4f}")
    print(f"  Parameter Adaptation: {regret_bounds['parameter_adaptation']:.4f}")
    print(f"  Kalman Optimization: {regret_bounds['kalman_optimization']:.4f}")
    print(f"  Total Regret: {regret_bounds['total_regret']:.4f}")
    
    # Current configuration
    config = hybrid_system.get_current_configuration()
    print(f"\nCurrent Configuration:")
    print(f"  Regime: {config['regime']}")
    print(f"  Regime Confidence: {config['regime_confidence']:.2%}")
    print(f"  Parameters: {config['parameters']}")
    
    # Experiment summary
    summary = experiment.get_experiment_summary()
    print(f"\nExperiment Summary:")
    print(f"  Experiments: {summary['experiments']}")
    print(f"  Average Performance: {summary['summary']['average_performance']:.4f}")
    print(f"  Average Regret: {summary['summary']['average_regret']:.4f}")

if __name__ == "__main__":
    main()
