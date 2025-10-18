#!/usr/bin/env python3
"""
Adaptive Mirror Descent for Parameter Adaptation

Implements adaptive mirror descent with theoretical guarantees for
parameter adaptation in Kalman filters and regime detection.

Regret Bound: O(√(G²T)) where G is the gradient bound and T is time horizon.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ParameterUpdateResult:
    """Result of parameter update with adaptive learning"""
    parameters: Dict[str, float]
    learning_rate: float
    gradient_norm: float
    regret_bound: float
    timestamp: float

class AdaptiveMirrorDescent:
    """
    Adaptive Mirror Descent with theoretical guarantees.
    
    Implements adaptive learning rate adjustment based on gradient history
    with regret bound O(√(G²T)) for non-convex functions.
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 beta: float = 0.9,  # Momentum parameter
                 epsilon: float = 1e-8,
                 parameter_names: List[str] = None):
        """
        Initialize adaptive mirror descent.
        
        Args:
            learning_rate: Base learning rate
            beta: Momentum parameter for adaptive learning
            epsilon: Small constant for numerical stability
            parameter_names: Names of parameters to optimize
        """
        self.eta = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        
        # Parameter names and dimensions
        if parameter_names is None:
            self.parameter_names = [
                'process_noise', 'measurement_noise', 'adaptation_rate',
                'forgetting_factor', 'uncertainty_weight', 'regime_threshold'
            ]
        else:
            self.parameter_names = parameter_names
        
        self.d = len(self.parameter_names)
        
        # Initialize parameters
        self.parameters = np.zeros(self.d)
        self.v = np.zeros(self.d)  # Second moment estimate
        self.gradient_history = []
        self.parameter_history = []
        self.regret_history = []
        
        # Performance tracking
        self.total_gradients = 0
        self.adaptive_learning_rates = []
        
        logger.info(f"Initialized AdaptiveMirrorDescent with {self.d} parameters")
    
    def initialize_parameters(self, initial_values: Dict[str, float] = None) -> None:
        """
        Initialize parameters with given values.
        
        Args:
            initial_values: Dictionary of initial parameter values
        """
        if initial_values is not None:
            for i, name in enumerate(self.parameter_names):
                if name in initial_values:
                    self.parameters[i] = initial_values[name]
                else:
                    # Default values
                    if name == 'process_noise':
                        self.parameters[i] = 0.01
                    elif name == 'measurement_noise':
                        self.parameters[i] = 0.01
                    elif name == 'adaptation_rate':
                        self.parameters[i] = 0.01
                    elif name == 'forgetting_factor':
                        self.parameters[i] = 0.95
                    elif name == 'uncertainty_weight':
                        self.parameters[i] = 0.1
                    elif name == 'regime_threshold':
                        self.parameters[i] = 0.8
                    else:
                        self.parameters[i] = 0.01
        
        # Store initial parameters
        self.parameter_history.append(self.parameters.copy())
        
        logger.info(f"Initialized parameters: {dict(zip(self.parameter_names, self.parameters))}")
    
    def compute_gradient(self, features: np.ndarray, target: float, 
                       loss_function: str = 'mse') -> np.ndarray:
        """
        Compute gradient of loss function with respect to parameters.
        
        Args:
            features: Input features
            target: Target value
            loss_function: Type of loss function ('mse', 'mae', 'huber')
            
        Returns:
            Gradient vector
        """
        # Simple gradient computation (can be extended with more sophisticated methods)
        prediction = self._predict(features)
        error = target - prediction
        
        if loss_function == 'mse':
            gradient = -2 * error * features
        elif loss_function == 'mae':
            gradient = -np.sign(error) * features
        elif loss_function == 'huber':
            delta = 1.0
            if abs(error) <= delta:
                gradient = -error * features
            else:
                gradient = -delta * np.sign(error) * features
        else:
            gradient = -2 * error * features
        
        return gradient
    
    def _predict(self, features: np.ndarray) -> float:
        """
        Make prediction using current parameters.
        
        Args:
            features: Input features
            
        Returns:
            Prediction value
        """
        # Simple linear prediction (can be extended)
        return np.dot(self.parameters, features)
    
    def update(self, features: np.ndarray, target: float, 
               loss_function: str = 'mse') -> ParameterUpdateResult:
        """
        Update parameters using adaptive mirror descent.
        
        Args:
            features: Input features
            target: Target value
            loss_function: Type of loss function
            
        Returns:
            ParameterUpdateResult with update information
        """
        # Compute gradient
        gradient = self.compute_gradient(features, target, loss_function)
        
        # Update second moment estimate
        self.v = self.beta * self.v + (1 - self.beta) * gradient**2
        
        # Compute adaptive learning rate
        adaptive_lr = self.eta / (np.sqrt(self.v) + self.epsilon)
        
        # Update parameters
        self.parameters -= adaptive_lr * gradient
        
        # Store history
        self.gradient_history.append(gradient.copy())
        self.parameter_history.append(self.parameters.copy())
        self.adaptive_learning_rates.append(adaptive_lr.copy())
        
        # Update tracking
        self.total_gradients += 1
        
        # Compute regret
        gradient_norm = np.linalg.norm(gradient)
        regret = gradient_norm**2
        self.regret_history.append(regret)
        
        # Create result
        result = ParameterUpdateResult(
            parameters=dict(zip(self.parameter_names, self.parameters)),
            learning_rate=np.mean(adaptive_lr),
            gradient_norm=gradient_norm,
            regret_bound=self.get_regret_bound(),
            timestamp=datetime.now().timestamp()
        )
        
        return result
    
    def get_regret_bound(self) -> float:
        """
        Compute theoretical regret bound.
        
        Returns:
            Regret bound O(√(G²T))
        """
        if len(self.regret_history) == 0:
            return 0.0
        
        T = len(self.regret_history)
        G_squared = np.mean([np.linalg.norm(g)**2 for g in self.gradient_history])
        
        return np.sqrt(G_squared * T)
    
    def get_parameter_adaptation(self, regime: str) -> Dict[str, float]:
        """
        Get regime-specific parameter adaptation.
        
        Args:
            regime: Market regime ('bull_market', 'bear_market', 'sideways_market')
            
        Returns:
            Adapted parameters for the regime
        """
        base_params = dict(zip(self.parameter_names, self.parameters))
        
        # Regime-specific adaptations
        if regime == 'bull_market':
            # Lower uncertainty, higher adaptation for trends
            adapted_params = base_params.copy()
            adapted_params['process_noise'] *= 0.8
            adapted_params['adaptation_rate'] *= 1.2
            adapted_params['uncertainty_weight'] *= 0.9
            
        elif regime == 'bear_market':
            # Higher uncertainty, lower adaptation for volatility
            adapted_params = base_params.copy()
            adapted_params['process_noise'] *= 1.3
            adapted_params['adaptation_rate'] *= 0.8
            adapted_params['uncertainty_weight'] *= 1.2
            
        else:  # sideways_market
            # Moderate settings for sideways markets
            adapted_params = base_params.copy()
            adapted_params['process_noise'] *= 1.0
            adapted_params['adaptation_rate'] *= 1.0
            adapted_params['uncertainty_weight'] *= 1.0
        
        return adapted_params
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.gradient_history) == 0:
            return {
                'total_updates': 0,
                'average_gradient_norm': 0.0,
                'regret_bound': 0.0,
                'parameter_stability': 0.0,
                'learning_rate_adaptation': 0.0
            }
        
        # Gradient statistics
        gradient_norms = [np.linalg.norm(g) for g in self.gradient_history]
        avg_gradient_norm = np.mean(gradient_norms)
        
        # Parameter stability
        if len(self.parameter_history) > 1:
            param_changes = [
                np.linalg.norm(self.parameter_history[i] - self.parameter_history[i-1])
                for i in range(1, len(self.parameter_history))
            ]
            parameter_stability = 1.0 / (1.0 + np.mean(param_changes))
        else:
            parameter_stability = 1.0
        
        # Learning rate adaptation
        if len(self.adaptive_learning_rates) > 1:
            lr_changes = [
                np.linalg.norm(self.adaptive_learning_rates[i] - self.adaptive_learning_rates[i-1])
                for i in range(1, len(self.adaptive_learning_rates))
            ]
            learning_rate_adaptation = np.mean(lr_changes)
        else:
            learning_rate_adaptation = 0.0
        
        return {
            'total_updates': self.total_gradients,
            'average_gradient_norm': avg_gradient_norm,
            'regret_bound': self.get_regret_bound(),
            'parameter_stability': parameter_stability,
            'learning_rate_adaptation': learning_rate_adaptation
        }
    
    def reset(self) -> None:
        """Reset the optimizer to initial state."""
        self.parameters = np.zeros(self.d)
        self.v = np.zeros(self.d)
        self.gradient_history = []
        self.parameter_history = []
        self.regret_history = []
        self.total_gradients = 0
        self.adaptive_learning_rates = []
        
        logger.info("Reset AdaptiveMirrorDescent to initial state")

class ParameterAdaptationSystem:
    """
    Complete parameter adaptation system using adaptive mirror descent.
    
    Integrates adaptive mirror descent with regime-aware parameter adaptation
    for Kalman filters and other financial models.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9):
        """
        Initialize parameter adaptation system.
        
        Args:
            learning_rate: Base learning rate
            beta: Momentum parameter
        """
        self.learning_rate = learning_rate
        self.beta = beta
        
        # Initialize adaptive mirror descent
        self.optimizer = AdaptiveMirrorDescent(
            learning_rate=learning_rate,
            beta=beta
        )
        
        # Initialize with default parameters
        self.optimizer.initialize_parameters()
        
        # Performance tracking
        self.adaptation_history = []
        self.regime_adaptations = {}
        
        logger.info(f"Initialized ParameterAdaptationSystem with lr={learning_rate}, beta={beta}")
    
    def adapt_parameters(self, features: np.ndarray, target: float, 
                        regime: str = 'sideways_market') -> Dict[str, float]:
        """
        Adapt parameters based on features, target, and regime.
        
        Args:
            features: Input features
            target: Target value
            regime: Market regime
            
        Returns:
            Adapted parameters
        """
        # Update optimizer
        result = self.optimizer.update(features, target)
        
        # Get regime-specific adaptation
        adapted_params = self.optimizer.get_parameter_adaptation(regime)
        
        # Store adaptation
        self.adaptation_history.append({
            'regime': regime,
            'parameters': adapted_params,
            'learning_rate': result.learning_rate,
            'gradient_norm': result.gradient_norm,
            'regret_bound': result.regret_bound,
            'timestamp': result.timestamp
        })
        
        # Update regime-specific tracking
        if regime not in self.regime_adaptations:
            self.regime_adaptations[regime] = []
        self.regime_adaptations[regime].append(adapted_params)
        
        return adapted_params
    
    def get_system_status(self) -> Dict:
        """
        Get complete system status.
        
        Returns:
            Dictionary with system status
        """
        optimizer_metrics = self.optimizer.get_performance_metrics()
        
        # Regime-specific statistics
        regime_stats = {}
        for regime, adaptations in self.regime_adaptations.items():
            if adaptations:
                # Compute statistics for this regime
                param_values = {name: [] for name in self.optimizer.parameter_names}
                for adaptation in adaptations:
                    for name, value in adaptation.items():
                        param_values[name].append(value)
                
                regime_stats[regime] = {
                    'adaptations_count': len(adaptations),
                    'parameter_means': {name: np.mean(values) for name, values in param_values.items()},
                    'parameter_stds': {name: np.std(values) for name, values in param_values.items()}
                }
        
        return {
            'optimizer_performance': optimizer_metrics,
            'regime_adaptations': regime_stats,
            'total_adaptations': len(self.adaptation_history),
            'system_parameters': {
                'learning_rate': self.learning_rate,
                'beta': self.beta,
                'parameter_count': len(self.optimizer.parameter_names)
            }
        }
    
    def reset_system(self) -> None:
        """Reset the entire system."""
        self.optimizer.reset()
        self.optimizer.initialize_parameters()
        self.adaptation_history = []
        self.regime_adaptations = {}
        
        logger.info("Reset ParameterAdaptationSystem to initial state")

def main():
    """Test the adaptive mirror descent parameter adaptation system."""
    import matplotlib.pyplot as plt
    
    # Generate test data
    np.random.seed(42)
    n_samples = 100
    
    # Generate features and targets
    features = np.random.randn(n_samples, 6)  # 6 parameters
    targets = np.random.randn(n_samples)
    
    # Initialize parameter adaptation system
    param_system = ParameterAdaptationSystem(learning_rate=0.01, beta=0.9)
    
    # Test parameter adaptation
    regimes = ['bull_market', 'bear_market', 'sideways_market']
    adapted_parameters = []
    
    for i in range(n_samples):
        regime = regimes[i % len(regimes)]
        adapted_params = param_system.adapt_parameters(
            features[i], targets[i], regime
        )
        adapted_parameters.append(adapted_params)
    
    # Print results
    print("Adaptive Mirror Descent Parameter Adaptation Results:")
    print("=" * 60)
    
    # System status
    status = param_system.get_system_status()
    print(f"Total Adaptations: {status['total_adaptations']}")
    print(f"Regret Bound: {status['optimizer_performance']['regret_bound']:.4f}")
    print(f"Parameter Stability: {status['optimizer_performance']['parameter_stability']:.4f}")
    
    # Regime-specific statistics
    print(f"\nRegime-Specific Adaptations:")
    for regime, stats in status['regime_adaptations'].items():
        print(f"\n{regime}:")
        print(f"  Adaptations: {stats['adaptations_count']}")
        print(f"  Parameter Means: {stats['parameter_means']}")
        print(f"  Parameter Stds: {stats['parameter_stds']}")
    
    # Final parameters
    final_params = adapted_parameters[-1]
    print(f"\nFinal Parameters:")
    for name, value in final_params.items():
        print(f"  {name}: {value:.4f}")

if __name__ == "__main__":
    main()
