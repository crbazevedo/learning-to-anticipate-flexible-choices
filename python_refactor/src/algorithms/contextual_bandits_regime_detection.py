#!/usr/bin/env python3
"""
Contextual Bandits for Regime Detection

Implements contextual bandits with Upper Confidence Bound (UCB) for robust
regime detection with theoretical guarantees.

Regret Bound: O(√(dT log T)) where d is context dimension and T is time horizon.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RegimeDetectionResult:
    """Result of regime detection with confidence bounds"""
    regime: str
    confidence: float
    ucb_value: float
    regret_bound: float
    timestamp: float

class ContextualBanditUCB:
    """
    Contextual Bandits with Upper Confidence Bound for regime detection.
    
    Implements the LinUCB algorithm with theoretical guarantees:
    - Regret Bound: O(√(dT log T))
    - Confidence: 95% by default
    - Context dimension: 4 (ROI, risk, trend, volatility)
    """
    
    def __init__(self, num_arms: int = 3,  # bull, bear, sideways
                 context_dim: int = 4,
                 confidence: float = 0.95,
                 regularization: float = 0.001):
        """
        Initialize contextual bandit for regime detection.
        
        Args:
            num_arms: Number of regimes (arms)
            context_dim: Dimension of context features
            confidence: Confidence level for UCB
            regularization: Regularization parameter
        """
        self.K = num_arms
        self.d = context_dim
        self.confidence = confidence
        self.regularization = regularization
        
        # UCB parameters for each arm (regime)
        self.A = [np.eye(context_dim) * regularization for _ in range(num_arms)]
        self.b = [np.zeros(context_dim) for _ in range(num_arms)]
        self.theta = [np.zeros(context_dim) for _ in range(num_arms)]
        
        # Regime names
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
        
        # Performance tracking
        self.total_rewards = [0.0] * num_arms
        self.pull_counts = [0] * num_arms
        self.regret_history = []
        self.regime_history = []
        
        # UCB parameters
        self.alpha = np.sqrt(2 * np.log(1 / (1 - confidence)))
        
        logger.info(f"Initialized ContextualBanditUCB with {num_arms} arms, {context_dim}D context")
    
    def extract_financial_features(self, data: np.ndarray, window: int = 10) -> np.ndarray:
        """
        Extract financial features for regime detection.
        
        Args:
            data: Financial data (time x 2) [ROI, risk]
            window: Window size for feature extraction
            
        Returns:
            Context features [ROI_trend, volatility, momentum, mean_reversion]
        """
        if len(data) < window:
            # Use available data
            window = len(data)
        
        recent_data = data[-window:]
        roi_data = recent_data[:, 0]
        risk_data = recent_data[:, 1]
        
        # ROI trend (moving average)
        roi_trend = np.mean(roi_data)
        
        # Volatility (standard deviation)
        volatility = np.std(roi_data)
        
        # Momentum (rate of change)
        if len(roi_data) > 1:
            momentum = (roi_data[-1] - roi_data[0]) / len(roi_data)
        else:
            momentum = 0.0
        
        # Mean reversion (distance from mean)
        mean_reversion = abs(roi_data[-1] - np.mean(roi_data))
        
        # Normalize features
        features = np.array([roi_trend, volatility, momentum, mean_reversion])
        
        # Simple normalization to [0, 1] range
        features = np.clip(features, -1, 1)
        
        return features
    
    def select_arm(self, context: np.ndarray) -> Tuple[int, float]:
        """
        Select arm (regime) using UCB.
        
        Args:
            context: Context features
            
        Returns:
            Tuple of (arm_index, ucb_value)
        """
        ucb_values = []
        
        for k in range(self.K):
            try:
                # Compute confidence interval
                A_inv = np.linalg.inv(self.A[k])
                theta_k = A_inv @ self.b[k]
                
                # UCB value
                confidence_radius = self.alpha * np.sqrt(
                    context.T @ A_inv @ context
                )
                
                ucb_value = np.dot(theta_k, context) + confidence_radius
                ucb_values.append(ucb_value)
                
            except np.linalg.LinAlgError:
                # Fallback to random selection if matrix is singular
                ucb_values.append(np.random.random())
        
        # Select arm with highest UCB value
        selected_arm = np.argmax(ucb_values)
        max_ucb = ucb_values[selected_arm]
        
        return selected_arm, max_ucb
    
    def update(self, context: np.ndarray, arm: int, reward: float) -> None:
        """
        Update parameters after receiving reward.
        
        Args:
            context: Context features
            arm: Selected arm
            reward: Received reward
        """
        # Update matrices
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update theta
        try:
            self.theta[arm] = np.linalg.inv(self.A[arm]) @ self.b[arm]
        except np.linalg.LinAlgError:
            # Keep previous theta if matrix is singular
            pass
        
        # Update tracking
        self.total_rewards[arm] += reward
        self.pull_counts[arm] += 1
        
        # Compute regret
        best_reward = max(self.total_rewards)
        current_regret = best_reward - self.total_rewards[arm]
        self.regret_history.append(current_regret)
    
    def detect_regime(self, data: np.ndarray, window: int = 10) -> RegimeDetectionResult:
        """
        Detect market regime using contextual bandits.
        
        Args:
            data: Financial data (time x 2) [ROI, risk]
            window: Window size for feature extraction
            
        Returns:
            RegimeDetectionResult with regime and confidence
        """
        # Extract features
        context = self.extract_financial_features(data, window)
        
        # Select regime
        arm, ucb_value = self.select_arm(context)
        regime = self.regime_names[arm]
        
        # Compute confidence
        confidence = min(ucb_value / (ucb_value + 1), 0.99)  # Normalize to [0, 1]
        
        # Compute regret bound
        T = len(self.regret_history) + 1
        regret_bound = self.get_regret_bound(T)
        
        # Store regime history
        self.regime_history.append(regime)
        
        return RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            ucb_value=ucb_value,
            regret_bound=regret_bound,
            timestamp=datetime.now().timestamp()
        )
    
    def get_regret_bound(self, T: int) -> float:
        """
        Compute theoretical regret bound.
        
        Args:
            T: Time horizon
            
        Returns:
            Regret bound O(√(dT log T))
        """
        if T <= 0:
            return 0.0
        
        return np.sqrt(self.d * T * np.log(T + 1))
    
    def get_performance_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        total_pulls = sum(self.pull_counts)
        if total_pulls == 0:
            return {
                'total_pulls': 0,
                'regime_distribution': {},
                'average_rewards': {},
                'regret_bound': 0.0,
                'exploration_rate': 0.0
            }
        
        # Regime distribution
        regime_distribution = {
            self.regime_names[i]: self.pull_counts[i] / total_pulls
            for i in range(self.K)
        }
        
        # Average rewards
        average_rewards = {
            self.regime_names[i]: self.total_rewards[i] / max(self.pull_counts[i], 1)
            for i in range(self.K)
        }
        
        # Exploration rate (how often we explore vs exploit)
        exploration_rate = sum(1 for r in self.regret_history if r > 0) / len(self.regret_history)
        
        return {
            'total_pulls': total_pulls,
            'regime_distribution': regime_distribution,
            'average_rewards': average_rewards,
            'regret_bound': self.get_regret_bound(total_pulls),
            'exploration_rate': exploration_rate
        }
    
    def reset(self) -> None:
        """Reset the bandit to initial state."""
        self.A = [np.eye(self.d) * self.regularization for _ in range(self.K)]
        self.b = [np.zeros(self.d) for _ in range(self.K)]
        self.theta = [np.zeros(self.d) for _ in range(self.K)]
        self.total_rewards = [0.0] * self.K
        self.pull_counts = [0] * self.K
        self.regret_history = []
        self.regime_history = []
        
        logger.info("Reset ContextualBanditUCB to initial state")

class RegimeDetectionSystem:
    """
    Complete regime detection system using contextual bandits.
    
    Integrates contextual bandits with financial data processing and
    provides a complete regime detection solution.
    """
    
    def __init__(self, window_size: int = 10, confidence: float = 0.95):
        """
        Initialize regime detection system.
        
        Args:
            window_size: Window size for feature extraction
            confidence: Confidence level for UCB
        """
        self.window_size = window_size
        self.confidence = confidence
        
        # Initialize contextual bandit
        self.bandit = ContextualBanditUCB(
            num_arms=3,
            context_dim=4,
            confidence=confidence
        )
        
        # Data storage
        self.data_history = []
        self.regime_history = []
        self.performance_history = []
        
        logger.info(f"Initialized RegimeDetectionSystem with window_size={window_size}")
    
    def process_data(self, data: np.ndarray) -> RegimeDetectionResult:
        """
        Process financial data and detect regime.
        
        Args:
            data: Financial data (time x 2) [ROI, risk]
            
        Returns:
            RegimeDetectionResult
        """
        # Store data
        self.data_history.append(data.copy())
        
        # Detect regime
        result = self.bandit.detect_regime(data, self.window_size)
        
        # Store result
        self.regime_history.append(result)
        
        return result
    
    def update_with_reward(self, reward: float) -> None:
        """
        Update system with reward signal.
        
        Args:
            reward: Reward signal (e.g., portfolio performance)
        """
        if len(self.data_history) > 0:
            # Use most recent data for update
            recent_data = self.data_history[-1]
            context = self.bandit.extract_financial_features(recent_data, self.window_size)
            
            # Get most recent regime
            if len(self.regime_history) > 0:
                recent_regime = self.regime_history[-1]
                arm = self.bandit.regime_names.index(recent_regime.regime)
                
                # Update bandit
                self.bandit.update(context, arm, reward)
                
                # Store performance
                self.performance_history.append(reward)
    
    def get_system_status(self) -> Dict:
        """
        Get complete system status.
        
        Returns:
            Dictionary with system status
        """
        bandit_metrics = self.bandit.get_performance_metrics()
        
        return {
            'regime_detection': {
                'current_regime': self.regime_history[-1].regime if self.regime_history else 'unknown',
                'confidence': self.regime_history[-1].confidence if self.regime_history else 0.0,
                'regret_bound': self.regime_history[-1].regret_bound if self.regime_history else 0.0
            },
            'bandit_performance': bandit_metrics,
            'data_processing': {
                'total_samples': len(self.data_history),
                'window_size': self.window_size,
                'confidence_level': self.confidence
            },
            'performance_tracking': {
                'total_rewards': len(self.performance_history),
                'average_reward': np.mean(self.performance_history) if self.performance_history else 0.0,
                'reward_std': np.std(self.performance_history) if self.performance_history else 0.0
            }
        }
    
    def reset_system(self) -> None:
        """Reset the entire system."""
        self.bandit.reset()
        self.data_history = []
        self.regime_history = []
        self.performance_history = []
        
        logger.info("Reset RegimeDetectionSystem to initial state")

def main():
    """Test the contextual bandits regime detection system."""
    import matplotlib.pyplot as plt
    
    # Generate test data
    np.random.seed(42)
    n_periods = 200
    
    # Generate regime-specific data
    data = []
    true_regimes = []
    
    # Bull market (periods 0-50)
    for i in range(50):
        roi = 0.02 + np.random.normal(0, 0.01)
        risk = 0.10 + np.random.normal(0, 0.02)
        data.append([roi, risk])
        true_regimes.append('bull_market')
    
    # Bear market (periods 50-100)
    for i in range(50):
        roi = -0.01 + np.random.normal(0, 0.02)
        risk = 0.20 + np.random.normal(0, 0.03)
        data.append([roi, risk])
        true_regimes.append('bear_market')
    
    # Sideways market (periods 100-200)
    for i in range(100):
        roi = 0.005 + np.random.normal(0, 0.015)
        risk = 0.15 + np.random.normal(0, 0.025)
        data.append([roi, risk])
        true_regimes.append('sideways_market')
    
    data = np.array(data)
    
    # Initialize regime detection system
    regime_system = RegimeDetectionSystem(window_size=10, confidence=0.95)
    
    # Process data
    detected_regimes = []
    confidences = []
    
    for i in range(10, len(data)):
        window_data = data[:i+1]
        result = regime_system.process_data(window_data)
        detected_regimes.append(result.regime)
        confidences.append(result.confidence)
        
        # Simulate reward based on true regime
        true_regime = true_regimes[i]
        if result.regime == true_regime:
            reward = 1.0  # Correct detection
        else:
            reward = 0.0  # Incorrect detection
        
        regime_system.update_with_reward(reward)
    
    # Print results
    print("Contextual Bandits Regime Detection Results:")
    print("=" * 50)
    
    # Accuracy
    accuracy = sum(1 for i, detected in enumerate(detected_regimes) 
                   if detected == true_regimes[i+10]) / len(detected_regimes)
    print(f"Accuracy: {accuracy:.2%}")
    
    # System status
    status = regime_system.get_system_status()
    print(f"\nSystem Status:")
    print(f"Current Regime: {status['regime_detection']['current_regime']}")
    print(f"Confidence: {status['regime_detection']['confidence']:.2%}")
    print(f"Regret Bound: {status['regime_detection']['regret_bound']:.4f}")
    
    # Bandit performance
    bandit_perf = status['bandit_performance']
    print(f"\nBandit Performance:")
    print(f"Total Pulls: {bandit_perf['total_pulls']}")
    print(f"Regime Distribution: {bandit_perf['regime_distribution']}")
    print(f"Average Rewards: {bandit_perf['average_rewards']}")
    print(f"Exploration Rate: {bandit_perf['exploration_rate']:.2%}")

if __name__ == "__main__":
    main()
