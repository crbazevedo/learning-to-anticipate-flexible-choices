"""
Sliding Window Dirichlet Model Implementation

Implements Equations 6.24-6.27 from the thesis for concentration parameter updates
in the Dirichlet Dynamical Model for decision space learning.

Based on C++ implementation in anticipatory-learning-asmoo/source/dirichlet.cpp
"""

import numpy as np
from typing import List, Optional
from scipy.stats import dirichlet
import logging

logger = logging.getLogger(__name__)


class SlidingWindowDirichlet:
    """
    Sliding Window Dirichlet Model for decision space tracking.
    
    Implements the theoretical framework from Equations 6.24-6.27:
    - Equation 6.24: α_t^(i) = α_{t-1}^(i) + s u_{t-1}^(i) if t < K
    - Equation 6.25: α_t^(i) = α_{t-1}^(i) + s u_{t-1}^(i) - α_0^(i) if t = K  
    - Equation 6.26: α_t^(i) = α_{t-1}^(i) + s u_{t-1}^(i) - s u_{t-K-1}^(i) if t > K
    """
    
    def __init__(self, window_size_K: int, concentration_scaling_s: float = 1.0):
        """
        Initialize sliding window Dirichlet model.
        
        Args:
            window_size_K: Size of sliding window (K parameter)
            concentration_scaling_s: Concentration scaling factor (s parameter)
        """
        self.K = window_size_K
        self.s = concentration_scaling_s
        self.alpha_history: List[np.ndarray] = []
        self.alpha_0: Optional[np.ndarray] = None
        self.u_history: List[np.ndarray] = []  # Store historical decision vectors
        
    def update_concentration(self, t: int, u_t_minus_1: np.ndarray) -> np.ndarray:
        """
        Update concentration parameters according to Equations 6.24-6.27.
        
        Args:
            t: Current time step
            u_t_minus_1: Decision vector at time t-1
            
        Returns:
            Updated concentration parameter vector α_t
        """
        # Ensure u_t_minus_1 is normalized
        u_t_minus_1 = u_t_minus_1 / np.sum(u_t_minus_1) if np.sum(u_t_minus_1) > 0 else u_t_minus_1
        
        if t == 0:
            # Initialize with even-handed concentration
            alpha_t = self.s * np.ones_like(u_t_minus_1) / len(u_t_minus_1)
            self.alpha_0 = alpha_t.copy()
            logger.debug(f"Initialized α_0: {alpha_t}")
        elif t < self.K:
            # Equation 6.24: Accumulating observations
            alpha_t = self.alpha_history[-1] + self.s * u_t_minus_1
            logger.debug(f"Equation 6.24 (t={t}): α_t = α_{t-1} + s*u_{t-1}")
        elif t == self.K:
            # Equation 6.25: First time window is full
            alpha_t = self.alpha_history[-1] + self.s * u_t_minus_1 - self.alpha_0
            logger.debug(f"Equation 6.25 (t={t}): α_t = α_{t-1} + s*u_{t-1} - α_0")
        else:
            # Equation 6.26: Sliding window
            # Remove the oldest element (u_{t-K-1})
            u_oldest = self.u_history[0]  # u_{t-K-1}
            alpha_t = (self.alpha_history[-1] + self.s * u_t_minus_1 - 
                      self.s * u_oldest)
            logger.debug(f"Equation 6.26 (t={t}): sliding window update")
        
        # Ensure concentration parameters are positive
        alpha_t = np.maximum(alpha_t, 1e-10)
        
        # Store history
        self.alpha_history.append(alpha_t.copy())
        self.u_history.append(u_t_minus_1.copy())
        
        # Keep only necessary history for sliding window
        if len(self.u_history) > self.K + 1:
            self.u_history.pop(0)
            
        return alpha_t
    
    def calculate_velocity(self, t: int) -> np.ndarray:
        """
        Calculate velocity for prediction (Equation 6.28).
        
        Args:
            t: Current time step
            
        Returns:
            Velocity vector for prediction
        """
        if len(self.alpha_history) < 2:
            return np.zeros_like(self.alpha_history[0]) if self.alpha_history else np.array([])
        
        # Calculate velocity as difference in concentration parameters
        velocity = self.alpha_history[-1] - self.alpha_history[-2]
        return velocity
    
    def predict_future_concentration(self, t: int, horizon: int) -> np.ndarray:
        """
        Predict future concentration parameters using velocity.
        
        Args:
            t: Current time step
            horizon: Prediction horizon
            
        Returns:
            Predicted concentration parameter vector
        """
        if not self.alpha_history:
            raise ValueError("No concentration history available for prediction")
        
        current_alpha = self.alpha_history[-1]
        velocity = self.calculate_velocity(t)
        
        # Simple linear prediction: α_{t+h} = α_t + h * velocity
        predicted_alpha = current_alpha + horizon * velocity
        
        # Ensure positive values
        predicted_alpha = np.maximum(predicted_alpha, 1e-10)
        
        return predicted_alpha
    
    def dirichlet_mean(self, alpha: np.ndarray) -> np.ndarray:
        """
        Calculate Dirichlet mean from concentration parameters.
        
        Args:
            alpha: Concentration parameter vector
            
        Returns:
            Mean vector of Dirichlet distribution
        """
        return alpha / np.sum(alpha)
    
    def dirichlet_variance(self, alpha: np.ndarray) -> np.ndarray:
        """
        Calculate Dirichlet variance from concentration parameters.
        
        Args:
            alpha: Concentration parameter vector
            
        Returns:
            Variance vector of Dirichlet distribution
        """
        alpha_sum = np.sum(alpha)
        factor = alpha_sum * alpha_sum * alpha_sum + alpha_sum * alpha_sum
        alpha_square = alpha * alpha
        variance = (alpha_sum * alpha - alpha_square) / factor
        return variance
    
    def dirichlet_variance_from_proportions(self, proportions: np.ndarray, concentration: float) -> np.ndarray:
        """
        Calculate Dirichlet variance from proportions and concentration.
        
        Args:
            proportions: Proportion vector
            concentration: Concentration parameter
            
        Returns:
            Variance vector
        """
        alpha = concentration * proportions
        return self.dirichlet_variance(alpha)
    
    def dirichlet_mean_map_estimate(self, alpha: np.ndarray) -> np.ndarray:
        """
        Calculate MAP estimate of Dirichlet mean.
        
        Args:
            alpha: Concentration parameter vector
            
        Returns:
            MAP estimate of mean
        """
        return (alpha - 1.0) / (np.sum(alpha) - len(alpha))
    
    def sample_from_dirichlet(self, alpha: np.ndarray, size: int = 1) -> np.ndarray:
        """
        Sample from Dirichlet distribution.
        
        Args:
            alpha: Concentration parameter vector
            size: Number of samples
            
        Returns:
            Samples from Dirichlet distribution
        """
        return dirichlet.rvs(alpha, size=size)
    
    def get_current_concentration(self) -> Optional[np.ndarray]:
        """Get current concentration parameters."""
        return self.alpha_history[-1] if self.alpha_history else None
    
    def get_window_size(self) -> int:
        """Get window size K."""
        return self.K
    
    def get_scaling_factor(self) -> float:
        """Get concentration scaling factor s."""
        return self.s
    
    def reset(self):
        """Reset the model to initial state."""
        self.alpha_history.clear()
        self.u_history.clear()
        self.alpha_0 = None
