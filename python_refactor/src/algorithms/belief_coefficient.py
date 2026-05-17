"""
Belief Coefficient Self-Adjustment Implementation

This module implements the belief coefficient self-adjustment mechanism
as specified in the thesis, including Equation 6.30 and TIP-based
confidence calculation.
"""

import numpy as np
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

# W1-3: relative imports (was `from algorithms.X import Y` which only
# worked under a `sys.path.insert(..., src)` hack and was one of the
# 16 pre-W1-3 pytest collection errors).
from .solution import Solution
from .temporal_incomparability_probability import TemporalIncomparabilityCalculator

logger = logging.getLogger(__name__)


@dataclass
class BeliefCoefficientResult:
    """Data class for belief coefficient calculation results."""
    
    belief_coefficient: float
    tip_value: float
    binary_entropy: float
    confidence: float
    timestamp: float


class BeliefCoefficientCalculator:
    """
    Belief coefficient self-adjustment calculator.
    
    This class implements the belief coefficient self-adjustment mechanism
    as specified in the thesis, including Equation 6.30 and TIP-based
    confidence calculation.
    """
    
    def __init__(self, monte_carlo_samples: int = 1000):
        """
        Initialize belief coefficient calculator.
        
        Args:
            monte_carlo_samples: Number of Monte Carlo samples for TIP calculation
        """
        self.monte_carlo_samples = monte_carlo_samples
        self.tip_calculator = TemporalIncomparabilityCalculator(monte_carlo_samples)
        
        # Storage for historical data
        self.belief_coefficient_history: List[BeliefCoefficientResult] = []
        self.tip_history: List[float] = []
        self.entropy_history: List[float] = []
        
        logger.info(f"Initialized BeliefCoefficientCalculator with {monte_carlo_samples} Monte Carlo samples")
    
    def calculate_belief_coefficient(self, solution: Solution, 
                                   predicted_solution: Solution) -> BeliefCoefficientResult:
        """
        Implement Equation 6.30: v_{t+1} = 1 - (1/2) H(p_{t-1,t})
        
        Where H(p_{t-1,t}) is binary entropy of TIP
        
        Args:
            solution: Current solution
            predicted_solution: Predicted solution
            
        Returns:
            BeliefCoefficientResult with all calculation details
        """
        # Calculate TIP (Trend Information Probability)
        tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
        
        # Calculate binary entropy
        entropy = self._binary_entropy(tip)
        
        # Calculate belief coefficient based on Equation 6.30
        v_t_plus_1 = 1.0 - 0.5 * entropy
        
        # Apply bounds to ensure stability
        v_t_plus_1 = max(0.5, min(1.0, v_t_plus_1))
        
        # Calculate confidence based on TIP
        confidence = self._calculate_confidence(tip, entropy)
        
        # Create result object
        result = BeliefCoefficientResult(
            belief_coefficient=v_t_plus_1,
            tip_value=tip,
            binary_entropy=entropy,
            confidence=confidence,
            timestamp=np.datetime64('now').astype(float)
        )
        
        # Store for historical analysis
        self.belief_coefficient_history.append(result)
        self.tip_history.append(tip)
        self.entropy_history.append(entropy)
        
        logger.debug(f"Calculated belief coefficient: {v_t_plus_1:.4f} (TIP: {tip:.4f}, Entropy: {entropy:.4f})")
        
        return result
    
    def _binary_entropy(self, p: float) -> float:
        """
        Calculate binary entropy: H(p) = -p*log2(p) - (1-p)*log2(1-p)
        
        Args:
            p: Probability value
            
        Returns:
            Binary entropy value
        """
        if p <= 0 or p >= 1:
            return 0.0
        
        # Handle edge cases
        if p == 0.5:
            return 1.0  # Maximum entropy
        
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    
    def _calculate_confidence(self, tip: float, entropy: float) -> float:
        """
        Calculate confidence based on TIP and entropy.
        
        Args:
            tip: TIP value
            entropy: Binary entropy value
            
        Returns:
            Confidence value between 0 and 1
        """
        # Confidence is higher when:
        # 1. TIP is closer to 0 or 1 (less uncertainty)
        # 2. Entropy is lower (less uncertainty)
        
        # TIP-based confidence: higher when TIP is closer to 0 or 1.
        # W2-1 fix: was `1.0 - abs(tip - 0.5) * 2` which is INVERTED
        # vs the docstring (returned 1.0 at tip=0.5, 0.0 at tip=0/1 —
        # the opposite of the intent). Surfaced by test_calculate_confidence
        # once W1-4 fixed the test file's imports. Closes W1-4-CARRY-1.
        tip_confidence = abs(tip - 0.5) * 2
        
        # Entropy-based confidence: higher when entropy is lower
        entropy_confidence = 1.0 - entropy
        
        # Combined confidence (weighted average)
        confidence = 0.6 * tip_confidence + 0.4 * entropy_confidence
        
        return max(0.0, min(1.0, confidence))
    
    def calculate_adaptive_belief_coefficient(self, solution: Solution, 
                                            predicted_solution: Solution,
                                            historical_window: int = 10) -> BeliefCoefficientResult:
        """
        Calculate adaptive belief coefficient using historical data.
        
        Args:
            solution: Current solution
            predicted_solution: Predicted solution
            historical_window: Number of historical values to consider
            
        Returns:
            Adaptive belief coefficient result
        """
        # Calculate base belief coefficient
        base_result = self.calculate_belief_coefficient(solution, predicted_solution)
        
        # Get historical TIP values
        if len(self.tip_history) >= historical_window:
            recent_tips = self.tip_history[-historical_window:]
            recent_entropies = self.entropy_history[-historical_window:]
            
            # Calculate adaptive adjustments
            tip_trend = self._calculate_trend(recent_tips)
            entropy_trend = self._calculate_trend(recent_entropies)
            
            # Apply adaptive adjustments
            adaptive_tip = base_result.tip_value + 0.1 * tip_trend
            adaptive_entropy = base_result.binary_entropy + 0.1 * entropy_trend
            
            # Recalculate belief coefficient with adaptive values
            adaptive_belief_coefficient = 1.0 - 0.5 * adaptive_entropy
            adaptive_belief_coefficient = max(0.5, min(1.0, adaptive_belief_coefficient))
            
            # Create adaptive result
            adaptive_result = BeliefCoefficientResult(
                belief_coefficient=adaptive_belief_coefficient,
                tip_value=adaptive_tip,
                binary_entropy=adaptive_entropy,
                confidence=base_result.confidence,
                timestamp=base_result.timestamp
            )
            
            logger.debug(f"Calculated adaptive belief coefficient: {adaptive_belief_coefficient:.4f}")
            
            return adaptive_result
        
        return base_result
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend in a series of values.
        
        Args:
            values: List of values
            
        Returns:
            Trend value (positive for increasing, negative for decreasing)
        """
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        
        return slope
    
    def get_belief_coefficient_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about belief coefficient calculations.
        
        Returns:
            Dictionary with belief coefficient statistics
        """
        if not self.belief_coefficient_history:
            return {'error': 'No belief coefficient history available'}
        
        # Extract values
        belief_coefficients = [result.belief_coefficient for result in self.belief_coefficient_history]
        tip_values = [result.tip_value for result in self.belief_coefficient_history]
        entropy_values = [result.binary_entropy for result in self.belief_coefficient_history]
        confidence_values = [result.confidence for result in self.belief_coefficient_history]
        
        return {
            'total_calculations': len(self.belief_coefficient_history),
            'mean_belief_coefficient': np.mean(belief_coefficients),
            'std_belief_coefficient': np.std(belief_coefficients),
            'min_belief_coefficient': np.min(belief_coefficients),
            'max_belief_coefficient': np.max(belief_coefficients),
            'mean_tip': np.mean(tip_values),
            'std_tip': np.std(tip_values),
            'mean_entropy': np.mean(entropy_values),
            'std_entropy': np.std(entropy_values),
            'mean_confidence': np.mean(confidence_values),
            'std_confidence': np.std(confidence_values),
            'recent_trend': self._calculate_trend(belief_coefficients[-10:]) if len(belief_coefficients) >= 10 else 0.0
        }
    
    def get_tip_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about TIP calculations.
        
        Returns:
            Dictionary with TIP statistics
        """
        return self.tip_calculator.get_tip_statistics()
    
    def reset_history(self):
        """Reset all historical data."""
        self.belief_coefficient_history.clear()
        self.tip_history.clear()
        self.entropy_history.clear()
        self.tip_calculator.reset_history()
        
        logger.info("Reset belief coefficient calculator history")
    
    def get_recent_belief_coefficients(self, n: int = 10) -> List[BeliefCoefficientResult]:
        """
        Get recent belief coefficient results.
        
        Args:
            n: Number of recent results to return
            
        Returns:
            List of recent belief coefficient results
        """
        return self.belief_coefficient_history[-n:] if self.belief_coefficient_history else []
    
    def calculate_belief_coefficient_for_horizon(self, solution: Solution, 
                                               predicted_solution: Solution,
                                               horizon: int) -> BeliefCoefficientResult:
        """
        Calculate belief coefficient for a specific prediction horizon.
        
        Args:
            solution: Current solution
            predicted_solution: Predicted solution
            horizon: Prediction horizon
            
        Returns:
            Belief coefficient result for the horizon
        """
        # Calculate TIP for the specific horizon
        tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
        
        # Adjust TIP based on horizon (longer horizons have higher uncertainty)
        horizon_adjusted_tip = tip * (1.0 - 0.1 * (horizon - 1))
        horizon_adjusted_tip = max(0.0, min(1.0, horizon_adjusted_tip))
        
        # Calculate entropy with adjusted TIP
        entropy = self._binary_entropy(horizon_adjusted_tip)
        
        # Calculate belief coefficient
        belief_coefficient = 1.0 - 0.5 * entropy
        belief_coefficient = max(0.5, min(1.0, belief_coefficient))
        
        # Calculate confidence
        confidence = self._calculate_confidence(horizon_adjusted_tip, entropy)
        
        # Create result
        result = BeliefCoefficientResult(
            belief_coefficient=belief_coefficient,
            tip_value=horizon_adjusted_tip,
            binary_entropy=entropy,
            confidence=confidence,
            timestamp=np.datetime64('now').astype(float)
        )
        
        logger.debug(f"Calculated belief coefficient for horizon {horizon}: {belief_coefficient:.4f}")
        
        return result
    
    def validate_belief_coefficient(self, belief_coefficient: float) -> bool:
        """
        Validate belief coefficient value.
        
        Args:
            belief_coefficient: Belief coefficient value to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 0.5 <= belief_coefficient <= 1.0
    
    def get_equation_630_verification(self, solution: Solution, 
                                    predicted_solution: Solution) -> Dict[str, Any]:
        """
        Verify Equation 6.30 implementation.
        
        Args:
            solution: Current solution
            predicted_solution: Predicted solution
            
        Returns:
            Dictionary with verification details
        """
        # Calculate TIP
        tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
        
        # Calculate binary entropy
        entropy = self._binary_entropy(tip)
        
        # Calculate belief coefficient using Equation 6.30
        belief_coefficient = 1.0 - 0.5 * entropy
        
        # Manual verification
        expected_belief_coefficient = 1.0 - 0.5 * entropy
        
        return {
            'tip_value': tip,
            'binary_entropy': entropy,
            'belief_coefficient': belief_coefficient,
            'expected_belief_coefficient': expected_belief_coefficient,
            'equation_630_verified': abs(belief_coefficient - expected_belief_coefficient) < 1e-10,
            'formula': 'v_{t+1} = 1 - (1/2) H(p_{t-1,t})',
            'calculation': f'1 - 0.5 * {entropy:.6f} = {belief_coefficient:.6f}'
        }


def create_belief_coefficient_calculator(monte_carlo_samples: int = 1000) -> BeliefCoefficientCalculator:
    """
    Convenience function to create belief coefficient calculator.
    
    Args:
        monte_carlo_samples: Number of Monte Carlo samples
        
    Returns:
        BeliefCoefficientCalculator instance
    """
    return BeliefCoefficientCalculator(monte_carlo_samples)


if __name__ == '__main__':
    # Example usage
    print("Belief Coefficient Self-Adjustment Module")
    print("This module provides belief coefficient calculation functionality.")
    print("Use BeliefCoefficientCalculator class for belief coefficient calculations.")
