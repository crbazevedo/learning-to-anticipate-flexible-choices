"""
Temporal Incomparability Probability (TIP) Implementation

Implements Definition 6.1 from the thesis:
P_{t,t+h} = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]

This module provides the core TIP calculation functionality that integrates
with the main anticipatory learning algorithm.
"""

import numpy as np
from typing import Tuple, Optional
import logging

# ---------------------------------------------------------------------------
# W3-4 (closes W1-4-CARRY-2): heuristic placeholders for the
# `_calculate_tip_simple` fallback path (used when KF covariance is
# unavailable). These constants are NOT paper-grounded — they were chosen
# during development to span the typical FTSE-100 ROI/risk range. Replace
# with empirically-fitted values when a calibration unit lands.
# ---------------------------------------------------------------------------
_TIP_FALLBACK_MAX_ROI_DIFF: float = 0.5
_TIP_FALLBACK_MAX_RISK_DIFF: float = 0.3

logger = logging.getLogger(__name__)


class TemporalIncomparabilityCalculator:
    """
    Calculate Temporal Incomparability Probability (TIP) according to Definition 6.1.
    
    TIP measures the probability that current and future predicted objective vectors
    are mutually non-dominated, which is core to anticipatory learning rate calculation.
    """
    
    def __init__(self, monte_carlo_samples: int = 1000,
                 clamp_range: Optional[Tuple[float, float]] = (0.05, 0.95)):
        """
        Initialize TIP calculator.

        Args:
            monte_carlo_samples: Number of Monte Carlo samples for probability estimation
            clamp_range: Optional (min, max) tuple bounding the returned TIP.
                Default `(0.05, 0.95)` preserves pre-W1-4 behaviour (and is
                what every live caller — anticipatory_learning_obj_space,
                MultiHorizonAnticipatoryLearning, TIPIntegratedAnticipatoryLearning
                — implicitly relies on for stability). Pass `None` to
                DISABLE the clamp entirely; this is the path that
                equation-level regression tests (paper Eq 12) use so they
                can observe TIP near 0 on disjoint Gaussians (the audit's
                "hides degenerate dominance" concern). Live callers
                should keep the default.
        """
        self.monte_carlo_samples = monte_carlo_samples
        self.clamp_range = clamp_range
        self.historical_tips = []

    def _clamp_tip(self, tip: float) -> float:
        """Apply the constructor-set TIP bound (or no-op when disabled).

        W1-4: extracted from 3 duplicated `max(0.05, min(0.95, tip))`
        sites in the calculation methods. Single point of truth, opt-out
        controlled by `self.clamp_range`.
        """
        if self.clamp_range is None:
            return tip
        return max(self.clamp_range[0], min(self.clamp_range[1], tip))
        
    def calculate_tip(self, current_solution, predicted_solution, 
                     prediction_uncertainty: Optional[float] = None) -> float:
        """
        Calculate Temporal Incomparability Probability (Definition 6.1).
        
        P_{t,t+h} = Pr[ẑ_t || ẑ_{t+h} | ẑ_t]
        
        Args:
            current_solution: Current solution with ROI and risk
            predicted_solution: Predicted solution with ROI and risk
            prediction_uncertainty: Optional uncertainty factor for prediction
            
        Returns:
            TIP value between 0 and 1
        """
        # Get current and predicted objectives
        current_roi, current_risk = current_solution.P.ROI, current_solution.P.risk
        predicted_roi, predicted_risk = predicted_solution.P.ROI, predicted_solution.P.risk
        
        # Use Kalman filter covariance if available
        if hasattr(current_solution.P, 'kalman_state') and current_solution.P.kalman_state is not None:
            current_cov = current_solution.P.kalman_state.P[:2, :2]
            predicted_cov = predicted_solution.P.kalman_state.P[:2, :2]
            
            tip = self._calculate_tip_with_covariance(
                current_roi, current_risk, current_cov,
                predicted_roi, predicted_risk, predicted_cov
            )
        else:
            # Fallback to Monte Carlo sampling with default uncertainties
            tip = self._calculate_tip_monte_carlo(
                current_roi, current_risk, predicted_roi, predicted_risk,
                prediction_uncertainty
            )
        
        # Store for historical analysis
        self.historical_tips.append(tip)
        
        return tip
    
    def _calculate_tip_with_covariance(self, current_roi: float, current_risk: float,
                                     current_cov: np.ndarray,
                                     predicted_roi: float, predicted_risk: float,
                                     predicted_cov: np.ndarray) -> float:
        """
        Calculate TIP using proper covariance matrices from Kalman filter.
        
        This is the most accurate method as it uses the actual uncertainty
        estimates from the Kalman filter state.
        """
        try:
            # Monte Carlo sampling with proper covariance
            mutual_non_dominance = 0
            
            for _ in range(self.monte_carlo_samples):
                # Sample from current distribution
                current_sample = np.random.multivariate_normal(
                    [current_roi, current_risk], current_cov
                )
                c_roi, c_risk = current_sample
                
                # Sample from predicted distribution
                predicted_sample = np.random.multivariate_normal(
                    [predicted_roi, predicted_risk], predicted_cov
                )
                p_roi, p_risk = predicted_sample
                
                # Check dominance relationships
                current_dominates = (c_roi > p_roi) and (c_risk < p_risk)
                predicted_dominates = (p_roi > c_roi) and (p_risk < c_risk)
                
                # Count mutual non-dominance
                if not current_dominates and not predicted_dominates:
                    mutual_non_dominance += 1
            
            tip = mutual_non_dominance / self.monte_carlo_samples
            
        except np.linalg.LinAlgError:
            # Fallback to simple method if covariance is not positive definite
            logger.warning("Covariance matrix not positive definite, using fallback TIP calculation")
            tip = self._calculate_tip_simple(current_roi, current_risk, predicted_roi, predicted_risk)
        
        return self._clamp_tip(tip)
    
    def _calculate_tip_monte_carlo(self, current_roi: float, current_risk: float,
                                 predicted_roi: float, predicted_risk: float,
                                 prediction_uncertainty: Optional[float] = None) -> float:
        """
        Calculate TIP using Monte Carlo sampling with default uncertainties.
        
        This method is used when Kalman filter covariance is not available.
        """
        # Default uncertainties
        current_roi_std = 0.01  # Low uncertainty for current
        current_risk_std = 0.005
        
        if prediction_uncertainty is not None:
            predicted_roi_std = max(0.02, prediction_uncertainty)
            predicted_risk_std = max(0.01, prediction_uncertainty * 0.5)
        else:
            predicted_roi_std = 0.02  # Higher uncertainty for prediction
            predicted_risk_std = 0.01
        
        # Monte Carlo sampling
        mutual_non_dominance = 0
        
        for _ in range(self.monte_carlo_samples):
            # Sample from current distribution
            c_roi = np.random.normal(current_roi, current_roi_std)
            c_risk = np.random.normal(current_risk, current_risk_std)
            
            # Sample from predicted distribution
            p_roi = np.random.normal(predicted_roi, predicted_roi_std)
            p_risk = np.random.normal(predicted_risk, predicted_risk_std)
            
            # Check dominance relationships
            current_dominates = (c_roi > p_roi) and (c_risk < p_risk)
            predicted_dominates = (p_roi > c_roi) and (p_risk < c_risk)
            
            # Count mutual non-dominance
            if not current_dominates and not predicted_dominates:
                mutual_non_dominance += 1
        
        tip = mutual_non_dominance / self.monte_carlo_samples
        return self._clamp_tip(tip)
    
    def _calculate_tip_simple(self, current_roi: float, current_risk: float,
                            predicted_roi: float, predicted_risk: float) -> float:
        """
        Simple TIP calculation based on objective similarity.
        
        This is a fallback method when more sophisticated calculations fail.
        """
        # Calculate dominance relationships
        current_dominates_predicted = (current_roi > predicted_roi) and (current_risk < predicted_risk)
        predicted_dominates_current = (predicted_roi > current_roi) and (predicted_risk < current_risk)
        
        # If neither dominates the other, they are mutually non-dominated
        if not current_dominates_predicted and not predicted_dominates_current:
            # Calculate similarity-based probability
            roi_distance = abs(current_roi - predicted_roi)
            risk_distance = abs(current_risk - predicted_risk)
            
            # W3-4: named module constants (see top of file) replace the
            # pre-W3-4 inline magic numbers. Flagged as heuristic
            # placeholders; not paper-grounded.
            normalized_roi_distance = min(
                roi_distance / _TIP_FALLBACK_MAX_ROI_DIFF, 1.0,
            )
            normalized_risk_distance = min(
                risk_distance / _TIP_FALLBACK_MAX_RISK_DIFF, 1.0,
            )
            
            # TIP is higher when objectives are more similar (closer)
            tip = 0.5 * (1.0 - normalized_roi_distance + 1.0 - normalized_risk_distance)
        else:
            # One dominates the other, so TIP is lower
            tip = 0.1
        
        return self._clamp_tip(tip)
    
    def binary_entropy(self, p: float) -> float:
        """
        Calculate binary entropy function: H(p) = -p*log2(p) - (1-p)*log2(1-p)
        
        This is used in Equation 6.6 for anticipatory learning rate calculation.
        
        Args:
            p: Probability value (should be between 0 and 1)
            
        Returns:
            Binary entropy value
        """
        if p <= 0 or p >= 1:
            return 0.0
        
        # Use natural log and convert to base 2
        return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2)
    
    def calculate_anticipatory_learning_rate_tip(self, tip: float, horizon: int) -> float:
        """
        Calculate anticipatory learning rate using TIP (Equation 6.6).
        
        λ_{t+h} = (1/(H-1)) * [1 - H(p_{t,t+h})]
        
        Args:
            tip: Temporal incomparability probability
            horizon: Prediction horizon H
            
        Returns:
            Anticipatory learning rate
        """
        if horizon <= 1:
            return 0.0
        
        entropy = self.binary_entropy(tip)
        learning_rate = (1.0 / (horizon - 1)) * (1.0 - entropy)
        
        # Ensure learning rate is in valid range
        return max(0.0, min(1.0, learning_rate))
    
    def get_historical_tip_trend(self) -> float:
        """
        Get trend in historical TIP values.
        
        Returns:
            Trend value (positive = increasing TIP, negative = decreasing TIP)
        """
        if len(self.historical_tips) < 5:
            return 0.0
        
        recent_tip = np.mean(self.historical_tips[-5:])
        older_tip = np.mean(self.historical_tips[-10:-5]) if len(self.historical_tips) >= 10 else 0.5
        
        return recent_tip - older_tip
    
    def get_average_tip(self, window_size: int = 10) -> float:
        """
        Get average TIP over recent window.
        
        Args:
            window_size: Number of recent TIP values to average
            
        Returns:
            Average TIP value
        """
        if not self.historical_tips:
            return 0.5
        
        recent_tips = self.historical_tips[-window_size:]
        return np.mean(recent_tips)
    
    def reset_history(self):
        """Reset historical TIP values."""
        self.historical_tips.clear()
    
    def get_tip_statistics(self) -> dict:
        """
        Get statistics about historical TIP values.
        
        Returns:
            Dictionary with TIP statistics
        """
        if not self.historical_tips:
            return {
                'count': 0,
                'mean': 0.5,
                'std': 0.0,
                'min': 0.5,
                'max': 0.5,
                'trend': 0.0
            }
        
        tips_array = np.array(self.historical_tips)
        
        return {
            'count': len(self.historical_tips),
            'mean': np.mean(tips_array),
            'std': np.std(tips_array),
            'min': np.min(tips_array),
            'max': np.max(tips_array),
            'trend': self.get_historical_tip_trend()
        }
