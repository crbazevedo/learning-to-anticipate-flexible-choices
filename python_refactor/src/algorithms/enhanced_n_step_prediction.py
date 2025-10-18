"""
Enhanced N-Step Prediction Integration

This module implements the enhanced N-step prediction with proper
anticipatory learning integration, belief coefficient usage, and
conditional expected hypervolume calculation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass

from algorithms.n_step_prediction import NStepPredictor
from algorithms.belief_coefficient import BeliefCoefficientCalculator
from algorithms.anticipatory_learning import AnticipatoryLearning
from algorithms.solution import Solution

logger = logging.getLogger(__name__)


@dataclass
class EnhancedPredictionResult:
    """Data class for enhanced prediction results."""
    
    conditional_hypervolumes: Dict[str, Any]
    belief_coefficients: Dict[str, float]
    anticipatory_adjustments: Dict[str, float]
    prediction_confidence: float
    horizon: int
    timestamp: float


class EnhancedNStepPredictor(NStepPredictor):
    """
    Enhanced N-Step Predictor with anticipatory learning integration.
    
    This class extends the base NStepPredictor with proper integration
    of anticipatory learning, belief coefficient calculation, and
    enhanced conditional expected hypervolume computation.
    """
    
    def __init__(self, max_horizon: int = 3):
        """
        Initialize enhanced N-step predictor.
        
        Args:
            max_horizon: Maximum prediction horizon
        """
        super().__init__(max_horizon)
        self.anticipatory_learning = None
        self.belief_calculator = BeliefCoefficientCalculator()
        
        # Storage for enhanced predictions
        self.enhanced_prediction_history: List[EnhancedPredictionResult] = []
        
        logger.info(f"Initialized EnhancedNStepPredictor with max_horizon={max_horizon}")
    
    def set_anticipatory_learning(self, anticipatory_learning: AnticipatoryLearning):
        """
        Set reference to anticipatory learning system.
        
        Args:
            anticipatory_learning: Anticipatory learning instance
        """
        self.anticipatory_learning = anticipatory_learning
        logger.info("Set anticipatory learning reference")
    
    def compute_conditional_expected_hypervolume(self, pareto_frontier: List[Solution],
                                               selected_solution: int,
                                               kalman_predictions: Dict,
                                               dirichlet_predictions: Dict,
                                               h: int) -> Dict:
        """
        Enhanced version with proper anticipatory learning integration.
        
        Based on Pseudocode 7: Anticipatory Distribution Estimation
        
        Args:
            pareto_frontier: Current Pareto frontier solutions
            selected_solution: Index of selected solution
            kalman_predictions: N-step Kalman predictions
            dirichlet_predictions: N-step Dirichlet predictions
            h: Prediction horizon
            
        Returns:
            Dict with conditional expected hypervolume distributions
        """
        if selected_solution >= len(pareto_frontier):
            raise ValueError(f"Invalid solution index {selected_solution}")
        
        # Get the selected solution
        selected = pareto_frontier[selected_solution]
        
        # Get predictions for horizon h
        kalman_pred = kalman_predictions[f'step_{h}']
        dirichlet_pred = dirichlet_predictions[f'step_{h}']
        
        # Calculate belief coefficient for this solution
        belief_coeff = self.belief_calculator.calculate_belief_coefficient(
            selected, selected  # Using same solution for current and predicted
        )
        
        # Compute conditional expectations with belief coefficient
        conditional_hypervolumes = {}
        belief_coefficients = {}
        anticipatory_adjustments = {}
        
        for i, solution in enumerate(pareto_frontier):
            if i == selected_solution:
                # For selected solution, use full expected hypervolume with belief coefficient
                conditional_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h
                ) * belief_coeff.belief_coefficient
                
                # Apply anticipatory learning adjustment if available
                if self.anticipatory_learning:
                    anticipatory_adjustment = self._get_anticipatory_adjustment(solution, h)
                    conditional_hv *= anticipatory_adjustment
                    anticipatory_adjustments[f'solution_{i}'] = anticipatory_adjustment
                else:
                    anticipatory_adjustments[f'solution_{i}'] = 1.0
                    
            else:
                # For other solutions, adjust based on selection
                base_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h
                )
                
                # Reduce hypervolume due to selection of another solution
                reduction_factor = 0.8 * belief_coeff.belief_coefficient  # Belief coefficient affects reduction
                conditional_hv = base_hv * reduction_factor
                
                # Apply anticipatory learning adjustment if available
                if self.anticipatory_learning:
                    anticipatory_adjustment = self._get_anticipatory_adjustment(solution, h)
                    conditional_hv *= anticipatory_adjustment
                    anticipatory_adjustments[f'solution_{i}'] = anticipatory_adjustment
                else:
                    anticipatory_adjustments[f'solution_{i}'] = 1.0
            
            conditional_hypervolumes[f'solution_{i}'] = {
                'conditional_expected_hypervolume': conditional_hv,
                'is_selected': (i == selected_solution),
                'horizon': h,
                'belief_coefficient': belief_coeff.belief_coefficient,
                'tip_value': belief_coeff.tip_value,
                'confidence': belief_coeff.confidence
            }
            
            belief_coefficients[f'solution_{i}'] = belief_coeff.belief_coefficient
        
        # Calculate overall prediction confidence
        prediction_confidence = self._calculate_prediction_confidence(
            conditional_hypervolumes, belief_coefficients
        )
        
        # Create enhanced result
        enhanced_result = EnhancedPredictionResult(
            conditional_hypervolumes=conditional_hypervolumes,
            belief_coefficients=belief_coefficients,
            anticipatory_adjustments=anticipatory_adjustments,
            prediction_confidence=prediction_confidence,
            horizon=h,
            timestamp=np.datetime64('now').astype(float)
        )
        
        # Store for historical analysis
        self.enhanced_prediction_history.append(enhanced_result)
        
        logger.debug(f"Computed enhanced conditional expected hypervolume for horizon {h}")
        
        return conditional_hypervolumes
    
    def _get_anticipatory_adjustment(self, solution: Solution, horizon: int) -> float:
        """
        Get anticipatory learning adjustment for a solution.
        
        Args:
            solution: Solution to get adjustment for
            horizon: Prediction horizon
            
        Returns:
            Anticipatory adjustment factor
        """
        if not self.anticipatory_learning:
            return 1.0
        
        try:
            # Get anticipatory learning rate for this solution
            learning_rate = self.anticipatory_learning.compute_anticipatory_learning_rate(
                solution, horizon
            )
            
            # Convert learning rate to adjustment factor
            # Higher learning rate means more confidence in prediction
            adjustment = 1.0 + 0.1 * learning_rate
            
            # Apply bounds
            adjustment = max(0.8, min(1.2, adjustment))
            
            return adjustment
            
        except Exception as e:
            logger.warning(f"Failed to get anticipatory adjustment: {e}")
            return 1.0
    
    def _calculate_prediction_confidence(self, conditional_hypervolumes: Dict,
                                       belief_coefficients: Dict) -> float:
        """
        Calculate overall prediction confidence.
        
        Args:
            conditional_hypervolumes: Conditional hypervolume results
            belief_coefficients: Belief coefficient values
            
        Returns:
            Overall prediction confidence
        """
        if not conditional_hypervolumes:
            return 0.0
        
        # Calculate confidence based on belief coefficients
        belief_confidences = list(belief_coefficients.values())
        mean_belief_confidence = np.mean(belief_confidences)
        
        # Calculate confidence based on hypervolume consistency
        hypervolumes = [result['conditional_expected_hypervolume'] 
                       for result in conditional_hypervolumes.values()]
        hypervolume_consistency = 1.0 - (np.std(hypervolumes) / (np.mean(hypervolumes) + 1e-10))
        hypervolume_consistency = max(0.0, min(1.0, hypervolume_consistency))
        
        # Combined confidence
        overall_confidence = 0.6 * mean_belief_confidence + 0.4 * hypervolume_consistency
        
        return max(0.0, min(1.0, overall_confidence))
    
    def compute_enhanced_expected_future_hypervolume(self, pareto_frontier: List[Solution],
                                                   kalman_predictions: Dict,
                                                   dirichlet_predictions: Dict,
                                                   h: int) -> Dict:
        """
        Compute enhanced expected future hypervolume with anticipatory learning.
        
        Args:
            pareto_frontier: Current Pareto frontier solutions
            kalman_predictions: N-step Kalman predictions
            dirichlet_predictions: N-step Dirichlet predictions
            h: Prediction horizon
            
        Returns:
            Dict with enhanced expected hypervolume distributions
        """
        if f'step_{h}' not in kalman_predictions:
            raise ValueError(f"No prediction available for horizon {h}")
        
        enhanced_hypervolumes = {}
        
        for i, solution in enumerate(pareto_frontier):
            # Get predictions for horizon h
            kalman_pred = kalman_predictions[f'step_{h}']
            dirichlet_pred = dirichlet_predictions[f'step_{h}']
            
            # Compute expected future state
            expected_state = kalman_pred['state']
            expected_portfolio_weights = dirichlet_pred['mean_prediction']
            
            # Compute base expected hypervolume contribution
            base_expected_hv = self._compute_solution_expected_hypervolume(
                solution, expected_state, expected_portfolio_weights, h
            )
            
            # Apply belief coefficient adjustment
            belief_coeff = self.belief_calculator.calculate_belief_coefficient(
                solution, solution
            )
            belief_adjusted_hv = base_expected_hv * belief_coeff.belief_coefficient
            
            # Apply anticipatory learning adjustment
            anticipatory_adjustment = self._get_anticipatory_adjustment(solution, h)
            enhanced_hv = belief_adjusted_hv * anticipatory_adjustment
            
            enhanced_hypervolumes[f'solution_{i}'] = {
                'enhanced_expected_hypervolume': enhanced_hv,
                'base_expected_hypervolume': base_expected_hv,
                'belief_coefficient': belief_coeff.belief_coefficient,
                'anticipatory_adjustment': anticipatory_adjustment,
                'kalman_state': expected_state,
                'dirichlet_weights': expected_portfolio_weights,
                'horizon': h,
                'confidence': belief_coeff.confidence
            }
        
        return enhanced_hypervolumes
    
    def get_enhanced_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about enhanced predictions.
        
        Returns:
            Dictionary with enhanced prediction statistics
        """
        if not self.enhanced_prediction_history:
            return {'error': 'No enhanced prediction history available'}
        
        # Extract values
        confidences = [result.prediction_confidence for result in self.enhanced_prediction_history]
        horizons = [result.horizon for result in self.enhanced_prediction_history]
        
        # Calculate statistics
        stats = {
            'total_predictions': len(self.enhanced_prediction_history),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'mean_horizon': np.mean(horizons),
            'std_horizon': np.std(horizons),
            'horizon_distribution': {h: horizons.count(h) for h in set(horizons)}
        }
        
        # Add belief coefficient statistics
        belief_stats = self.belief_calculator.get_belief_coefficient_statistics()
        if 'error' not in belief_stats:
            stats['belief_coefficient_stats'] = belief_stats
        
        return stats
    
    def get_enhanced_prediction_for_horizon(self, horizon: int) -> Optional[EnhancedPredictionResult]:
        """
        Get enhanced prediction result for a specific horizon.
        
        Args:
            horizon: Prediction horizon
            
        Returns:
            Enhanced prediction result for the horizon, or None if not found
        """
        for result in reversed(self.enhanced_prediction_history):
            if result.horizon == horizon:
                return result
        return None
    
    def reset_enhanced_prediction_history(self):
        """Reset enhanced prediction history."""
        self.enhanced_prediction_history.clear()
        self.belief_calculator.reset_history()
        logger.info("Reset enhanced prediction history")
    
    def validate_enhanced_prediction(self, result: EnhancedPredictionResult) -> bool:
        """
        Validate enhanced prediction result.
        
        Args:
            result: Enhanced prediction result to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check basic structure
        if not isinstance(result, EnhancedPredictionResult):
            return False
        
        # Check confidence bounds
        if not (0.0 <= result.prediction_confidence <= 1.0):
            return False
        
        # Check horizon bounds
        if not (1 <= result.horizon <= self.max_horizon):
            return False
        
        # Check conditional hypervolumes
        if not isinstance(result.conditional_hypervolumes, dict):
            return False
        
        # Check belief coefficients
        if not isinstance(result.belief_coefficients, dict):
            return False
        
        # Validate belief coefficient values
        for bc in result.belief_coefficients.values():
            if not (0.5 <= bc <= 1.0):
                return False
        
        return True
    
    def get_enhanced_prediction_summary(self) -> Dict[str, Any]:
        """
        Get summary of enhanced predictions.
        
        Returns:
            Dictionary with enhanced prediction summary
        """
        if not self.enhanced_prediction_history:
            return {'error': 'No enhanced prediction history available'}
        
        # Get latest prediction
        latest = self.enhanced_prediction_history[-1]
        
        # Calculate summary statistics
        summary = {
            'latest_prediction': {
                'horizon': latest.horizon,
                'confidence': latest.prediction_confidence,
                'num_solutions': len(latest.conditional_hypervolumes),
                'timestamp': latest.timestamp
            },
            'total_predictions': len(self.enhanced_prediction_history),
            'horizon_coverage': list(set(r.horizon for r in self.enhanced_prediction_history)),
            'confidence_trend': self._calculate_confidence_trend(),
            'belief_coefficient_summary': self._get_belief_coefficient_summary()
        }
        
        return summary
    
    def _calculate_confidence_trend(self) -> float:
        """Calculate confidence trend over time."""
        if len(self.enhanced_prediction_history) < 2:
            return 0.0
        
        confidences = [result.prediction_confidence for result in self.enhanced_prediction_history]
        return self.belief_calculator._calculate_trend(confidences)
    
    def _get_belief_coefficient_summary(self) -> Dict[str, Any]:
        """Get belief coefficient summary."""
        stats = self.belief_calculator.get_belief_coefficient_statistics()
        if 'error' in stats:
            return {'error': 'No belief coefficient data available'}
        
        return {
            'mean_belief_coefficient': stats.get('mean_belief_coefficient', 0.0),
            'std_belief_coefficient': stats.get('std_belief_coefficient', 0.0),
            'total_calculations': stats.get('total_calculations', 0)
        }


def create_enhanced_n_step_predictor(max_horizon: int = 3) -> EnhancedNStepPredictor:
    """
    Convenience function to create enhanced N-step predictor.
    
    Args:
        max_horizon: Maximum prediction horizon
        
    Returns:
        EnhancedNStepPredictor instance
    """
    return EnhancedNStepPredictor(max_horizon)


if __name__ == '__main__':
    # Example usage
    print("Enhanced N-Step Prediction Integration Module")
    print("This module provides enhanced N-step prediction with anticipatory learning.")
    print("Use EnhancedNStepPredictor class for enhanced predictions.")



