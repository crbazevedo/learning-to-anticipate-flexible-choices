"""
Multi-Horizon Anticipatory Learning Implementation

This module implements the complete multi-horizon anticipatory learning
framework as specified in the thesis, including Equation 6.10 and
multi-horizon prediction capabilities.
"""

import numpy as np
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

# W1-3: relative imports (was `from algorithms.X import Y` which only
# worked under a `sys.path.insert(..., src)` hack and was one of the
# 16 pre-W1-3 pytest collection errors).
from .solution import Solution
from .n_step_prediction import NStepPredictor
from .sliding_window_dirichlet import SlidingWindowDirichlet
from .kalman_filter import KalmanParams
from .anticipatory_learning import TIPIntegratedAnticipatoryLearning

logger = logging.getLogger(__name__)


@dataclass
class MultiHorizonPrediction:
    """Data class for multi-horizon prediction results."""
    
    horizon: int
    predicted_state: np.ndarray
    predicted_covariance: np.ndarray
    lambda_rate: float
    tip_value: float
    confidence: float


class MultiHorizonAnticipatoryLearning(TIPIntegratedAnticipatoryLearning):
    """
    Multi-horizon anticipatory learning implementation.

    Implements paper Eq (14) (= thesis Eq 6.10) — the multi-horizon
    anticipatory learning convex combination over the predictive
    distributions ẑ_{t+h} | ẑ_t for h=1,...,H-1.

    Per W1-3 (2026-05-16) this class inherits from
    `TIPIntegratedAnticipatoryLearning` (which itself inherits from
    `AnticipatoryLearning`). Why TIPIntegrated and not the base:
    almost all of the actual learning-loop machinery
    (`learn_population`, `learn_single_solution`, `anticipatory_learning_obj_space`,
    `anticipatory_learning_dec_space`) lives on the TIPIntegrated
    subclass, not on the base. Inheriting from TIPIntegrated makes
    MultiHorizonAnticipatoryLearning a drop-in `set_learning(...)`
    target for SMS-EMOA / NSGA-II via the
    `ExperimentManager.learning.use_multi_horizon: true` flag.

    The multi-horizon-specific machinery (`apply_anticipatory_learning_rule`,
    `calculate_multi_horizon_lambda_rates`, `perform_multi_horizon_prediction`)
    remains this class's own surface; the inherited learn_* methods use the
    parent's single-horizon path until a deeper integration unit
    rewires them to drive the multi-horizon convex combination
    directly inside the run loop.
    """

    def __init__(self, max_horizon: int = 3, monte_carlo_samples: int = 1000):
        """
        Initialize multi-horizon anticipatory learning.

        Args:
            max_horizon: Maximum prediction horizon (H parameter)
            monte_carlo_samples: Number of Monte Carlo samples for TIP calculation
        """
        # Initialise the inherited TIPIntegratedAnticipatoryLearning surface
        # (which itself initialises AnticipatoryLearning). That gives this
        # instance learn_population / learn_single_solution /
        # anticipatory_learning_obj_space / kalman_filter_functions /
        # correspondence_mapping / tip_calculator — all for free.
        # IMPORTANT: keyword args only — the pre-W1-2 super-bug pattern
        # silently maps `super().__init__(N)` to the parent's first positional arg.
        super().__init__(window_size=20, monte_carlo_samples=monte_carlo_samples)

        self.max_horizon = max_horizon
        # Override the parent's prediction_horizon to match max_horizon
        # so multi-horizon callers read the right value.
        self.prediction_horizon = max_horizon
        # `monte_carlo_samples` is also stored by the parent's
        # tip_calculator; we keep a direct reference for backward
        # compatibility with calls inside this class.
        self.monte_carlo_samples = monte_carlo_samples

        # Initialize multi-horizon-specific components. tip_calculator is
        # already created by the parent (TIPIntegratedAnticipatoryLearning.__init__),
        # so we don't re-create it here.
        self.n_step_predictor = NStepPredictor(max_horizon)
        # W1-3: SlidingWindowDirichlet expects `window_size_K`, not
        # `window_size`. Pre-W1-3 the kwarg was wrong, but the class was
        # dead code (never instantiated from any live path), so the bug
        # never surfaced. Surfaced by test_eq14_integration constructing
        # the class for the first time.
        self.dirichlet_model = SlidingWindowDirichlet(window_size_K=20)

        # Storage for predictions and learning rates
        self.prediction_history: List[Dict[str, Any]] = []
        self.lambda_rates_history: List[Dict[str, float]] = []

        logger.info(f"Initialized MultiHorizonAnticipatoryLearning with max_horizon={max_horizon}")
    
    def apply_anticipatory_learning_rule(self, current_state: np.ndarray, 
                                       predicted_states: List[np.ndarray], 
                                       lambda_rates: List[float]) -> np.ndarray:
        """
        Implement complete Equation 6.10:
        ẑ_t | z_{t+1:t+H-1} = (1 - Σ_{h=1}^{H-1} λ_{t+h}) z_t + Σ_{h=1}^{H-1} λ_{t+h} ẑ_{t+h} | z_t
        
        Args:
            current_state: Current state vector z_t
            predicted_states: List of predicted states ẑ_{t+h} for h=1,...,H-1
            lambda_rates: List of learning rates λ_{t+h} for h=1,...,H-1
            
        Returns:
            Anticipatory state ẑ_t | z_{t+1:t+H-1}
        """
        if len(predicted_states) != len(lambda_rates):
            raise ValueError("Number of predicted states must match number of lambda rates")
        
        if len(predicted_states) == 0:
            return current_state.copy()
        
        # Calculate sum of lambda rates
        lambda_sum = sum(lambda_rates)
        
        # Ensure lambda_sum doesn't exceed 1.0
        if lambda_sum > 1.0:
            logger.warning(f"Lambda sum {lambda_sum} > 1.0, normalizing")
            lambda_rates = [rate / lambda_sum for rate in lambda_rates]
            lambda_sum = 1.0
        
        # First term: (1 - Σλ) z_t
        anticipatory_state = (1 - lambda_sum) * current_state
        
        # Second term: Σλ ẑ_{t+h}
        for predicted_state, lambda_h in zip(predicted_states, lambda_rates):
            anticipatory_state += lambda_h * predicted_state
        
        logger.debug(f"Applied anticipatory learning rule: lambda_sum={lambda_sum:.4f}")
        
        return anticipatory_state
    
    def calculate_multi_horizon_lambda_rates(self, solution: Solution, 
                                           prediction_horizon: int) -> List[float]:
        """
        Calculate λ_{t+h} rates for multiple horizons.
        
        Based on Equation 6.6: λ_{t+h} = (1/(H-1)) [1 - H(p_{t,t+h})]
        
        Args:
            solution: Current solution
            prediction_horizon: Prediction horizon H
            
        Returns:
            List of lambda rates for each horizon
        """
        if prediction_horizon < 2:
            # W3-1: was `return [0.0]` which violates the invariant
            # len(predicted_states) == len(lambda_rates) used by
            # apply_anticipatory_learning_rule (H<2 has 0 future
            # horizons, so 0 predicted_states, so 0 lambda_rates).
            return []
        
        lambda_rates = []
        
        for h in range(1, prediction_horizon):
            # Calculate TIP for horizon h
            tip = self._calculate_tip_for_horizon(solution, h)
            
            # Calculate binary entropy
            entropy = self.tip_calculator.binary_entropy(tip)
            
            # Calculate lambda rate based on Equation 6.6
            lambda_h = (1.0 / (prediction_horizon - 1)) * (1.0 - entropy)
            
            # Apply bounds to ensure stability
            lambda_h = max(0.0, min(0.5, lambda_h))
            
            lambda_rates.append(lambda_h)
        
        logger.debug(f"Calculated lambda rates for horizon {prediction_horizon}: {lambda_rates}")
        
        return lambda_rates
    
    def _calculate_tip_for_horizon(self, solution: Solution, horizon: int) -> float:
        """
        Calculate TIP for a specific horizon.
        
        Args:
            solution: Current solution
            horizon: Prediction horizon
            
        Returns:
            TIP value for the horizon
        """
        # Get current state
        current_roi = solution.P.ROI
        current_risk = solution.P.risk
        
        # Generate predicted solution for horizon h
        predicted_solution = self._generate_predicted_solution(solution, horizon)
        
        # Calculate TIP using the temporal incomparability calculator
        tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
        
        return tip
    
    def _generate_predicted_solution(self, solution: Solution, horizon: int) -> Solution:
        """
        Generate predicted solution for a specific horizon.
        
        Args:
            solution: Current solution
            horizon: Prediction horizon
            
        Returns:
            Predicted solution
        """
        # Create a copy of the solution
        predicted_solution = Solution(solution.P.num_assets)
        predicted_solution.P.investment = solution.P.investment.copy()
        
        # Use Kalman filter for state prediction if available
        if hasattr(solution.P, 'kalman_state') and solution.P.kalman_state is not None:
            kalman_state = solution.P.kalman_state
            
            # Perform n-step prediction
            predictions = self.n_step_predictor.kalman_n_step_prediction(kalman_state, horizon)
            
            if f'step_{horizon}' in predictions:
                step_prediction = predictions[f'step_{horizon}']
                predicted_state = step_prediction['state']
                
                # Update predicted solution
                predicted_solution.P.ROI = predicted_state[0]
                predicted_solution.P.risk = predicted_state[1]
                
                # Update Kalman state
                predicted_solution.P.kalman_state = KalmanParams(
                    x=predicted_state,
                    P=step_prediction['covariance'],
                    F=kalman_state.F,
                    H=kalman_state.H,
                    Q=kalman_state.Q,
                    R=kalman_state.R
                )
        else:
            # Fallback: simple linear prediction
            predicted_solution.P.ROI = solution.P.ROI * (1 + 0.01 * horizon)
            predicted_solution.P.risk = solution.P.risk * (1 + 0.005 * horizon)
        
        return predicted_solution
    
    def perform_multi_horizon_prediction(self, solution: Solution, 
                                       prediction_horizon: int) -> List[MultiHorizonPrediction]:
        """
        Perform multi-horizon prediction for a solution.
        
        Args:
            solution: Current solution
            prediction_horizon: Prediction horizon H
            
        Returns:
            List of multi-horizon predictions
        """
        if prediction_horizon > self.max_horizon:
            raise ValueError(f"Prediction horizon {prediction_horizon} exceeds maximum {self.max_horizon}")
        
        predictions = []
        
        for h in range(1, prediction_horizon + 1):
            # Generate predicted solution for horizon h
            predicted_solution = self._generate_predicted_solution(solution, h)
            
            # Calculate TIP for this horizon
            tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
            
            # Calculate lambda rate
            if prediction_horizon > 1:
                entropy = self.tip_calculator.binary_entropy(tip)
                lambda_rate = (1.0 / (prediction_horizon - 1)) * (1.0 - entropy)
                lambda_rate = max(0.0, min(0.5, lambda_rate))
            else:
                lambda_rate = 0.0
            
            # Calculate confidence based on TIP
            confidence = 1.0 - abs(tip - 0.5) * 2  # Higher confidence when TIP is closer to 0 or 1
            
            # Create prediction object
            prediction = MultiHorizonPrediction(
                horizon=h,
                predicted_state=np.array([predicted_solution.P.ROI, predicted_solution.P.risk]),
                predicted_covariance=self._get_predicted_covariance(predicted_solution),
                lambda_rate=lambda_rate,
                tip_value=tip,
                confidence=confidence
            )
            
            predictions.append(prediction)
        
        # Store prediction history
        self.prediction_history.append({
            'solution_id': id(solution),
            'horizon': prediction_horizon,
            'predictions': predictions,
            'timestamp': np.datetime64('now')
        })
        
        return predictions
    
    def _get_predicted_covariance(self, predicted_solution: Solution) -> np.ndarray:
        """
        Get predicted covariance matrix for a solution.
        
        Args:
            predicted_solution: Predicted solution
            
        Returns:
            Predicted covariance matrix
        """
        if (hasattr(predicted_solution.P, 'kalman_state') and 
            predicted_solution.P.kalman_state is not None):
            return predicted_solution.P.kalman_state.P[:2, :2]  # ROI and risk covariance
        else:
            # Default covariance matrix
            return np.eye(2) * 0.01
    
    def apply_multi_horizon_anticipatory_learning(self, solution: Solution, 
                                                prediction_horizon: int) -> Solution:
        """
        Apply multi-horizon anticipatory learning to a solution.
        
        Args:
            solution: Current solution
            prediction_horizon: Prediction horizon H
            
        Returns:
            Solution with applied anticipatory learning
        """
        # Perform multi-horizon prediction
        predictions = self.perform_multi_horizon_prediction(solution, prediction_horizon)
        
        if len(predictions) < 2:
            return solution  # No multi-horizon learning possible
        
        # Extract current state and predicted states
        current_state = np.array([solution.P.ROI, solution.P.risk])
        predicted_states = [pred.predicted_state for pred in predictions[1:]]  # Skip h=0
        lambda_rates = [pred.lambda_rate for pred in predictions[1:]]  # Skip h=0
        
        # Apply anticipatory learning rule (Equation 6.10)
        anticipatory_state = self.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Create new solution with anticipatory state
        anticipatory_solution = Solution(solution.P.num_assets)
        anticipatory_solution.P.investment = solution.P.investment.copy()
        anticipatory_solution.P.ROI = anticipatory_state[0]
        anticipatory_solution.P.risk = anticipatory_state[1]
        
        # Copy other attributes
        anticipatory_solution.alpha = solution.alpha
        anticipatory_solution.prediction_error = solution.prediction_error
        anticipatory_solution.anticipation = True
        
        # Store lambda rates history
        self.lambda_rates_history.append({
            'solution_id': id(solution),
            'horizon': prediction_horizon,
            'lambda_rates': lambda_rates,
            'timestamp': np.datetime64('now')
        })
        
        logger.debug(f"Applied multi-horizon anticipatory learning with horizon {prediction_horizon}")
        
        return anticipatory_solution
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about multi-horizon predictions.
        
        Returns:
            Dictionary with prediction statistics
        """
        if not self.prediction_history:
            return {'error': 'No prediction history available'}
        
        # Calculate statistics
        all_tips = []
        all_lambdas = []
        all_confidences = []
        
        for pred_entry in self.prediction_history:
            for pred in pred_entry['predictions']:
                all_tips.append(pred.tip_value)
                all_lambdas.append(pred.lambda_rate)
                all_confidences.append(pred.confidence)
        
        return {
            'total_predictions': len(self.prediction_history),
            'mean_tip': np.mean(all_tips) if all_tips else 0.0,
            'std_tip': np.std(all_tips) if all_tips else 0.0,
            'mean_lambda': np.mean(all_lambdas) if all_lambdas else 0.0,
            'std_lambda': np.std(all_lambdas) if all_lambdas else 0.0,
            'mean_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'std_confidence': np.std(all_confidences) if all_confidences else 0.0,
            'max_horizon_used': max(entry['horizon'] for entry in self.prediction_history)
        }
    
    def get_lambda_rates_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about lambda rates.
        
        Returns:
            Dictionary with lambda rates statistics
        """
        if not self.lambda_rates_history:
            return {'error': 'No lambda rates history available'}
        
        # Calculate statistics
        all_lambdas = []
        horizon_counts = {}
        
        for lambda_entry in self.lambda_rates_history:
            horizon = lambda_entry['horizon']
            horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
            
            for lambda_rate in lambda_entry['lambda_rates']:
                all_lambdas.append(lambda_rate)
        
        return {
            'total_entries': len(self.lambda_rates_history),
            'mean_lambda': np.mean(all_lambdas) if all_lambdas else 0.0,
            'std_lambda': np.std(all_lambdas) if all_lambdas else 0.0,
            'min_lambda': np.min(all_lambdas) if all_lambdas else 0.0,
            'max_lambda': np.max(all_lambdas) if all_lambdas else 0.0,
            'horizon_distribution': horizon_counts
        }
    
    def reset_history(self):
        """Reset prediction and lambda rates history."""
        self.prediction_history.clear()
        self.lambda_rates_history.clear()
        self.tip_calculator.reset_history()
        
        logger.info("Reset multi-horizon anticipatory learning history")
    
    def validate_prediction_horizon(self, horizon: int) -> bool:
        """
        Validate prediction horizon.
        
        Args:
            horizon: Prediction horizon to validate
            
        Returns:
            True if valid, False otherwise
        """
        return 1 <= horizon <= self.max_horizon
    
    def get_max_horizon(self) -> int:
        """Get maximum prediction horizon."""
        return self.max_horizon
    
    def set_max_horizon(self, max_horizon: int):
        """
        Set maximum prediction horizon.
        
        Args:
            max_horizon: New maximum prediction horizon
        """
        if max_horizon < 1:
            raise ValueError("Maximum horizon must be at least 1")
        
        self.max_horizon = max_horizon
        self.n_step_predictor.max_horizon = max_horizon
        
        logger.info(f"Set maximum prediction horizon to {max_horizon}")


def create_multi_horizon_anticipatory_learning(max_horizon: int = 3, 
                                             monte_carlo_samples: int = 1000) -> MultiHorizonAnticipatoryLearning:
    """
    Convenience function to create multi-horizon anticipatory learning instance.
    
    Args:
        max_horizon: Maximum prediction horizon
        monte_carlo_samples: Number of Monte Carlo samples
        
    Returns:
        MultiHorizonAnticipatoryLearning instance
    """
    return MultiHorizonAnticipatoryLearning(max_horizon, monte_carlo_samples)


if __name__ == '__main__':
    # Example usage
    print("Multi-Horizon Anticipatory Learning Module")
    print("This module provides multi-horizon anticipatory learning functionality.")
    print("Use MultiHorizonAnticipatoryLearning class for multi-horizon predictions.")
