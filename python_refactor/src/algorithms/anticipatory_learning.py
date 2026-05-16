"""
Enhanced Anticipatory Learning Module

Fully aligned with C++ ASMS-EMOA implementation including:
- Correct anticipatory learning rate formula
- Dirichlet mean prediction for decision space learning
- Historical population tracking
- Epsilon feasibility constraints
- Transaction cost integration
- Separate objective space and decision space learning
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

from .kalman_filter import KalmanParams, kalman_filter, kalman_prediction, kalman_update
from .statistics import multi_norm, normal_cdf, linear_entropy
from .solution import Solution
from .temporal_incomparability_probability import TemporalIncomparabilityCalculator
from .correspondence_mapping import CorrespondenceMapping

class AnticipativeDistribution:
    """Represents the anticipative distribution for portfolio state prediction."""
    
    def __init__(self, current_state: np.ndarray, predicted_state: np.ndarray, 
                 current_covariance: np.ndarray, predicted_covariance: np.ndarray):
        """
        Initialize anticipative distribution.
        
        Args:
            current_state: Current portfolio state [ROI, risk, ROI_velocity, risk_velocity]
            predicted_state: Predicted portfolio state
            current_covariance: Current state covariance matrix
            predicted_covariance: Predicted state covariance matrix
        """
        self.current_state = current_state
        self.predicted_state = predicted_state
        self.current_covariance = current_covariance
        self.predicted_covariance = predicted_covariance
        
        # Combined covariance for anticipative distribution
        self.anticipative_covariance = current_covariance + predicted_covariance
        
        # Anticipative mean (weighted combination)
        self.anticipative_mean = (current_state + predicted_state) / 2.0
    
    def sample_anticipative_state(self, num_samples: int = 1000) -> np.ndarray:
        """Sample from anticipative distribution."""
        return multi_norm(self.anticipative_mean, self.anticipative_covariance, num_samples)
    
    def compute_anticipative_confidence(self) -> float:
        """Compute confidence in anticipative distribution."""
        # Confidence based on determinant of anticipative covariance
        det = np.linalg.det(self.anticipative_covariance[:2, :2])  # ROI and risk only
        return 1.0 / (1.0 + det)

class DirichletPredictor:
    """Dirichlet-based prediction for portfolio weights."""
    
    @staticmethod
    def dirichlet_mean_prediction_vec(prev_proportions: np.ndarray, current_proportions: np.ndarray, 
                                    anticipative_rate: float) -> np.ndarray:
        """
        Predict portfolio weights using Dirichlet mean prediction.
        
        Args:
            prev_proportions: Previous portfolio weights
            current_proportions: Current portfolio weights
            anticipative_rate: Anticipatory learning rate
            
        Returns:
            Predicted portfolio weights
        """
        anticipative_rate = 0.5 * anticipative_rate
        prediction = prev_proportions + anticipative_rate * (current_proportions - prev_proportions)
        
        # Normalize and ensure bounds
        prediction = np.maximum(prediction, 0.0)
        prediction = np.minimum(prediction, 1.0)
        
        return prediction / np.sum(prediction)
    
    @staticmethod
    def dirichlet_mean_map_update(p_predicted: np.ndarray, p_obs: np.ndarray, 
                                concentration: float) -> np.ndarray:
        """
        Update portfolio weights using Dirichlet MAP estimation.
        
        Args:
            p_predicted: Predicted portfolio weights
            p_obs: Observed portfolio weights
            concentration: Concentration parameter
            
        Returns:
            Updated portfolio weights
        """
        # Compute Dirichlet variance
        alpha = concentration * p_predicted
        alpha_sum = np.sum(alpha)
        factor = alpha_sum**3 + alpha_sum**2
        alpha_square = alpha**2
        var = (alpha_sum * alpha - alpha_square) / factor
        
        # MAP update
        p_updated = np.zeros_like(p_predicted)
        for i in range(len(p_predicted)):
            if p_predicted[i] == 0.0 or var[i] == 0.0:
                p_updated[i] = 0.0
            else:
                p_updated[i] = p_predicted[i] + var[i] * (p_obs[i] - p_predicted[i]) / (p_predicted[i] * (1 - p_predicted[i]))
        
        return p_updated / np.sum(p_updated)

class AnticipatoryLearning:
    """Enhanced anticipatory learning system aligned with C++ ASMS-EMOA implementation."""
    
    def __init__(self, learning_rate: float = 0.01, prediction_horizon: int = 1,
                 monte_carlo_simulations: int = 1000, state_observation_frequency: int = 10,
                 error_threshold: float = 0.05, learning_type: str = "single_solution",
                 adaptive_learning: bool = True, window_size: int = 20):
        """
        Initialize enhanced anticipatory learning system.
        
        Args:
            learning_rate: Base learning rate for state updates
            prediction_horizon: Number of time steps to predict ahead (default: 1)
            monte_carlo_simulations: Number of Monte Carlo simulations
            state_observation_frequency: Frequency of state observations
            error_threshold: Threshold for prediction error
            learning_type: Type of learning ("single_solution" or "population")
            adaptive_learning: Whether to use adaptive learning rate
            window_size: Kalman filter window size
        """
        self.base_learning_rate = learning_rate
        self.prediction_horizon = prediction_horizon
        self.monte_carlo_simulations = monte_carlo_simulations
        self.state_observation_frequency = state_observation_frequency
        self.error_threshold = error_threshold
        self.learning_type = learning_type
        self.adaptive_learning = adaptive_learning
        self.window_size = window_size
        
        # Kalman filter functions for state tracking
        self.kalman_filter_functions = {
            'predict': kalman_prediction,
            'update': kalman_update,
            'filter': kalman_filter
        }
        
        # Historical tracking (aligned with C++ implementation)
        self.historical_populations = []
        self.historical_anticipative_decisions = []
        self.predicted_anticipative_decision = None
        
        # Initialize correspondence mapping
        self.correspondence_mapping = CorrespondenceMapping(max_history_size=50)
        
        # Learning history
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = []
        self.adaptive_learning_rates = []
        
        # Anticipative distributions storage
        self.anticipative_distributions = []
        
        # Stochastic Pareto frontiers storage for visualization
        self.stochastic_pareto_frontiers = []
        self.anticipative_pareto_frontiers = []
        
        # Epsilon feasibility parameters
        self.epsilon = 0.05
        self.ref_point = (0.0, 1.0)  # (ROI_ref, Risk_ref)
        
        # Transaction cost parameters
        self.brokerage_commission = 0.005  # 0.5%
        self.brokerage_fee = 0.0
        self.current_wealth = 1.0
        self.current_investment = None
    
    def store_historical_population(self, population: List[Solution]):
        """Store historical population for decision space learning."""
        historical_pop = []
        for solution in population:
            historical_sol = Solution(num_assets=len(solution.P.investment))
            historical_sol.P.investment = solution.P.investment.copy()
            historical_sol.P.ROI = solution.P.ROI
            historical_sol.P.risk = solution.P.risk
            historical_sol.alpha = getattr(solution, 'alpha', 0.0)
            historical_sol.prediction_error = getattr(solution, 'prediction_error', 0.0)
            historical_pop.append(historical_sol)
        
        self.historical_populations.append(historical_pop)
    
    def store_anticipative_decision(self, solution: Solution):
        """Store anticipative decision for historical tracking."""
        anticipative_sol = Solution(num_assets=len(solution.P.investment))
        anticipative_sol.P.investment = solution.P.investment.copy()
        anticipative_sol.P.ROI = solution.P.ROI
        anticipative_sol.P.risk = solution.P.risk
        anticipative_sol.alpha = getattr(solution, 'alpha', 0.0)
        anticipative_sol.prediction_error = getattr(solution, 'prediction_error', 0.0)
        
        self.historical_anticipative_decisions.append(anticipative_sol)
        self.predicted_anticipative_decision = anticipative_sol

    def _build_predicted_shadow(self, current: Solution):
        """Build a lightweight transient Solution-shaped object from `current`'s
        Kalman-filter one-step-ahead prediction.

        The TIP calculator (paper Eq 12) reads only ``.P.ROI``, ``.P.risk``,
        and ``.P.kalman_state.P[:2, :2]`` from each of its two arguments;
        this helper assembles a minimal shadow that satisfies that
        interface without paying the cost of a full ``Solution`` +
        ``Portfolio`` construction.

        State-vector indices follow paper Eq (11): ``x = [ROI, risk,
        ROI_velocity, risk_velocity]`` — canonicalised by W1-1.

        Returns None when no prediction is available (callers fall back
        to the KF-residuals-only path in `compute_anticipatory_learning_rate`).
        """
        from types import SimpleNamespace
        kf = getattr(current.P, 'kalman_state', None)
        if kf is None or getattr(kf, 'x_next', None) is None:
            return None
        # Reuse the predicted covariance when present; fall back to the
        # filtered covariance otherwise.
        cov_next = getattr(kf, 'P_next', None)
        if cov_next is None:
            cov_next = kf.P
        shadow_kalman = SimpleNamespace(
            x=kf.x_next.copy(),
            P=cov_next.copy(),
        )
        shadow_portfolio = SimpleNamespace(
            ROI=float(kf.x_next[0]),   # paper Eq (11): x[0] = ROI
            risk=float(kf.x_next[1]),  # paper Eq (11): x[1] = risk
            kalman_state=shadow_kalman,
        )
        return SimpleNamespace(P=shadow_portfolio)

    def compute_anticipatory_learning_rate(self, solution: Solution, min_error: float,
                                         max_error: float, min_alpha: float, max_alpha: float, 
                                         current_time: int, tip_calculator: Optional[TemporalIncomparabilityCalculator] = None,
                                         predicted_solution: Optional[Solution] = None, horizon: int = 2) -> float:
        """
        Compute anticipatory learning rate using TIP integration (Equation 7.16).
        
        This implements the enhanced formula that combines TIP and Kalman Filter residuals:
        λ_{t+h} = (1/2) * (λ^{(H)}_{t+h} + λ^{(K)}_{t+h})
        
        Args:
            solution: Solution to compute learning rate for
            min_error: Minimum prediction error in population
            max_error: Maximum prediction error in population
            min_alpha: Minimum alpha in population
            max_alpha: Maximum alpha in population
            current_time: Current time step
            tip_calculator: TIP calculator instance
            predicted_solution: Predicted solution for TIP calculation
            horizon: Prediction horizon
            
        Returns:
            Anticipatory learning rate
        """
        # Traditional learning rate (Kalman Filter based)
        traditional_rate = self._compute_traditional_learning_rate(
            solution, min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # TIP-based learning rate (Equation 6.6)
        tip_rate = 0.0
        if tip_calculator is not None and predicted_solution is not None:
            try:
                # Calculate TIP
                tip = tip_calculator.calculate_tip(solution, predicted_solution)
                
                # Calculate TIP-based learning rate (Equation 6.6)
                tip_rate = tip_calculator.calculate_anticipatory_learning_rate_tip(tip, horizon)
                
            except Exception as e:
                logger.warning(f"TIP calculation failed: {e}, using traditional rate only")
                tip_rate = 0.0
        
        # Combined learning rate (Equation 7.16)
        if tip_rate > 0.0:
            combined_rate = 0.5 * (traditional_rate + tip_rate)
        else:
            combined_rate = traditional_rate
        
        return combined_rate
    
    def _compute_traditional_learning_rate(self, solution: Solution, min_error: float, 
                                         max_error: float, min_alpha: float, max_alpha: float, 
                                         current_time: int) -> float:
        """
        Compute traditional anticipatory learning rate using the exact C++ formula.
        
        This is the original implementation for backward compatibility.
        """
        # Accuracy factor (aligned with C++ implementation)
        if min_error > 0.0 and (max_error - min_error) > 0.0:
            accuracy_factor = 1.0 - (solution.prediction_error - min_error) / (max_error - min_error)
        else:
            accuracy_factor = 0.0
        
        # Uncertainty factor
        uncertainty_factor = solution.alpha
        
        # Rate bounds (aligned with C++ implementation)
        if current_time == 0 or self.window_size == 0:
            rate_upb = 0.0
            rate_lwb = 0.0
        else:
            rate_upb = 0.5
            rate_lwb = 0.0
        
        # Anticipatory learning rate formula (exact C++ implementation)
        anticipation_rate = (rate_lwb + 
                           0.5 * uncertainty_factor * (rate_upb - rate_lwb) + 
                           0.5 * accuracy_factor * (rate_upb - rate_lwb))
        
        return anticipation_rate


class TIPIntegratedAnticipatoryLearning(AnticipatoryLearning):
    """
    Enhanced Anticipatory Learning with TIP integration.
    
    This class extends the base AnticipatoryLearning class to include
    Temporal Incomparability Probability (TIP) calculation and integration
    according to the thesis theoretical framework.
    """
    
    def __init__(self, window_size: int = 10, monte_carlo_samples: int = 1000):
        """
        Initialize TIP-integrated anticipatory learning.

        Args:
            window_size: Window size for historical tracking
            monte_carlo_samples: Number of Monte Carlo samples for TIP calculation
        """
        # Pre-W1-2 bug: `super().__init__(window_size)` passed window_size
        # POSITIONALLY, which silently became `learning_rate` in the parent
        # signature `(learning_rate, prediction_horizon, monte_carlo_simulations,
        # state_observation_frequency, error_threshold, learning_type,
        # adaptive_learning, window_size)`. Fixed to keyword form.
        super().__init__(window_size=window_size)
        self.tip_calculator = TemporalIncomparabilityCalculator(monte_carlo_samples)
        self.prediction_horizon = 2  # Default prediction horizon
        
    def set_prediction_horizon(self, horizon: int):
        """Set the prediction horizon for TIP calculation."""
        self.prediction_horizon = horizon
        
    def compute_enhanced_anticipatory_learning_rate(self, solution: Solution, 
                                                  predicted_solution: Solution,
                                                  min_error: float, max_error: float, 
                                                  min_alpha: float, max_alpha: float, 
                                                  current_time: int) -> Tuple[float, float, float]:
        """
        Compute enhanced anticipatory learning rate with TIP integration.
        
        Returns:
            Tuple of (combined_rate, traditional_rate, tip_rate)
        """
        # Traditional learning rate
        traditional_rate = self._compute_traditional_learning_rate(
            solution, min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # TIP-based learning rate
        tip_rate = 0.0
        try:
            # Calculate TIP
            tip = self.tip_calculator.calculate_tip(solution, predicted_solution)
            
            # Calculate TIP-based learning rate (Equation 6.6)
            tip_rate = self.tip_calculator.calculate_anticipatory_learning_rate_tip(
                tip, self.prediction_horizon
            )
            
        except Exception as e:
            logger.warning(f"TIP calculation failed: {e}, using traditional rate only")
            tip_rate = 0.0
        
        # Combined learning rate (Equation 7.16)
        if tip_rate > 0.0:
            combined_rate = 0.5 * (traditional_rate + tip_rate)
        else:
            combined_rate = traditional_rate
        
        return combined_rate, traditional_rate, tip_rate
    
    def get_tip_statistics(self) -> dict:
        """Get TIP statistics for analysis."""
        return self.tip_calculator.get_tip_statistics()
    
    def reset_tip_history(self):
        """Reset TIP calculation history."""
        self.tip_calculator.reset_history()
    
    def store_population_snapshot(self, population: List[Solution], current_time: int, 
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        Store a population snapshot for correspondence mapping.
        
        Args:
            population: Current population of solutions
            current_time: Current time step
            metadata: Optional metadata about the population
        """
        self.correspondence_mapping.store_population(population, current_time, metadata)
        
        # Also store in historical populations for backward compatibility
        self.historical_populations.append([sol for sol in population])
    
    def get_solution_evolution(self, solution_index: int, start_time: int, 
                              end_time: int) -> List[Solution]:
        """
        Get the evolution of a specific solution across time steps.
        
        Args:
            solution_index: Index of the solution to track
            start_time: Starting time step
            end_time: Ending time step
            
        Returns:
            List of solutions representing the evolution
        """
        return self.correspondence_mapping.track_solution_evolution(
            solution_index, start_time, end_time
        )
    
    def find_corresponding_solution(self, target_solution: Solution, target_time: int,
                                  search_time: int, similarity_threshold: float = 0.95) -> Optional[Solution]:
        """
        Find the solution in a different time step that corresponds to the target solution.
        
        Args:
            target_solution: Solution to find correspondence for
            target_time: Time step of the target solution
            search_time: Time step to search in
            similarity_threshold: Minimum similarity threshold for correspondence
            
        Returns:
            Corresponding solution or None if not found
        """
        return self.correspondence_mapping.find_corresponding_solution(
            target_solution, target_time, search_time, similarity_threshold
        )
    
    def get_correspondence_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about correspondence mapping.
        
        Returns:
            Dictionary with correspondence mapping statistics
        """
        return self.correspondence_mapping.get_history_summary()
    
    def anticipatory_learning_obj_space(self, solution: Solution, anticipative_portfolio: Solution,
                                       current_investment: np.ndarray, min_error: float, max_error: float,
                                       min_alpha: float, max_alpha: float, current_time: int):
        """
        Apply anticipatory learning in objective space (aligned with C++ implementation).
        
        Args:
            solution: Solution to update
            anticipative_portfolio: Anticipative portfolio
            current_investment: Current investment weights
            min_error: Minimum prediction error
            max_error: Maximum prediction error
            min_alpha: Minimum alpha
            max_alpha: Maximum alpha
            current_time: Current time step
        """
        if anticipative_portfolio is None:
            anticipative_portfolio = solution
        
        # Update bounds
        if anticipative_portfolio.prediction_error < min_error:
            min_error = anticipative_portfolio.prediction_error
        elif anticipative_portfolio.prediction_error > max_error:
            max_error = anticipative_portfolio.prediction_error
        
        if anticipative_portfolio.alpha < min_alpha:
            min_alpha = anticipative_portfolio.alpha
        elif anticipative_portfolio.alpha > max_alpha:
            max_alpha = anticipative_portfolio.alpha
        
        # Compute anticipatory learning rate.
        # When `self` is a TIPIntegratedAnticipatoryLearning instance (or any
        # subclass that exposes a `tip_calculator` attribute), thread the
        # TIP machinery through so paper Eq (13) actually fires the TIP arm
        # in the published λ — instead of silently falling back to the
        # KF-residuals-only path (the pre-W1-2 default for every live run).
        tip_calculator = getattr(self, 'tip_calculator', None)
        predicted_shadow = None
        if tip_calculator is not None:
            predicted_shadow = self._build_predicted_shadow(anticipative_portfolio)
        solution.anticipation_rate = self.compute_anticipatory_learning_rate(
            anticipative_portfolio, min_error, max_error, min_alpha, max_alpha, current_time,
            tip_calculator=tip_calculator,
            predicted_solution=predicted_shadow,
            horizon=getattr(self, 'prediction_horizon', 2),
        )
        
        # Update state using anticipatory learning rate
        x_state = np.array([anticipative_portfolio.P.ROI, anticipative_portfolio.P.risk, 0.0, 0.0])
        x = x_state.copy()
        C = anticipative_portfolio.P.kalman_state.P.copy()
        
        if self.window_size > 0:
            # Update state vector
            x = x_state + solution.anticipation_rate * (anticipative_portfolio.P.kalman_state.x_next - x_state)
            
            # Update covariance matrix
            C = (anticipative_portfolio.P.kalman_state.P + 
                 solution.anticipation_rate**2 * 
                 (anticipative_portfolio.P.kalman_state.P_next - anticipative_portfolio.P.kalman_state.P))
        
        # Update ROI with transaction cost consideration
        anticipative_portfolio.P.ROI = x[0]
        
        if anticipative_portfolio is not None:
            # Compute transaction costs (simplified version)
            cost_current = self.compute_transaction_cost(self.current_investment, solution.P.investment)
            cost_predicted = self.compute_transaction_cost(current_investment, anticipative_portfolio.P.investment)
            
            # Adjust ROI based on transaction costs
            adjusted_ROI = ((self.current_wealth + cost_current) / self.current_wealth) * anticipative_portfolio.P.ROI
            nominal_ROI = (self.current_wealth - cost_predicted) * adjusted_ROI
            anticipative_portfolio.P.ROI = nominal_ROI / self.current_wealth
        
        # Update risk (ensure non-negative)
        if x[1] > 0.0:
            anticipative_portfolio.P.risk = x[1]
        elif x[1] < 0.0 and anticipative_portfolio.P.kalman_state.x[1] > 0.0:
            anticipative_portfolio.P.risk = anticipative_portfolio.P.kalman_state.x[1]
        else:
            anticipative_portfolio.P.risk = 0.0
        
        # Update covariance
        anticipative_portfolio.P.kalman_state.P = C
        solution.P.kalman_state.P = C
        
        # Update solution
        solution.P.kalman_state = anticipative_portfolio.P.kalman_state
        solution.P.ROI = anticipative_portfolio.P.ROI
        solution.P.risk = anticipative_portfolio.P.risk
        
        # Mark as anticipated
        solution.anticipation = True
        anticipative_portfolio.anticipation = True
    
    def anticipatory_learning_dec_space(self, population: List[Solution], solution_idx: int, 
                                       current_time: int) -> Solution:
        """
        Apply anticipatory learning in decision space (aligned with C++ implementation).
        
        Args:
            population: Current population
            solution_idx: Index of solution to update
            current_time: Current time step
            
        Returns:
            Anticipative solution
        """
        if current_time == 0 or self.window_size == 0:
            return population[solution_idx]
        
        initial_t = max(0, current_time - self.window_size)
        
        # Get historical solution
        if initial_t < len(self.historical_populations):
            w_previous = self.historical_populations[initial_t][solution_idx]
        else:
            w_previous = population[solution_idx]
        
        # Iterate through time steps
        for t in range(initial_t + 1, current_time):
            if t < len(self.historical_populations):
                w_current = self.historical_populations[t][solution_idx]
            else:
                w_current = population[solution_idx]
            
            # Compute concentration
            previous_concentration = getattr(w_previous, 'nominal_ROI', 1.0)
            current_concentration = getattr(w_current, 'nominal_ROI', 1.0)
            concentration = previous_concentration + current_concentration
            
            # Dirichlet mean prediction
            w_prediction = self.dirichlet_mean_prediction(w_previous, w_current, t)
            
            # Update weights using Dirichlet MAP
            if t == current_time - 1:
                w_prediction.P.investment = DirichletPredictor.dirichlet_mean_map_update(
                    w_prediction.P.investment, population[solution_idx].P.investment, concentration
                )
            else:
                if t + 1 < current_time and t + 1 < len(self.historical_populations):
                    next_weights = self.historical_populations[t + 1][solution_idx].P.investment
                else:
                    next_weights = population[solution_idx].P.investment
                
                w_prediction.P.investment = DirichletPredictor.dirichlet_mean_map_update(
                    w_prediction.P.investment, next_weights, concentration
                )
            
            w_previous = w_prediction
        
        # Final prediction
        w_current = population[solution_idx]
        w_prediction = self.dirichlet_mean_prediction(w_previous, w_current, current_time)
        
        return w_prediction
    
    def dirichlet_mean_prediction(self, prev_solution: Solution, current_solution: Solution, 
                                current_time: int) -> Solution:
        """
        Predict solution using Dirichlet mean prediction (aligned with C++ implementation).
        
        Args:
            prev_solution: Previous solution
            current_solution: Current solution
            current_time: Current time step
            
        Returns:
            Predicted solution
        """
        predicted_solution = Solution(num_assets=len(current_solution.P.investment))
        predicted_solution.P.investment = current_solution.P.investment.copy()
        predicted_solution.P.ROI = current_solution.P.ROI
        predicted_solution.P.risk = current_solution.P.risk
        
        # Compute anticipative rate
        anticipative_rate = 2.0 - self.non_dominance_probability(prev_solution, current_solution)
        
        # Predict weights
        predicted_solution.P.investment = DirichletPredictor.dirichlet_mean_prediction_vec(
            prev_solution.P.investment, current_solution.P.investment, anticipative_rate
        )
        
        # Observe state
        initial_t = max(0, current_time - self.window_size)
        self._observe_state_1step_ahead(predicted_solution, current_time)
        
        # Kalman filter prediction
        if current_time == getattr(self, 'current_period', current_time):
            self.kalman_filter_functions['predict'](predicted_solution.P.kalman_state)
        
        return predicted_solution
    
    def non_dominance_probability(self, solution1: Solution, solution2: Solution) -> float:
        """
        Compute non-dominance probability between two solutions.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            Non-dominance probability
        """
        # Compute deltas
        delta1 = np.array([solution1.P.ROI - solution2.P.ROI, solution1.P.risk - solution2.P.risk])
        delta2 = np.array([solution2.P.ROI - solution1.P.ROI, solution2.P.risk - solution1.P.risk])
        
        # Point of interest
        u = np.array([0.0, 0.0])
        
        # Combine covariances
        covar = solution1.P.kalman_state.P[:2, :2] + solution2.P.kalman_state.P[:2, :2]
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(covar)
            L_inv = np.linalg.inv(L)
            
            # Transform to standard normal
            z1 = L_inv @ (u - delta1)
            z2 = L_inv @ (u - delta2)
            
            # Compute probabilities
            prob1 = normal_cdf(z1, np.eye(2))
            prob2 = normal_cdf(z2, np.eye(2))
            
            return prob1 + prob2
        except np.linalg.LinAlgError:
            return 0.5
    
    def compute_transaction_cost(self, current_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        Compute transaction cost for portfolio rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            new_weights: New portfolio weights
            
        Returns:
            Transaction cost
        """
        if current_weights is None:
            return 0.0
        
        # Compute turnover
        turnover = np.sum(np.abs(new_weights - current_weights))
        
        # Transaction cost = commission * turnover + fixed fee
        transaction_cost = self.brokerage_commission * turnover + self.brokerage_fee
        
        return transaction_cost
    
    def epsilon_feasibility(self, solution: Solution) -> Tuple[float, float]:
        """
        Compute epsilon feasibility probabilities (aligned with C++ implementation).
        
        Args:
            solution: Solution to evaluate
            
        Returns:
            Tuple of (ROI_feasibility, Risk_feasibility)
        """
        covar = solution.P.kalman_state.P[:2, :2]
        
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(covar)
            L_inv = np.linalg.inv(L)
            
            # Current point
            zpoint = np.array([solution.P.ROI, solution.P.risk])
            
            # Reference points
            zref_bottom = np.array([self.ref_point[0], 0.0])
            zref_upper = np.array([0.5, self.ref_point[1]])
            
            # Transform to standard normal
            zref_bottom_transformed = L_inv @ (zref_bottom - zpoint)
            zref_upper_transformed = L_inv @ (zref_upper - zpoint)
            
            # Compute probabilities
            p1 = 1.0 - normal_cdf(zref_bottom_transformed[0:1], np.eye(1))
            p2 = normal_cdf(zref_upper_transformed[1:2], np.eye(1))
            
            return (p1, p2)
        except np.linalg.LinAlgError:
            return (0.5, 0.5)
    
    def is_epsilon_feasible(self, solution: Solution) -> bool:
        """
        Check if solution is epsilon feasible.
        
        Args:
            solution: Solution to check
            
        Returns:
            True if epsilon feasible
        """
        feasibility = self.epsilon_feasibility(solution)
        return bool(feasibility[0] >= self.epsilon and feasibility[1] >= self.epsilon)
    
    def learn_single_solution(self, solution: Solution, current_time: int):
        """
        Apply enhanced anticipatory learning to a single solution.
        
        Args:
            solution: Solution to apply learning to
            current_time: Current time step
        """
        # Observe state via Monte Carlo simulation with 1-step ahead horizon
        self._observe_state_1step_ahead(solution, current_time)
        
        # Create anticipative distribution
        anticipative_dist = self._create_anticipative_distribution(solution)
        self.anticipative_distributions.append(anticipative_dist)
        
        # Compute prediction error and uncertainty
        prediction_error = self._compute_prediction_error(solution, current_time)
        state_quality = self._compute_state_quality(solution)
        
        # Compute non-dominance probability
        nd_probability = self._compute_non_dominance_probability(solution)
        
        # Compute adaptive learning rate
        if self.adaptive_learning:
            alpha = self._compute_adaptive_learning_rate(solution, prediction_error, nd_probability)
        else:
            alpha = 1.0 - linear_entropy(nd_probability)
        
        # Update solution state based on anticipatory knowledge
        self._update_solution_state_anticipative(solution, alpha, anticipative_dist)
        
        # Apply predicted portfolio rebalancing for maximal expected hypervolume
        self._apply_predicted_rebalancing(solution, anticipative_dist)
        
        # Store learning metrics
        solution.anticipation = True
        solution.alpha = alpha
        solution.prediction_error = prediction_error
        
        # Log learning event
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'solution_id': id(solution),
            'current_time': current_time,
            'alpha': alpha,
            'prediction_error': prediction_error,
            'state_quality': state_quality,
            'nd_probability': nd_probability,
            'anticipative_confidence': anticipative_dist.compute_anticipative_confidence(),
            'epsilon_feasibility': self.epsilon_feasibility(solution)
        })
        
        self.prediction_errors.append(prediction_error)
        self.state_qualities.append(state_quality)
        self.adaptive_learning_rates.append(alpha)
    
    def learn_population(self, population: List[Solution], current_time: int):
        """
        Apply anticipatory learning to entire population.
        
        Args:
            population: Population of solutions
            current_time: Current time step
        """
        # Store historical population
        self.store_historical_population(population)
        
        # Compute bounds for learning rate calculation
        min_error = float('inf')
        max_error = 0.0
        min_alpha = 1.0
        max_alpha = 0.0
        
        for solution in population:
            if not hasattr(solution, 'anticipation') or not solution.anticipation:
                self._observe_state_1step_ahead(solution, current_time)
                solution.prediction_error = self._compute_prediction_error(solution, current_time)
                solution.alpha = 1.0 - self._compute_non_dominance_probability(solution)
            
            # Update bounds
            if solution.alpha < min_alpha:
                min_alpha = solution.alpha
            if solution.alpha > max_alpha:
                max_alpha = solution.alpha
            if solution.prediction_error < min_error:
                min_error = solution.prediction_error
            if solution.prediction_error > max_error:
                max_error = solution.prediction_error
        
        # Apply anticipatory learning to each solution
        for i, solution in enumerate(population):
            if not hasattr(solution, 'anticipation') or not solution.anticipation:
                # Decision space learning
                if current_time > 0:
                    anticipative_solution = self.anticipatory_learning_dec_space(population, i, current_time)
                    
                    # Objective space learning
                    self.anticipatory_learning_obj_space(
                        solution, anticipative_solution, 
                        self.current_investment if self.current_investment is not None else solution.P.investment,
                        min_error, max_error, min_alpha, max_alpha, current_time
                    )
                else:
                    self.anticipatory_learning_obj_space(
                        solution, None, 
                        self.current_investment if self.current_investment is not None else solution.P.investment,
                        min_error, max_error, min_alpha, max_alpha, current_time
                    )
        
        # Store stochastic Pareto frontier for visualization
        self._store_stochastic_pareto_frontier(population, current_time)
    
    def _observe_state_1step_ahead(self, solution: Solution, current_time: int):
        """
        Observe portfolio state with 1-step ahead horizon.
        
        Args:
            solution: Solution to observe
            current_time: Current time step
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Initialize state if not already done
        if not hasattr(portfolio, 'state_initialized') or not portfolio.state_initialized:
            self._initialize_state(portfolio, current_time)
            portfolio.state_initialized = True
        
        # Run Monte Carlo simulations for 1-step ahead prediction
        roi_samples = []
        risk_samples = []
        
        for _ in range(self.monte_carlo_simulations):
            # Sample from current portfolio state
            current_state = kalman_state.x
            
            # Generate 1-step ahead scenarios
            future_state = self._simulate_1step_ahead_state(current_state, kalman_state.P)
            
            # Extract ROI and risk from future state
            future_roi = future_state[0]
            future_risk = future_state[1]
            
            roi_samples.append(future_roi)
            risk_samples.append(future_risk)
        
        # Compute statistics from samples
        mean_roi = np.mean(roi_samples)
        mean_risk = np.mean(risk_samples)
        var_roi = np.var(roi_samples)
        var_risk = np.var(risk_samples)
        cov_roi_risk = np.cov(roi_samples, risk_samples)[0, 1] if len(roi_samples) > 1 else 0.0
        
        # Update measurement noise covariance
        kalman_state.R = np.array([
            [var_roi / self.monte_carlo_simulations, cov_roi_risk / self.monte_carlo_simulations],
            [cov_roi_risk / self.monte_carlo_simulations, var_risk / self.monte_carlo_simulations]
        ])
        
        # Update Kalman filter with observed state
        measurement = np.array([mean_roi, mean_risk])
        self.kalman_filter_functions['update'](kalman_state, measurement)
        
        # Store predictions for 1-step ahead
        portfolio.ROI_prediction = kalman_state.x_next[0] if hasattr(kalman_state, 'x_next') else mean_roi
        portfolio.risk_prediction = kalman_state.x_next[1] if hasattr(kalman_state, 'x_next') else mean_risk
        portfolio.error_covar_prediction = kalman_state.P_next if hasattr(kalman_state, 'P_next') else kalman_state.P
    
    def _simulate_1step_ahead_state(self, current_state: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Simulate 1-step ahead state using Monte Carlo sampling.
        
        Args:
            current_state: Current state vector
            covariance: State covariance matrix
            
        Returns:
            Simulated 1-step ahead state
        """
        # Sample from multivariate normal distribution
        future_state = multi_norm(current_state, covariance, 1).flatten()
        
        # Apply 1-step state transition
        F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        return F @ future_state
    
    def _create_anticipative_distribution(self, solution: Solution) -> AnticipativeDistribution:
        """
        Create anticipative distribution for solution.
        
        Args:
            solution: Solution to create distribution for
            
        Returns:
            Anticipative distribution
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        current_state = kalman_state.x
        predicted_state = kalman_state.x_next if hasattr(kalman_state, 'x_next') else current_state
        current_covariance = kalman_state.P
        predicted_covariance = kalman_state.P_next if hasattr(kalman_state, 'P_next') else current_covariance
        
        return AnticipativeDistribution(
            current_state, predicted_state, current_covariance, predicted_covariance
        )
    
    def _initialize_state(self, portfolio, current_time: int):
        """Initialize Kalman filter state."""
        # Create Kalman state if it doesn't exist
        if portfolio.kalman_state is None:
            from .kalman_filter import create_kalman_params
            portfolio.kalman_state = create_kalman_params(portfolio.ROI, portfolio.risk)
        
        kalman_state = portfolio.kalman_state
        
        # Initial state: [ROI, risk, ROI_velocity, risk_velocity]
        initial_state = np.array([portfolio.ROI, portfolio.risk, 0.0, 0.0])
        kalman_state.x = initial_state
        
        # Initial covariance matrix
        kalman_state.P = np.array([
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0]
        ])
        
        # State transition matrix (constant velocity model)
        kalman_state.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement matrix
        kalman_state.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Initial measurement noise
        kalman_state.R = np.array([
            [0.01, 0.0],
            [0.0, 0.01]
        ])
    
    def _compute_prediction_error(self, solution: Solution, current_time: int) -> float:
        """Compute prediction error for solution."""
        portfolio = solution.P
        
        # Compare predicted vs current values
        roi_error = abs(portfolio.ROI_prediction - portfolio.ROI)
        risk_error = abs(portfolio.risk_prediction - portfolio.risk)
        
        # Normalize errors
        roi_error_norm = roi_error / (abs(portfolio.ROI) + 1e-8)
        risk_error_norm = risk_error / (abs(portfolio.risk) + 1e-8)
        
        # Combined error
        prediction_error = (roi_error_norm + risk_error_norm) / 2.0
        
        return prediction_error
    
    def _compute_state_quality(self, solution: Solution) -> float:
        """Compute quality of state observation."""
        kalman_state = solution.P.kalman_state
        
        # State quality based on covariance matrix determinant
        covariance_det = np.linalg.det(kalman_state.P[:2, :2])
        
        # Normalize to [0, 1] range
        state_quality = 1.0 / (1.0 + covariance_det)
        
        return state_quality
    
    def _compute_non_dominance_probability(self, solution: Solution) -> float:
        """Compute probability that solution will be non-dominated in future."""
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Current state
        current_roi = portfolio.ROI
        current_risk = portfolio.risk
        
        # Predicted state
        predicted_roi = portfolio.ROI_prediction
        predicted_risk = portfolio.risk_prediction
        
        # Compute deltas
        delta1 = np.array([current_roi - predicted_roi, current_risk - predicted_risk])
        delta2 = np.array([predicted_roi - current_roi, predicted_risk - current_risk])
        
        # Point of interest: u = [0, 0]^T
        u = np.array([0.0, 0.0])
        
        # Combine prediction and observation covariances
        covar = kalman_state.P_next[:2, :2] + kalman_state.P[:2, :2]
        
        # Compute Cholesky decomposition
        try:
            L = np.linalg.cholesky(covar)
            L_inv = np.linalg.inv(L)
            
            # Transform to standard normal
            z1 = L_inv @ (u - delta1)
            z2 = L_inv @ (u - delta2)
            
            # Compute probabilities
            prob1 = normal_cdf(z1, np.eye(2))
            prob2 = normal_cdf(z2, np.eye(2))
            
            # Non-dominance probability
            nd_probability = prob1 + prob2
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is not positive definite
            nd_probability = 0.5
        
        return nd_probability
    
    def _compute_adaptive_learning_rate(self, solution: Solution, prediction_error: float, 
                                      nd_probability: float) -> float:
        """
        Compute adaptive learning rate based on Kalman error and entropy.
        
        Args:
            solution: Solution to compute learning rate for
            prediction_error: Current prediction error
            nd_probability: Non-dominance probability
            
        Returns:
            Adaptive learning rate
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Kalman error: determinant of prediction covariance
        kalman_error = np.linalg.det(kalman_state.P_next[:2, :2]) if hasattr(kalman_state, 'P_next') else np.linalg.det(kalman_state.P[:2, :2])
        
        # Entropy over probability of dominance
        dominance_entropy = linear_entropy(nd_probability)
        
        # Adaptive learning rate: combines base rate, Kalman error, and entropy
        error_factor = 1.0 / (1.0 + kalman_error)
        entropy_factor = 1.0 - dominance_entropy
        
        adaptive_alpha = self.base_learning_rate * error_factor * entropy_factor
        
        # Ensure alpha is in [0, 1] range
        adaptive_alpha = np.clip(adaptive_alpha, 0.0, 1.0)
        
        return adaptive_alpha
    
    def _update_solution_state_anticipative(self, solution: Solution, alpha: float, 
                                          anticipative_dist: AnticipativeDistribution):
        """
        Update solution state using anticipative distribution.
        
        Args:
            solution: Solution to update
            alpha: Learning rate
            anticipative_dist: Anticipative distribution
        """
        portfolio = solution.P
        kalman_state = portfolio.kalman_state
        
        # Update state vector using anticipative distribution
        anticipative_state = anticipative_dist.anticipative_mean
        kalman_state.x = kalman_state.x + alpha * (anticipative_state - kalman_state.x)
        
        # Update covariance matrix
        anticipative_covariance = anticipative_dist.anticipative_covariance
        kalman_state.P = kalman_state.P + alpha * (anticipative_covariance - kalman_state.P)
        
        # Update portfolio metrics
        portfolio.ROI = kalman_state.x[0]
        portfolio.risk = kalman_state.x[1]
        
        # Update robust/non-robust metrics
        if hasattr(portfolio, 'robustness') and portfolio.robustness:
            portfolio.robust_ROI = portfolio.ROI
            portfolio.robust_risk = portfolio.risk
        else:
            portfolio.non_robust_ROI = portfolio.ROI
            portfolio.non_robust_risk = portfolio.risk
    
    def _apply_predicted_rebalancing(self, solution: Solution, anticipative_dist: AnticipativeDistribution):
        """
        Apply predicted portfolio rebalancing for maximal expected hypervolume.
        
        Args:
            solution: Solution to rebalance
            anticipative_dist: Anticipative distribution
        """
        portfolio = solution.P
        
        # Sample from anticipative distribution to get predicted states
        predicted_states = anticipative_dist.sample_anticipative_state(100)
        
        # Extract predicted ROI and risk values
        predicted_rois = predicted_states[:, 0]
        predicted_risks = predicted_states[:, 1]
        
        # Compute expected hypervolume contribution for current weights
        current_expected_hv = self._compute_expected_hypervolume_contribution(
            predicted_rois, predicted_risks, portfolio.investment
        )
        
        # Optimize weights for maximal expected hypervolume
        optimal_weights = self._optimize_weights_for_expected_hypervolume(
            predicted_rois, predicted_risks, portfolio.investment
        )
        
        # Apply rebalancing if improvement is significant
        if optimal_weights is not None:
            optimal_expected_hv = self._compute_expected_hypervolume_contribution(
                predicted_rois, predicted_risks, optimal_weights
            )
            
            if optimal_expected_hv > current_expected_hv * 1.05:  # 5% improvement threshold
                portfolio.investment = optimal_weights
                portfolio.cardinality = np.sum(optimal_weights > 0.01)
    
    def _compute_expected_hypervolume_contribution(self, predicted_rois: np.ndarray, 
                                                 predicted_risks: np.ndarray, 
                                                 weights: np.ndarray) -> float:
        """Compute expected hypervolume contribution for given weights."""
        # This is a simplified version - in practice, you'd need to compute
        # hypervolume contribution considering the full Pareto front
        expected_roi = np.mean(predicted_rois)
        expected_risk = np.mean(predicted_risks)
        
        # Simple hypervolume approximation
        return expected_roi * (1.0 - expected_risk)
    
    def _optimize_weights_for_expected_hypervolume(self, predicted_rois: np.ndarray, 
                                                 predicted_risks: np.ndarray, 
                                                 current_weights: np.ndarray) -> Optional[np.ndarray]:
        """Optimize weights for maximal expected hypervolume."""
        # This is a placeholder - in practice, you'd implement a proper optimization
        # algorithm to find weights that maximize expected hypervolume
        
        # For now, return None (no rebalancing)
        return None
    
    def _store_stochastic_pareto_frontier(self, population: List[Solution], current_time: int):
        """
        Store stochastic Pareto frontier for visualization.
        
        Args:
            population: Current population
            current_time: Current time step
        """
        # Extract Pareto front
        pareto_front = [s for s in population if s.Pareto_rank == 0]
        
        # Create stochastic Pareto frontier representation
        stochastic_frontier = []
        for solution in pareto_front:
            if hasattr(solution, 'P') and hasattr(solution.P, 'kalman_state'):
                kalman_state = solution.P.kalman_state
                stochastic_frontier.append({
                    'roi': solution.P.ROI,
                    'risk': solution.P.risk,
                    'roi_prediction': solution.P.ROI_prediction,
                    'risk_prediction': solution.P.risk_prediction,
                    'roi_variance': kalman_state.P[0, 0],
                    'risk_variance': kalman_state.P[1, 1],
                    'covariance': kalman_state.P[0, 1],
                    'alpha': getattr(solution, 'alpha', 0.0),
                    'prediction_error': getattr(solution, 'prediction_error', 0.0),
                    'weights': solution.P.investment.tolist()
                })
        
        # Store with timestamp
        self.stochastic_pareto_frontiers.append({
            'timestamp': datetime.now().isoformat(),
            'current_time': current_time,
            'frontier': stochastic_frontier
        })
        
        # Create anticipative Pareto frontier
        anticipative_frontier = []
        for solution in pareto_front:
            if hasattr(solution, 'P') and hasattr(solution.P, 'kalman_state'):
                kalman_state = solution.P.kalman_state
                anticipative_frontier.append({
                    'roi': kalman_state.x[0],
                    'risk': kalman_state.x[1],
                    'roi_variance': kalman_state.P[0, 0],
                    'risk_variance': kalman_state.P[1, 1],
                    'covariance': kalman_state.P[0, 1],
                    'weights': solution.P.investment.tolist()
                })
        
        self.anticipative_pareto_frontiers.append({
            'timestamp': datetime.now().isoformat(),
            'current_time': current_time,
            'frontier': anticipative_frontier
        })
    
    def apply_dirichlet_map_filtering(self, solution: Solution):
        """Apply Dirichlet MAP filtering to portfolio weights."""
        portfolio = solution.P
        weights = portfolio.investment
        
        # Dirichlet MAP estimation
        alpha_prior = np.ones_like(weights)
        alpha_likelihood = weights * 10
        alpha_posterior = alpha_prior + alpha_likelihood
        
        # MAP estimate
        K = len(weights)
        sum_alpha = np.sum(alpha_posterior)
        
        if sum_alpha > K:
            map_weights = (alpha_posterior - 1) / (sum_alpha - K)
        else:
            map_weights = np.ones_like(weights) / K
        
        # Ensure weights sum to 1 and are non-negative
        map_weights = np.maximum(map_weights, 0.0)
        map_weights = map_weights / np.sum(map_weights)
        
        # Update portfolio weights
        portfolio.investment = map_weights
        portfolio.cardinality = np.sum(map_weights > 0.01)
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning performance metrics."""
        if not self.learning_history:
            return {}
        
        # Compute metrics from history
        alphas = [entry['alpha'] for entry in self.learning_history]
        prediction_errors = [entry['prediction_error'] for entry in self.learning_history]
        state_qualities = [entry['state_quality'] for entry in self.learning_history]
        
        metrics = {
            'total_learning_events': len(self.learning_history),
            'mean_alpha': np.mean(alphas),
            'std_alpha': np.std(alphas),
            'mean_prediction_error': np.mean(prediction_errors),
            'std_prediction_error': np.std(prediction_errors),
            'mean_state_quality': np.mean(state_qualities),
            'std_state_quality': np.std(state_qualities),
            'learning_trend': self._compute_learning_trend(),
            'adaptive_learning_enabled': self.adaptive_learning,
            'prediction_horizon': self.prediction_horizon
        }
        
        return metrics
    
    def get_stochastic_pareto_frontiers(self) -> List[Dict[str, Any]]:
        """Get stored stochastic Pareto frontiers for visualization."""
        return self.stochastic_pareto_frontiers
    
    def get_anticipative_pareto_frontiers(self) -> List[Dict[str, Any]]:
        """Get stored anticipative Pareto frontiers for visualization."""
        return self.anticipative_pareto_frontiers
    
    def _compute_learning_trend(self) -> float:
        """Compute learning trend based on prediction error."""
        if len(self.prediction_errors) < 10:
            return 0.0
        
        recent_errors = self.prediction_errors[-10:]
        x = np.arange(len(recent_errors))
        slope = np.polyfit(x, recent_errors, 1)[0]
        
        return -slope
    
    def reset(self):
        """Reset learning system."""
        self.learning_history = []
        self.prediction_errors = []
        self.state_qualities = []
        self.adaptive_learning_rates = []
        self.anticipative_distributions = []
        self.stochastic_pareto_frontiers = []
        self.anticipative_pareto_frontiers = []
        self.historical_populations = []
        self.historical_anticipative_decisions = []
        self.predicted_anticipative_decision = None 