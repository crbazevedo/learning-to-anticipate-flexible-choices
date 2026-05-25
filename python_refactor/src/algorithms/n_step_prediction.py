#!/usr/bin/env python3
"""
N-Step Ahead Prediction Module
Extends Kalman filter and Dirichlet filter for multi-step ahead predictions
"""

import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class NStepPredictor:
    """N-step ahead prediction for anticipatory learning"""
    
    def __init__(self, max_horizon: int = 3):
        self.max_horizon = max_horizon
        
    # W22-NC13a: clamp ceiling on predicted-covariance diagonal elements.
    # Without a ceiling, h-step covariance propagation `F @ P @ F^T` grows
    # the position component by P[velocity, velocity] each step. With NC7's
    # high velocity prior (P[2,2] = 1000), after h steps the position
    # covariance reaches ~h × 1000. TIP MC sampling from
    # Normal([predicted_roi, predicted_risk], predicted_cov) then has
    # std ≈ √(h × 1000) around means ≈ 0.001 — pure noise → mutual
    # non-dominance ≈ 0.5 → TIP saturates regardless of NC12 fix
    # (Probe B POST-NC12 still showed 99.87% TIP saturation).
    #
    # Rationale for clamp ceiling = 1.0: position covariance > 1.0 means
    # the implied std > 1.0 (100% point swing) which dwarfs any realistic
    # portfolio ROI/risk magnitude (typically O(0.001-0.1)). Beyond this
    # ceiling, the predictive distribution is uninformative for TIP
    # comparison. Effectively this is a "predict but don't claim more
    # uncertainty than 100% std" rule.
    #
    # Predicted impact (to be validated by post-NC13a Probe B):
    #   - TIP MC samples will have std ≤ 1.0 (vs current ~30) → samples
    #     concentrate near means → mutual non-dominance probability depends
    #     on means' relative dominance → TIP escapes 0.5 saturation
    #   - Tradeoff: under-states true uncertainty when |F^h · P · F^{h.T}|
    #     is genuinely large; but the alternative (pure noise TIP) is
    #     worse — it actively destroys the anticipation signal.
    _PREDICTED_COV_DIAG_CLAMP: float = 1.0

    def kalman_n_step_prediction(self, kalman_state, h: int) -> Dict:
        """
        Perform n-step ahead prediction using Kalman filter.

        Args:
            kalman_state: Current Kalman state
            h: Prediction horizon (1, 2, 3, ...)

        Returns:
            Dict with predicted state and covariance for each step

        W22-NC13a: covariance ceiling applied to diagonal elements to
        prevent TIP MC saturation under unbounded h-step propagation.
        """
        if h > self.max_horizon:
            raise ValueError(f"Horizon {h} exceeds maximum {self.max_horizon}")

        predictions = {}
        current_state = kalman_state.x.copy()
        current_cov = kalman_state.P.copy()

        for step in range(1, h + 1):
            # Predict next state
            F = kalman_state.F  # State transition matrix

            # State prediction
            predicted_state = F @ current_state

            # Covariance prediction (without process noise Q)
            predicted_cov = F @ current_cov @ F.T

            # W22-NC13a: clamp diagonal of predicted_cov to a ceiling so
            # downstream TIP MC sampling does not collapse into pure noise.
            # The clamp ONLY affects predicted_cov returned to callers; the
            # internal `current_cov` rolled forward to the next step is the
            # UNCLAMPED true propagation (preserving Kalman invariants for
            # multi-step compounding consistency).
            clamped_cov = predicted_cov.copy()
            diag = np.diag(clamped_cov)
            if np.any(diag > self._PREDICTED_COV_DIAG_CLAMP):
                # Scale-down: shrink all diagonals proportionally so the
                # max diag = clamp, preserving the correlation structure.
                # If only some exceed, use min(1, clamp / max_diag) factor.
                max_diag = float(np.max(diag))
                scale = self._PREDICTED_COV_DIAG_CLAMP / max_diag
                clamped_cov = clamped_cov * scale

            predictions[f'step_{step}'] = {
                'state': predicted_state.copy(),
                'covariance': clamped_cov,
                'horizon': step
            }

            # Update for next iteration with UNCLAMPED propagation
            current_state = predicted_state.copy()
            current_cov = predicted_cov.copy()

        return predictions
    
    def dirichlet_n_step_prediction(self, dirichlet_params: np.ndarray, 
                                  historical_data: List[np.ndarray], 
                                  h: int) -> Dict:
        """
        Perform n-step ahead prediction using Dirichlet filter
        
        Args:
            dirichlet_params: Current Dirichlet parameters
            historical_data: Historical portfolio compositions
            h: Prediction horizon
            
        Returns:
            Dict with predicted distributions for each step
        """
        if h > self.max_horizon:
            raise ValueError(f"Horizon {h} exceeds maximum {self.max_horizon}")
            
        predictions = {}
        current_params = dirichlet_params.copy()
        
        for step in range(1, h + 1):
            # Dirichlet mean prediction for step h
            if len(historical_data) > 0:
                # Use historical trend for prediction
                recent_trend = np.mean(historical_data[-min(5, len(historical_data)):], axis=0)
                trend_weight = 0.1 * step  # Increasing trend weight with horizon
                
                # Update parameters based on trend
                updated_params = current_params * (1 - trend_weight) + recent_trend * trend_weight
            else:
                updated_params = current_params
                
            # Normalize parameters
            updated_params = np.maximum(updated_params, 0.01)  # Avoid zero parameters
            
            predictions[f'step_{step}'] = {
                'dirichlet_params': updated_params.copy(),
                'mean_prediction': updated_params / np.sum(updated_params),
                'horizon': step
            }
            
            current_params = updated_params.copy()
            
        return predictions
    
    def compute_expected_future_hypervolume(self, pareto_frontier: List, 
                                          kalman_predictions: Dict,
                                          dirichlet_predictions: Dict,
                                          h: int) -> Dict:
        """
        Compute expected future hypervolume distribution for current population
        
        Args:
            pareto_frontier: Current Pareto frontier solutions
            kalman_predictions: N-step Kalman predictions
            dirichlet_predictions: N-step Dirichlet predictions
            h: Prediction horizon
            
        Returns:
            Dict with expected hypervolume distributions
        """
        if f'step_{h}' not in kalman_predictions:
            raise ValueError(f"No prediction available for horizon {h}")
            
        expected_hypervolumes = {}
        
        for i, solution in enumerate(pareto_frontier):
            # Get predictions for horizon h
            kalman_pred = kalman_predictions[f'step_{h}']
            dirichlet_pred = dirichlet_predictions[f'step_{h}']
            
            # Compute expected future state
            expected_state = kalman_pred['state']
            expected_portfolio_weights = dirichlet_pred['mean_prediction']
            
            # Compute expected hypervolume contribution
            expected_hv = self._compute_solution_expected_hypervolume(
                solution, expected_state, expected_portfolio_weights, h)
            
            expected_hypervolumes[f'solution_{i}'] = {
                'expected_hypervolume': expected_hv,
                'kalman_state': expected_state,
                'dirichlet_weights': expected_portfolio_weights,
                'horizon': h
            }
            
        return expected_hypervolumes
    
    def _compute_solution_expected_hypervolume(self, solution, 
                                             expected_state: np.ndarray,
                                             expected_weights: np.ndarray,
                                             h: int) -> float:
        """Compute expected hypervolume for a single solution"""
        # Base hypervolume contribution
        base_hv = getattr(solution, 'hypervolume_contribution', 0.0)
        
        # Adjust based on expected state and weights
        state_factor = np.mean(expected_state) if len(expected_state) > 0 else 1.0
        weight_factor = np.sum(expected_weights) if len(expected_weights) > 0 else 1.0
        
        # Horizon discount factor
        horizon_factor = 1.0 / (1.0 + 0.1 * h)
        
        expected_hv = base_hv * state_factor * weight_factor * horizon_factor
        
        return max(expected_hv, 0.0)  # Ensure non-negative
    
    def compute_conditional_expected_hypervolume(self, pareto_frontier: List,
                                               selected_solution: int,
                                               kalman_predictions: Dict,
                                               dirichlet_predictions: Dict,
                                               h: int) -> Dict:
        """
        Compute conditional expected hypervolume given choice of AMFC/R-DM/M-DM
        
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
        
        # Compute conditional expectations
        conditional_hypervolumes = {}
        
        for i, solution in enumerate(pareto_frontier):
            if i == selected_solution:
                # For selected solution, use full expected hypervolume
                conditional_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h)
            else:
                # For other solutions, adjust based on selection
                base_hv = self._compute_solution_expected_hypervolume(
                    solution, kalman_pred['state'], dirichlet_pred['mean_prediction'], h)
                
                # Reduce hypervolume due to selection of another solution
                reduction_factor = 0.8  # 20% reduction
                conditional_hv = base_hv * reduction_factor
            
            conditional_hypervolumes[f'solution_{i}'] = {
                'conditional_expected_hypervolume': conditional_hv,
                'is_selected': (i == selected_solution),
                'horizon': h
            }
            
        return conditional_hypervolumes

class BenchmarkCalculator:
    """Calculate traditional benchmarks for comparison"""
    
    def __init__(self):
        self.benchmarks = {}
        
    def calculate_index_benchmark(self, returns_data: np.ndarray) -> Dict:
        """Calculate equal-weighted index benchmark"""
        # Equal-weighted index
        index_returns = np.mean(returns_data, axis=1)
        cumulative_return = np.prod(1 + index_returns) - 1
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.mean(index_returns) / np.std(index_returns) if np.std(index_returns) > 0 else 0
        
        return {
            'type': 'Equal-Weighted Index',
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': np.std(index_returns),
            'returns': index_returns
        }
    
    def calculate_sharpe_optimal(self, returns_data: np.ndarray) -> Dict:
        """Calculate Sharpe ratio optimal portfolio"""
        # Mean returns and covariance
        mean_returns = np.mean(returns_data, axis=0)
        cov_matrix = np.cov(returns_data.T)
        
        try:
            # Optimal weights for maximum Sharpe ratio
            inv_cov = np.linalg.inv(cov_matrix)
            optimal_weights = inv_cov @ mean_returns
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            
            # Calculate portfolio returns
            portfolio_returns = returns_data @ optimal_weights
            cumulative_return = np.prod(1 + portfolio_returns) - 1
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
            
            return {
                'type': 'Sharpe Optimal',
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'volatility': np.std(portfolio_returns),
                'weights': optimal_weights,
                'returns': portfolio_returns
            }
        except np.linalg.LinAlgError:
            # Fallback to equal weights if covariance matrix is singular
            return self.calculate_index_benchmark(returns_data)
    
    def calculate_minimum_variance(self, returns_data: np.ndarray) -> Dict:
        """Calculate minimum variance portfolio"""
        # Covariance matrix
        cov_matrix = np.cov(returns_data.T)
        
        try:
            # Minimum variance weights
            n_assets = returns_data.shape[1]
            ones = np.ones(n_assets)
            inv_cov = np.linalg.inv(cov_matrix)
            
            # Minimum variance portfolio weights
            min_var_weights = inv_cov @ ones
            min_var_weights = min_var_weights / np.sum(min_var_weights)
            
            # Calculate portfolio returns
            portfolio_returns = returns_data @ min_var_weights
            cumulative_return = np.prod(1 + portfolio_returns) - 1
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
            
            return {
                'type': 'Minimum Variance',
                'cumulative_return': cumulative_return,
                'sharpe_ratio': sharpe_ratio,
                'volatility': np.std(portfolio_returns),
                'weights': min_var_weights,
                'returns': portfolio_returns
            }
        except np.linalg.LinAlgError:
            # Fallback to equal weights
            return self.calculate_index_benchmark(returns_data)
    
    def calculate_all_benchmarks(self, returns_data: np.ndarray) -> Dict:
        """Calculate all benchmarks"""
        benchmarks = {}
        
        benchmarks['index'] = self.calculate_index_benchmark(returns_data)
        benchmarks['sharpe_optimal'] = self.calculate_sharpe_optimal(returns_data)
        benchmarks['min_variance'] = self.calculate_minimum_variance(returns_data)
        
        return benchmarks 