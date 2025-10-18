"""
Correspondence Mapping Implementation

Implements the correspondence mapping functionality that tracks individual
solutions across time periods, enabling anticipatory learning in decision space.

This module provides the core functionality for maintaining historical
populations and tracking solution evolution over time.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CorrespondenceMapping:
    """
    Correspondence mapping for tracking individual solutions across time.
    
    This class maintains historical populations and provides methods to
    track how individual solutions evolve over time, enabling anticipatory
    learning in decision space.
    """
    
    def __init__(self, max_history_size: int = 50):
        """
        Initialize correspondence mapping.
        
        Args:
            max_history_size: Maximum number of historical populations to keep
        """
        self.max_history_size = max_history_size
        self.historical_populations: List[List[Any]] = []  # List of populations over time
        self.historical_anticipative_decisions: List[Any] = []  # Historical anticipative decisions
        self.predicted_anticipative_decision: Optional[Any] = None
        
        # Tracking metadata
        self.population_metadata: List[Dict[str, Any]] = []
        
    def store_population(self, population: List[Any], current_time: int, 
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Store a population snapshot for correspondence mapping.
        
        Args:
            population: Current population of solutions
            current_time: Current time step
            metadata: Optional metadata about the population
        """
        # Create a deep copy of the population for historical tracking
        historical_population = []
        for solution in population:
            # Create a copy of the solution
            historical_solution = self._copy_solution(solution)
            historical_population.append(historical_solution)
        
        # Store the population
        self.historical_populations.append(historical_population)
        
        # Store metadata
        population_meta = {
            'timestamp': datetime.now().isoformat(),
            'current_time': current_time,
            'population_size': len(population),
            'metadata': metadata or {}
        }
        self.population_metadata.append(population_meta)
        
        # Maintain maximum history size
        if len(self.historical_populations) > self.max_history_size:
            self.historical_populations.pop(0)
            self.population_metadata.pop(0)
        
        logger.debug(f"Stored population at time {current_time} with {len(population)} solutions")
    
    def get_historical_solution(self, solution_index: int, time_step: int) -> Optional[Any]:
        """
        Get a specific solution from historical populations.
        
        Args:
            solution_index: Index of the solution in the population
            time_step: Time step to retrieve from
            
        Returns:
            Historical solution or None if not found
        """
        if time_step < 0 or time_step >= len(self.historical_populations):
            return None
        
        population = self.historical_populations[time_step]
        if solution_index < 0 or solution_index >= len(population):
            return None
        
        return population[solution_index]
    
    def track_solution_evolution(self, solution_index: int, start_time: int, 
                               end_time: int) -> List[Any]:
        """
        Track the evolution of a specific solution across time steps.
        
        Args:
            solution_index: Index of the solution to track
            start_time: Starting time step
            end_time: Ending time step
            
        Returns:
            List of solutions representing the evolution
        """
        evolution = []
        
        for t in range(start_time, min(end_time + 1, len(self.historical_populations))):
            solution = self.get_historical_solution(solution_index, t)
            if solution is not None:
                evolution.append(solution)
        
        return evolution
    
    def find_corresponding_solution(self, target_solution: Any, target_time: int,
                                  search_time: int, similarity_threshold: float = 0.95) -> Optional[Any]:
        """
        Find the solution in a different time step that corresponds to the target solution.
        
        This uses portfolio weight similarity to find corresponding solutions.
        
        Args:
            target_solution: Solution to find correspondence for
            target_time: Time step of the target solution
            search_time: Time step to search in
            similarity_threshold: Minimum similarity threshold for correspondence
            
        Returns:
            Corresponding solution or None if not found
        """
        if (target_time < 0 or target_time >= len(self.historical_populations) or
            search_time < 0 or search_time >= len(self.historical_populations)):
            return None
        
        target_weights = target_solution.P.investment
        search_population = self.historical_populations[search_time]
        
        best_similarity = 0.0
        best_solution = None
        
        for solution in search_population:
            search_weights = solution.P.investment
            
            # Calculate cosine similarity between portfolio weights
            similarity = self._calculate_weight_similarity(target_weights, search_weights)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_solution = solution
        
        return best_solution
    
    def _calculate_weight_similarity(self, weights1: np.ndarray, weights2: np.ndarray) -> float:
        """
        Calculate similarity between two portfolio weight vectors.
        
        Uses cosine similarity for robustness to scaling.
        
        Args:
            weights1: First weight vector
            weights2: Second weight vector
            
        Returns:
            Similarity score between 0 and 1
        """
        # Ensure vectors are normalized
        w1_norm = weights1 / (np.linalg.norm(weights1) + 1e-10)
        w2_norm = weights2 / (np.linalg.norm(weights2) + 1e-10)
        
        # Calculate cosine similarity
        cosine_sim = np.dot(w1_norm, w2_norm)
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, cosine_sim))
    
    def _copy_solution(self, solution: Any) -> Any:
        """
        Create a deep copy of a solution for historical storage.
        
        Args:
            solution: Solution to copy
            
        Returns:
            Deep copy of the solution
        """
        # This is a simplified copy - in practice, you'd want a proper deep copy
        # that preserves all attributes and nested objects
        try:
            # Create a new solution with the same number of assets
            from .solution import Solution
            copied_solution = Solution(solution.P.num_assets)
            
            # Copy portfolio weights
            copied_solution.P.investment = solution.P.investment.copy()
            copied_solution.P.ROI = solution.P.ROI
            copied_solution.P.risk = solution.P.risk
            copied_solution.P.cardinality = solution.P.cardinality
            
            # Copy solution attributes
            copied_solution.cd = solution.cd
            copied_solution.Delta_S = solution.Delta_S
            copied_solution.Pareto_rank = solution.Pareto_rank
            copied_solution.stability = solution.stability
            copied_solution.rank_ROI = solution.rank_ROI
            copied_solution.rank_risk = solution.rank_risk
            copied_solution.alpha = solution.alpha
            copied_solution.anticipation = solution.anticipation
            copied_solution.prediction_error = solution.prediction_error
            
            # Copy Kalman state if available
            if hasattr(solution.P, 'kalman_state') and solution.P.kalman_state is not None:
                # This would need proper Kalman state copying
                copied_solution.P.kalman_state = solution.P.kalman_state
            
            return copied_solution
            
        except Exception as e:
            logger.warning(f"Failed to copy solution: {e}")
            return solution
    
    def store_anticipative_decision(self, solution: Any, current_time: int):
        """
        Store an anticipative decision for historical tracking.
        
        Args:
            solution: Anticipative decision solution
            current_time: Current time step
        """
        anticipative_copy = self._copy_solution(solution)
        self.historical_anticipative_decisions.append(anticipative_copy)
        
        logger.debug(f"Stored anticipative decision at time {current_time}")
    
    def get_anticipative_decision_history(self) -> List[Any]:
        """
        Get the history of anticipative decisions.
        
        Returns:
            List of historical anticipative decisions
        """
        return self.historical_anticipative_decisions.copy()
    
    def set_predicted_anticipative_decision(self, solution: Any):
        """
        Set the predicted anticipative decision.
        
        Args:
            solution: Predicted anticipative decision
        """
        self.predicted_anticipative_decision = self._copy_solution(solution)
    
    def get_predicted_anticipative_decision(self) -> Optional[Any]:
        """
        Get the predicted anticipative decision.
        
        Returns:
            Predicted anticipative decision or None
        """
        return self.predicted_anticipative_decision
    
    def get_population_statistics(self, time_step: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific population at a given time step.
        
        Args:
            time_step: Time step to get statistics for
            
        Returns:
            Population statistics or None if not found
        """
        if time_step < 0 or time_step >= len(self.historical_populations):
            return None
        
        population = self.historical_populations[time_step]
        if not population:
            return None
        
        # Calculate statistics
        rois = [sol.P.ROI for sol in population]
        risks = [sol.P.risk for sol in population]
        alphas = [sol.alpha for sol in population]
        prediction_errors = [sol.prediction_error for sol in population]
        
        stats = {
            'time_step': time_step,
            'population_size': len(population),
            'roi_mean': np.mean(rois),
            'roi_std': np.std(rois),
            'roi_min': np.min(rois),
            'roi_max': np.max(rois),
            'risk_mean': np.mean(risks),
            'risk_std': np.std(risks),
            'risk_min': np.min(risks),
            'risk_max': np.max(risks),
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
            'prediction_error_mean': np.mean(prediction_errors),
            'prediction_error_std': np.std(prediction_errors),
            'metadata': self.population_metadata[time_step] if time_step < len(self.population_metadata) else {}
        }
        
        return stats
    
    def get_evolution_statistics(self, solution_index: int, start_time: int, 
                               end_time: int) -> Optional[Dict[str, Any]]:
        """
        Get statistics for the evolution of a specific solution.
        
        Args:
            solution_index: Index of the solution to analyze
            start_time: Starting time step
            end_time: Ending time step
            
        Returns:
            Evolution statistics or None if not found
        """
        evolution = self.track_solution_evolution(solution_index, start_time, end_time)
        if not evolution:
            return None
        
        # Calculate evolution statistics
        rois = [sol.P.ROI for sol in evolution]
        risks = [sol.P.risk for sol in evolution]
        alphas = [sol.alpha for sol in evolution]
        prediction_errors = [sol.prediction_error for sol in evolution]
        
        # Calculate trends
        roi_trend = np.polyfit(range(len(rois)), rois, 1)[0] if len(rois) > 1 else 0.0
        risk_trend = np.polyfit(range(len(risks)), risks, 1)[0] if len(risks) > 1 else 0.0
        alpha_trend = np.polyfit(range(len(alphas)), alphas, 1)[0] if len(alphas) > 1 else 0.0
        
        stats = {
            'solution_index': solution_index,
            'start_time': start_time,
            'end_time': end_time,
            'evolution_length': len(evolution),
            'roi_evolution': rois,
            'risk_evolution': risks,
            'alpha_evolution': alphas,
            'prediction_error_evolution': prediction_errors,
            'roi_trend': roi_trend,
            'risk_trend': risk_trend,
            'alpha_trend': alpha_trend,
            'roi_volatility': np.std(rois),
            'risk_volatility': np.std(risks),
            'alpha_volatility': np.std(alphas)
        }
        
        return stats
    
    def clear_history(self):
        """Clear all historical data."""
        self.historical_populations.clear()
        self.historical_anticipative_decisions.clear()
        self.population_metadata.clear()
        self.predicted_anticipative_decision = None
        
        logger.info("Cleared all correspondence mapping history")
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the correspondence mapping history.
        
        Returns:
            Summary of historical data
        """
        return {
            'num_historical_populations': len(self.historical_populations),
            'num_anticipative_decisions': len(self.historical_anticipative_decisions),
            'has_predicted_decision': self.predicted_anticipative_decision is not None,
            'max_history_size': self.max_history_size,
            'time_range': {
                'start': self.population_metadata[0]['current_time'] if self.population_metadata else None,
                'end': self.population_metadata[-1]['current_time'] if self.population_metadata else None
            }
        }



