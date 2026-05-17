"""
SMS-EMOA (S-metric Selection Evolutionary Multi-Objective Algorithm)

Revised implementation to include:
- Stochastic Pareto frontiers over time
- Dirichlet MAP filtering for portfolios
- Expected future hypervolume metrics
- Anticipatory learning integration
"""

import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from .solution import Solution
from .anticipatory_learning import AnticipatoryLearning

class StochasticParams:
    """Stochastic parameters for portfolio state tracking."""
    
    def __init__(self, solution: Solution):
        """
        Initialize stochastic parameters from solution's Kalman state.
        
        Args:
            solution: Solution with Kalman filter state
        """
        # Extract covariance matrix from Kalman state
        kalman_state = solution.P.kalman_state
        P = kalman_state.P
        
        # Extract parameters
        self.cov = P[0, 1]  # Covariance between ROI and risk
        self.var_ROI = P[0, 0]  # Variance of ROI
        self.var_risk = P[1, 1]  # Variance of risk
        
        # Compute correlation
        self.corr = self.cov / (np.sqrt(self.var_ROI) * np.sqrt(self.var_risk)) if self.var_ROI > 0 and self.var_risk > 0 else 0.0
        
        # Variance ratio
        self.var_ratio = np.sqrt(self.var_ROI) / np.sqrt(self.var_risk) if self.var_risk > 0 else 0.0
        
        # Conditional parameters (assuming independence)
        self.conditional_mean_ROI = solution.P.ROI
        self.conditional_var_ROI = (1.0 - self.corr**2) * self.var_ROI
        self.conditional_mean_risk = solution.P.risk
        self.conditional_var_risk = (1.0 - self.corr**2) * self.var_risk

class SMSEMOA:
    """SMS-EMOA with stochastic Pareto frontiers and anticipatory learning."""
    
    def __init__(self, population_size: int = 100, generations: int = 200,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1,
                 tournament_size: int = 3, reference_point_1: float = 0.0,
                 reference_point_2: float = 1.0):
        """
        Initialize SMS-EMOA algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            tournament_size: Tournament selection size
            reference_point_1: First reference point for hypervolume (ROI)
            reference_point_2: Second reference point for hypervolume (Risk)
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.R1 = reference_point_1  # ROI reference point
        self.R2 = reference_point_2  # Risk reference point
        
        # Anticipatory learning components
        self.anticipatory_learning = None
        
        # Population and Pareto front tracking
        self.population = []
        self.pareto_front = []
        self.hypervolume_history = []
        self.stochastic_hypervolume_history = []
        
        # Performance tracking
        self.function_evaluations = 0
        self.current_generation = 0
        
    def set_learning(self, learning: AnticipatoryLearning):
        """Set anticipatory learning component."""
        self.anticipatory_learning = learning
    
    def run(self, data: Dict[str, Any]) -> List[Solution]:
        """
        Run SMS-EMOA algorithm.
        
        Args:
            data: Market data dictionary
            
        Returns:
            Final population
        """
        # Initialize population
        self._initialize_population(data)
        
        # Main evolution loop
        for generation in range(self.generations):
            self.current_generation = generation
            
            # Apply anticipatory learning if enabled
            if self.anticipatory_learning is not None:
                self._apply_anticipatory_learning(generation)
            
            # Run one generation
            self._run_generation()
            
            # Track hypervolume
            self._track_hypervolume()
            
            # Log progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Population size = {len(self.population)}, "
                      f"Pareto front size = {len(self.pareto_front)}")
        
        return self.population
    
    def _initialize_population(self, data: Dict[str, Any]):
        """Initialize population with random solutions."""
        self.population = []
        
        # Get number of assets from data
        num_assets = data.get('num_assets', 3)  # Default to 3 if not specified
        
        for _ in range(self.population_size):
            solution = Solution(num_assets=num_assets)
            
            # Initialize Kalman filter state
            self._initialize_kalman_state(solution, data)
            
            # Evaluate solution
            self._evaluate_solution(solution, data)
            
            self.population.append(solution)
            self.function_evaluations += 1
    
    def _initialize_kalman_state(self, solution: Solution, data: Dict[str, Any]):
        """Initialize Kalman filter state for solution."""
        # Initialize with current portfolio state
        kalman_state = solution.P.kalman_state
        
        # State vector: [ROI, risk, ROI_velocity, risk_velocity]
        kalman_state.x = np.array([solution.P.ROI, solution.P.risk, 0.0, 0.0])
        
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
        
        # Measurement matrix (observe ROI and risk)
        kalman_state.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Measurement noise covariance
        kalman_state.R = np.array([
            [0.01, 0.0],
            [0.0, 0.01]
        ])
    
    def _evaluate_solution(self, solution: Solution, data: Dict[str, Any]):
        """Evaluate solution and update objectives."""
        # Compute portfolio metrics
        portfolio = solution.P
        
        # Update Kalman filter with current observation if available
        if hasattr(solution.P, 'kalman_state') and solution.P.kalman_state is not None:
            from .kalman_filter import kalman_update
            measurement = np.array([portfolio.ROI, portfolio.risk])
            kalman_update(solution.P.kalman_state, measurement)
        
        # Store objectives
        solution.objectives = [portfolio.ROI, portfolio.risk]
        
        # Compute stability (simplified)
        portfolio.stability = 1.0 / (1.0 + np.std(solution.P.investment))
    
    def _apply_anticipatory_learning(self, generation: int):
        """Apply anticipatory learning to population."""
        if self.anticipatory_learning is None:
            return
        
        # Apply to each solution that hasn't been learned yet
        for solution in self.population:
            if not hasattr(solution, 'anticipation') or not solution.anticipation:
                self.anticipatory_learning.learn_single_solution(solution, generation)
    
    def _run_generation(self):
        """Run one generation of SMS-EMOA."""
        # Fast non-dominated sorting
        pareto_ranks = self._fast_non_dominated_sort()
        
        # Compute hypervolume contributions
        if self.anticipatory_learning is not None:
            self._compute_stochastic_hypervolume_contributions()
        else:
            self._compute_hypervolume_contributions()
        
        # Tournament selection
        parent1_idx = self._tournament_selection()
        parent2_idx = self._tournament_selection()
        
        # Create offspring
        from .operators import crossover, mutation
        offspring1, offspring2 = crossover(
            self.population[parent1_idx],
            self.population[parent2_idx],
            self.crossover_rate
        )
        
        # Apply mutation
        mutation(offspring1, self.mutation_rate)
        mutation(offspring2, self.mutation_rate)
        
        # Add offspring to population
        self.population.append(offspring1)
        self.population.append(offspring2)
        
        # Remove worst solution based on hypervolume contribution
        self._remove_worst_solution()
        
        # Update Pareto front
        self._update_pareto_front()
    
    def _fast_non_dominated_sort(self) -> int:
        """Perform fast non-dominated sorting."""
        pareto_ranks = [0] * len(self.population)
        
        for i, solution1 in enumerate(self.population):
            for j, solution2 in enumerate(self.population):
                if i != j:
                    if self._dominates(solution1, solution2):
                        pareto_ranks[j] += 1
        
        # Assign ranks
        for i, solution in enumerate(self.population):
            solution.Pareto_rank = pareto_ranks[i]
        
        # Return number of fronts (unique ranks)
        return len(set(pareto_ranks))
    
    def _dominates(self, solution1: Solution, solution2: Solution) -> bool:
        """Check if solution1 dominates solution2."""
        # Get objectives from portfolio
        roi1, risk1 = solution1.P.ROI, solution1.P.risk
        roi2, risk2 = solution2.P.ROI, solution2.P.risk
        
        # Check if solution1 is better in at least one objective
        # and not worse in any objective
        better_in_one = False
        
        # Higher ROI is better
        if roi1 > roi2:
            better_in_one = True
        elif roi1 < roi2:
            return False
        
        # Lower risk is better
        if risk1 < risk2:
            better_in_one = True
        elif risk1 > risk2:
            return False
        
        return better_in_one
    
    def _compute_hypervolume_contributions(self):
        """Compute hypervolume contributions for each solution."""
        # Group solutions by Pareto rank
        pareto_classes = {}
        for solution in self.population:
            rank = solution.Pareto_rank
            if rank not in pareto_classes:
                pareto_classes[rank] = []
            pareto_classes[rank].append(solution)
        
        # Compute contributions for each class
        for rank, solutions in pareto_classes.items():
            self._compute_hypervolume_contributions_class(solutions)
    
    def _compute_stochastic_hypervolume_contributions(self):
        """Compute stochastic hypervolume contributions considering future uncertainty."""
        # Group solutions by Pareto rank
        pareto_classes = {}
        for solution in self.population:
            rank = solution.Pareto_rank
            if rank not in pareto_classes:
                pareto_classes[rank] = []
            pareto_classes[rank].append(solution)
        
        # Compute stochastic contributions for each class
        for rank, solutions in pareto_classes.items():
            self._compute_stochastic_hypervolume_contributions_class(solutions)
    
    def _compute_hypervolume_contributions_class(self, solutions: List[Solution]):
        """Compute hypervolume contributions for a Pareto class."""
        if len(solutions) == 1:
            # Single solution in class
            solution = solutions[0]
            solution.hypervolume_contribution = (solution.P.ROI - self.R1) * (self.R2 - solution.P.risk)
            return
        
        # Sort by ROI (ascending)
        solutions.sort(key=lambda s: s.P.ROI)
        
        # Compute contributions
        for i, solution in enumerate(solutions):
            if i == 0:
                # First solution
                next_solution = solutions[i + 1]
                solution.hypervolume_contribution = (solution.P.ROI - self.R1) * (self.R2 - solution.P.risk)
            elif i == len(solutions) - 1:
                # Last solution
                prev_solution = solutions[i - 1]
                solution.hypervolume_contribution = (solution.P.ROI - prev_solution.P.ROI) * (self.R2 - solution.P.risk)
            else:
                # Middle solution
                prev_solution = solutions[i - 1]
                next_solution = solutions[i + 1]
                solution.hypervolume_contribution = (solution.P.ROI - next_solution.P.ROI) * (prev_solution.P.risk - solution.P.risk)
            
            # Apply stability factor
            solution.hypervolume_contribution *= solution.stability
    
    def _compute_stochastic_hypervolume_contributions_class(self, solutions: List[Solution]):
        """Compute stochastic hypervolume contributions considering uncertainty."""
        if len(solutions) == 1:
            # Single solution with uncertainty
            solution = solutions[0]
            stoch_params = StochasticParams(solution)
            
            mean_delta_ROI = stoch_params.conditional_mean_ROI - self.R1
            mean_delta_risk = self.R2 - stoch_params.conditional_mean_risk
            var_delta_ROI = stoch_params.conditional_var_ROI
            var_delta_risk = stoch_params.conditional_var_risk
            
            # Expected hypervolume contribution
            solution.hypervolume_contribution = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI) / (var_delta_ROI + var_delta_risk)
            solution.hypervolume_contribution *= solution.stability
            return
        
        # Sort by ROI
        solutions.sort(key=lambda s: s.P.ROI)
        
        # Compute stochastic contributions
        for i, solution in enumerate(solutions):
            stoch_params = StochasticParams(solution)
            
            if i == 0:
                # First solution
                next_solution = solutions[i + 1]
                next_stoch_params = StochasticParams(next_solution)
                
                mean_delta_ROI = stoch_params.conditional_mean_ROI - next_stoch_params.conditional_mean_ROI
                mean_delta_risk = self.R2 - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI + next_stoch_params.conditional_var_ROI
                var_delta_risk = stoch_params.conditional_var_risk
                
            elif i == len(solutions) - 1:
                # Last solution
                prev_solution = solutions[i - 1]
                prev_stoch_params = StochasticParams(prev_solution)
                
                mean_delta_ROI = stoch_params.conditional_mean_ROI - self.R1
                mean_delta_risk = prev_stoch_params.conditional_mean_risk - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI
                var_delta_risk = prev_stoch_params.conditional_var_risk + stoch_params.conditional_var_risk
                
            else:
                # Middle solution
                prev_solution = solutions[i - 1]
                next_solution = solutions[i + 1]
                prev_stoch_params = StochasticParams(prev_solution)
                next_stoch_params = StochasticParams(next_solution)
                
                # Compute mean delta product
                mean_delta_ROI = stoch_params.conditional_mean_ROI - next_stoch_params.conditional_mean_ROI
                mean_delta_risk = prev_stoch_params.conditional_mean_risk - stoch_params.conditional_mean_risk
                var_delta_ROI = stoch_params.conditional_var_ROI + next_stoch_params.conditional_var_ROI
                var_delta_risk = prev_stoch_params.conditional_var_risk + stoch_params.conditional_var_risk
            
            # Expected hypervolume contribution
            solution.hypervolume_contribution = (mean_delta_ROI * var_delta_risk + mean_delta_risk * var_delta_ROI) / (var_delta_ROI + var_delta_risk)
            solution.hypervolume_contribution *= solution.stability
    
    def _tournament_selection(self) -> int:
        """Perform tournament selection based on hypervolume contribution."""
        # Select random individuals (handle case where tournament_size > population_size)
        tournament_size = min(self.tournament_size, len(self.population))
        indices = np.random.choice(len(self.population), tournament_size, replace=False)
        
        # Find the best based on hypervolume contribution
        best_idx = indices[0]
        best_contribution = self.population[best_idx].hypervolume_contribution
        
        for idx in indices[1:]:
            contribution = self.population[idx].hypervolume_contribution
            if contribution > best_contribution:
                best_contribution = contribution
                best_idx = idx
        
        return best_idx
    
    def _remove_worst_solution(self):
        """Remove the solution with the lowest hypervolume contribution."""
        # Remove solutions until population size is correct
        while len(self.population) > self.population_size:
            # Find worst solution
            worst_idx = 0
            worst_contribution = getattr(self.population[0], 'hypervolume_contribution', float('inf'))
            
            for i, solution in enumerate(self.population):
                contribution = getattr(solution, 'hypervolume_contribution', float('inf'))
                if contribution < worst_contribution:
                    worst_contribution = contribution
                    worst_idx = i
            
            # Remove worst solution
            self.population.pop(worst_idx)
    
    def _update_pareto_front(self):
        """Update the Pareto front."""
        self.pareto_front = [s for s in self.population if s.Pareto_rank == 0]
    
    def _track_hypervolume(self):
        """Track hypervolume over generations."""
        if self.pareto_front:
            # Compute current hypervolume
            hypervolume = self._compute_hypervolume()
            self.hypervolume_history.append(hypervolume)
            
            # Compute expected future hypervolume if using anticipatory learning
            if self.anticipatory_learning is not None:
                future_hypervolume = self._compute_expected_future_hypervolume()
                self.stochastic_hypervolume_history.append(future_hypervolume)
    
    def _compute_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front."""
        if not self.pareto_front:
            return 0.0
        
        # Sort by ROI
        sorted_front = sorted(self.pareto_front, key=lambda s: s.P.ROI)
        
        hypervolume = 0.0
        prev_roi = self.R1
        
        for solution in sorted_front:
            roi = solution.P.ROI
            risk = solution.P.risk
            hypervolume += (roi - prev_roi) * (self.R2 - risk)
            prev_roi = roi
        
        return hypervolume
    
    def _compute_expected_future_hypervolume(self) -> float:
        """Compute expected future hypervolume considering uncertainty."""
        if not self.pareto_front:
            return 0.0
        
        # Compute expected hypervolume using Monte Carlo sampling
        num_samples = 1000
        total_hypervolume = 0.0
        
        for _ in range(num_samples):
            # Sample future states for each solution
            future_front = []
            for solution in self.pareto_front:
                # Sample from Kalman filter prediction
                from .kalman_filter import kalman_prediction
                kalman_prediction(solution.P.kalman_state)
                # Paper Eq (11) ordering: x_next = [ROI, risk, ROI_vel, risk_vel]
                # Prior to W1-1 this read x_next[2] (= ROI_velocity, NOT risk)
                # under sms_emoa.py's own paper-canonical F matrix.
                future_roi = solution.P.kalman_state.x_next[0]
                future_risk = solution.P.kalman_state.x_next[1]
                
                # Create temporary solution for hypervolume computation
                temp_solution = Solution(num_assets=len(solution.P.investment))
                temp_solution.P.ROI = future_roi
                temp_solution.P.risk = future_risk
                future_front.append(temp_solution)
            
            # Compute hypervolume for this sample
            if future_front:
                sorted_front = sorted(future_front, key=lambda s: s.P.ROI)
                sample_hypervolume = 0.0
                prev_roi = self.R1
                
                for solution in sorted_front:
                    roi = solution.P.ROI
                    risk = solution.P.risk
                    sample_hypervolume += (roi - prev_roi) * (self.R2 - risk)
                    prev_roi = roi
                
                total_hypervolume += sample_hypervolume
        
        return total_hypervolume / num_samples
    
    def get_pareto_front(self) -> List[Solution]:
        """Get current Pareto front."""
        return self.pareto_front
    
    def get_hypervolume(self) -> float:
        """Get current hypervolume."""
        return self._compute_hypervolume()
    
    def get_expected_future_hypervolume(self) -> float:
        """Get expected future hypervolume."""
        return self._compute_expected_future_hypervolume()
    
    def get_function_evaluations(self) -> int:
        """Get number of function evaluations."""
        return self.function_evaluations 