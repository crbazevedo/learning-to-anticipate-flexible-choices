"""
SMS-EMOA (S-Metric Selection Evolutionary Multi-objective Optimization Algorithm)

This module implements the SMS-EMOA algorithm for multi-objective portfolio optimization.
The algorithm uses hypervolume-based selection to maintain diversity in the Pareto front.
"""

import numpy as np
from typing import List, Tuple, Optional
from .solution import Solution
from .operators import crossover, mutation, tournament_selection
from ..portfolio.portfolio import Portfolio


def compute_hypervolume(solutions: List[Solution], reference_point: Tuple[float, float]) -> float:
    """
    Compute the hypervolume indicator for a set of solutions.
    
    Args:
        solutions: List of solutions to evaluate
        reference_point: Reference point (rx, ry) for hypervolume calculation
        
    Returns:
        Hypervolume value
    """
    if not solutions:
        return 0.0
    
    # Filter Pareto front solutions (rank 0)
    pareto_solutions = [s for s in solutions if s.Pareto_rank == 0]
    
    if not pareto_solutions:
        return 0.0
    
    # Sort by ROI (first objective)
    pareto_solutions.sort(key=lambda x: x.P.ROI, reverse=True)
    
    rx, ry = reference_point
    hypervolume = 0.0
    
    # Calculate hypervolume contribution of each solution
    for i, solution in enumerate(pareto_solutions):
        if i == 0:
            # First solution: area from solution to reference point
            width = solution.P.ROI - rx
            height = solution.P.risk - ry
        else:
            # Subsequent solutions: area between current and previous solution
            prev_solution = pareto_solutions[i-1]
            width = solution.P.ROI - prev_solution.P.ROI
            height = solution.P.risk - ry
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
    return hypervolume


def compute_delta_s_contribution(solutions: List[Solution], reference_point: Tuple[float, float]) -> None:
    """
    Compute the Delta-S (hypervolume contribution) for each solution.
    
    Args:
        solutions: List of solutions to evaluate
        reference_point: Reference point for hypervolume calculation
    """
    if not solutions:
        return
    
    # Group solutions by Pareto rank
    pareto_ranks = {}
    for solution in solutions:
        rank = solution.Pareto_rank
        if rank not in pareto_ranks:
            pareto_ranks[rank] = []
        pareto_ranks[rank].append(solution)
    
    # Compute Delta-S for each Pareto rank
    for rank in sorted(pareto_ranks.keys()):
        compute_delta_s_class(pareto_ranks[rank], reference_point)


def compute_delta_s_class(solutions: List[Solution], reference_point: Tuple[float, float]) -> None:
    """
    Compute Delta-S contribution for solutions in the same Pareto rank.
    
    Args:
        solutions: List of solutions in the same Pareto rank
        reference_point: Reference point for hypervolume calculation
    """
    if len(solutions) <= 1:
        if solutions:
            solutions[0].Delta_S = float('inf')
        return
    
    # Sort by ROI (first objective)
    solutions.sort(key=lambda x: x.P.ROI, reverse=True)
    
    rx, ry = reference_point
    
    # Compute Delta-S for each solution
    for i, solution in enumerate(solutions):
        if i == 0:
            # First solution: contribution from solution to reference point
            width = solution.P.ROI - rx
            height = solution.P.risk - ry
        elif i == len(solutions) - 1:
            # Last solution: contribution from solution to previous solution
            prev_solution = solutions[i-1]
            width = prev_solution.P.ROI - solution.P.ROI
            height = solution.P.risk - ry
        else:
            # Middle solution: contribution between adjacent solutions
            prev_solution = solutions[i-1]
            next_solution = solutions[i+1]
            width = prev_solution.P.ROI - next_solution.P.ROI
            height = solution.P.risk - ry
        
        solution.Delta_S = max(0.0, width * height)


def remove_worst_s_metric(solutions: List[Solution], reference_point: Tuple[float, float]) -> None:
    """
    Remove the solution with the worst S-metric (lowest Delta-S) contribution.
    
    Args:
        solutions: List of solutions (will be modified)
        reference_point: Reference point for hypervolume calculation
    """
    if len(solutions) <= 1:
        return
    
    # Compute Delta-S contributions
    compute_delta_s_contribution(solutions, reference_point)
    
    # Find solution with minimum Delta-S
    worst_solution = min(solutions, key=lambda x: x.Delta_S)
    
    # Remove the worst solution
    solutions.remove(worst_solution)


def run_sms_emoa_generation(population: List[Solution], 
                           mutation_rate: float = 0.3,
                           tournament_size: int = 2,
                           reference_point: Tuple[float, float] = (-1.0, 10.0)) -> None:
    """
    Run one generation of SMS-EMOA algorithm.
    
    Args:
        population: Current population of solutions
        mutation_rate: Probability of mutation
        tournament_size: Size of tournament for selection
        reference_point: Reference point for hypervolume calculation
    """
    # Perform fast non-dominated sorting
    from .nsga2 import fast_non_dominated_sort
    num_classes = fast_non_dominated_sort(population)
    
    # Compute Delta-S contributions
    compute_delta_s_contribution(population, reference_point)
    
    # Tournament selection for parent selection
    parent1_idx = tournament_selection(population, tournament_size, selection_type='delta_s')
    parent2_idx = tournament_selection(population, tournament_size, selection_type='delta_s')
    
    # Ensure different parents
    while parent2_idx == parent1_idx:
        parent2_idx = tournament_selection(population, tournament_size, selection_type='delta_s')
    
    # Create offspring through crossover
    offspring1, offspring2 = crossover(population[parent1_idx], population[parent2_idx])
    
    # Apply mutation to offspring1
    mutation(offspring1, mutation_rate)
    
    # Recompute efficiency metrics for offspring
    if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
        if Portfolio.robustness:
            Portfolio.compute_robust_efficiency(offspring1.P)
        else:
            Portfolio.compute_efficiency(offspring1.P)
    
    # Evaluate stability
    offspring1.stability = Portfolio.evaluate_stability(offspring1.P)
    
    # Add offspring to population
    population.append(offspring1)
    
    # Perform non-dominated sorting on extended population
    num_classes = fast_non_dominated_sort(population)
    
    # Compute Delta-S contributions
    compute_delta_s_contribution(population, reference_point)
    
    # Remove worst solution based on S-metric
    remove_worst_s_metric(population, reference_point)


def run_sms_emoa(initial_population: List[Solution],
                 generations: int = 50,
                 mutation_rate: float = 0.3,
                 tournament_size: int = 2,
                 reference_point: Tuple[float, float] = (-1.0, 10.0)) -> List[Solution]:
    """
    Run the complete SMS-EMOA algorithm.
    
    Args:
        initial_population: Initial population of solutions
        generations: Number of generations to run
        mutation_rate: Probability of mutation
        tournament_size: Size of tournament for selection
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Final population after optimization
    """
    population = initial_population.copy()
    
    for generation in range(generations):
        run_sms_emoa_generation(population, mutation_rate, tournament_size, reference_point)
        
        # Optional: Print progress
        if generation % 10 == 0:
            pareto_front = [s for s in population if s.Pareto_rank == 0]
            hypervolume = compute_hypervolume(population, reference_point)
            print(f"Generation {generation}: Pareto front size = {len(pareto_front)}, "
                  f"Hypervolume = {hypervolume:.6f}")
    
    return population


def get_sms_emoa_pareto_front(population: List[Solution]) -> List[Solution]:
    """
    Get the Pareto front from SMS-EMOA population.
    
    Args:
        population: Population of solutions
        
    Returns:
        List of solutions in the Pareto front (rank 0)
    """
    return [solution for solution in population if solution.Pareto_rank == 0]


def evaluate_sms_emoa_statistics(population: List[Solution], 
                                reference_point: Tuple[float, float] = (-1.0, 10.0)) -> dict:
    """
    Evaluate statistics for SMS-EMOA population.
    
    Args:
        population: Population of solutions
        reference_point: Reference point for hypervolume calculation
        
    Returns:
        Dictionary containing population statistics
    """
    if not population:
        return {}
    
    pareto_front = get_sms_emoa_pareto_front(population)
    
    # Basic statistics
    stats = {
        'population_size': len(population),
        'pareto_front_size': len(pareto_front),
        'hypervolume': compute_hypervolume(population, reference_point),
        'mean_pareto_rank': np.mean([s.Pareto_rank for s in population]),
        'mean_delta_s': np.mean([s.Delta_S for s in population if s.Delta_S != float('inf')]),
        'mean_roi': np.mean([s.P.ROI for s in population]),
        'mean_risk': np.mean([s.P.risk for s in population]),
        'mean_cardinality': np.mean([s.P.cardinality for s in population]),
        'mean_stability': np.mean([s.stability for s in population])
    }
    
    # Robust statistics if available
    if hasattr(population[0].P, 'robust_ROI'):
        stats.update({
            'mean_robust_roi': np.mean([s.P.robust_ROI for s in population]),
            'mean_robust_risk': np.mean([s.P.robust_risk for s in population])
        })
    
    return stats 