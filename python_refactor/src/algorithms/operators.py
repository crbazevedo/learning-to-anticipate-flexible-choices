"""
Genetic operators for evolutionary algorithms.

This module implements crossover, mutation, and selection operators
used in the NSGA-II and SMS-EMOA algorithms.
"""

import numpy as np
from typing import List, Tuple
from .solution import Solution
from ..portfolio.portfolio import Portfolio


def crossover(parent1: Solution, parent2: Solution, crossover_rate: float = 0.9) -> Tuple[Solution, Solution]:
    """
    Perform crossover between two parent solutions.
    
    Args:
        parent1: First parent solution
        parent2: Second parent solution
        crossover_rate: Probability of crossover
    
    Returns:
        Tuple of two offspring solutions
    """
    if np.random.random() > crossover_rate:
        return parent1, parent2
    
    # Create offspring by copying parents
    offspring1 = Solution(parent1.P.num_assets)
    offspring2 = Solution(parent2.P.num_assets)
    
    # Perform SBX (Simulated Binary Crossover) on portfolio weights
    eta = 20  # Distribution index
    
    # Crossover weights
    weights1, weights2 = sbx_crossover(
        parent1.P.investment, 
        parent2.P.investment, 
        eta
    )
    
    # Normalize weights to sum to 1
    offspring1.P.investment = weights1 / np.sum(weights1)
    offspring2.P.investment = weights2 / np.sum(weights2)
    
    # Recompute efficiency metrics (only if data is available)
    if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
        if Portfolio.robustness:
            Portfolio.compute_robust_efficiency(offspring1.P)
            Portfolio.compute_robust_efficiency(offspring2.P)
        else:
            Portfolio.compute_efficiency(offspring1.P)
            Portfolio.compute_efficiency(offspring2.P)
    
    return offspring1, offspring2


def sbx_crossover(parent1_weights: np.ndarray, parent2_weights: np.ndarray, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulated Binary Crossover (SBX) for real-valued variables.
    
    Args:
        parent1_weights: Weights from first parent
        parent2_weights: Weights from second parent
        eta: Distribution index
    
    Returns:
        Tuple of offspring weights
    """
    offspring1 = np.copy(parent1_weights)
    offspring2 = np.copy(parent2_weights)
    
    for i in range(len(parent1_weights)):
        if np.random.random() < 0.5:
            # Perform SBX
            if abs(parent1_weights[i] - parent2_weights[i]) > 1e-14:
                if parent1_weights[i] < parent2_weights[i]:
                    y1, y2 = parent1_weights[i], parent2_weights[i]
                else:
                    y1, y2 = parent2_weights[i], parent1_weights[i]
                
                lb = 0.0  # Lower bound for weights
                ub = 1.0  # Upper bound for weights
                
                rand = np.random.random()
                beta = 1.0 + (2.0 * (y1 - lb) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta + 1)
                
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                
                c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                
                beta = 1.0 + (2.0 * (ub - y2) / (y2 - y1))
                alpha = 2.0 - beta ** -(eta + 1)
                
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                
                c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                
                if c1 < lb:
                    c1 = lb
                if c2 < lb:
                    c2 = lb
                if c1 > ub:
                    c1 = ub
                if c2 > ub:
                    c2 = ub
                
                if np.random.random() <= 0.5:
                    offspring1[i] = c2
                    offspring2[i] = c1
                else:
                    offspring1[i] = c1
                    offspring2[i] = c2
    
    return offspring1, offspring2


def mutation(solution: Solution, mutation_rate: float = 0.1, eta: float = 20) -> Solution:
    """
    Perform polynomial mutation on a solution.
    
    Args:
        solution: Solution to mutate
        mutation_rate: Probability of mutation per gene
        eta: Distribution index
    
    Returns:
        Mutated solution
    """
    mutated = Solution(solution.P.num_assets)
    mutated.P.investment = np.copy(solution.P.investment)
    
    for i in range(len(mutated.P.investment)):
        if np.random.random() < mutation_rate:
            # Perform polynomial mutation
            y = mutated.P.investment[i]
            lb = 0.0
            ub = 1.0
            
            delta1 = (y - lb) / (ub - lb)
            delta2 = (ub - y) / (ub - lb)
            
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1)
            
            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** mut_pow
            
            y = y + deltaq * (ub - lb)
            y = np.clip(y, lb, ub)
            mutated.P.investment[i] = y
    
    # Normalize weights
    mutated.P.investment = mutated.P.investment / np.sum(mutated.P.investment)
    
    # Recompute efficiency metrics (only if data is available)
    if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
        if Portfolio.robustness:
            Portfolio.compute_robust_efficiency(mutated.P)
        else:
            Portfolio.compute_efficiency(mutated.P)
    
    return mutated


def tournament_selection(population: List[Solution], tournament_size: int = 2, selection_type: str = 'crowding_distance') -> int:
    """
    Perform tournament selection.
    
    Args:
        population: List of solutions
        tournament_size: Size of tournament
        selection_type: Type of selection ('crowding_distance', 'delta_s', 'pareto_rank')
    
    Returns:
        Index of selected solution
    """
    if not population:
        raise ValueError("Population cannot be empty")
    
    # Randomly select tournament_size individuals
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament = [population[i] for i in tournament_indices]
    
    # Return the best individual based on selection type
    if selection_type == 'crowding_distance':
        # Best based on Pareto rank and crowding distance
        best_solution = min(tournament)
        return population.index(best_solution)
    elif selection_type == 'delta_s':
        # Best based on Pareto rank and Delta-S
        best_solution = min(tournament, key=lambda x: (x.Pareto_rank, -x.Delta_S))
        return population.index(best_solution)
    elif selection_type == 'pareto_rank':
        # Best based on Pareto rank only
        best_solution = min(tournament, key=lambda x: x.Pareto_rank)
        return population.index(best_solution)
    else:
        raise ValueError(f"Unknown selection type: {selection_type}")


def tournament_selection_solution(population: List[Solution], tournament_size: int = 2, selection_type: str = 'crowding_distance') -> Solution:
    """
    Perform tournament selection and return the solution object.
    
    Args:
        population: List of solutions
        tournament_size: Size of tournament
        selection_type: Type of selection ('crowding_distance', 'delta_s', 'pareto_rank')
    
    Returns:
        Selected solution
    """
    selected_idx = tournament_selection(population, tournament_size, selection_type)
    return population[selected_idx]


def binary_tournament_selection(population: List[Solution]) -> Solution:
    """
    Perform binary tournament selection.
    
    Args:
        population: List of solutions
    
    Returns:
        Selected solution
    """
    return tournament_selection_solution(population, tournament_size=2)


def rank_based_selection(population: List[Solution], num_parents: int) -> List[Solution]:
    """
    Perform rank-based selection.
    
    Args:
        population: List of solutions
        num_parents: Number of parents to select
    
    Returns:
        List of selected parents
    """
    # Sort population by Pareto rank and crowding distance
    sorted_population = sorted(population)
    
    # Calculate selection probabilities (linear ranking)
    n = len(sorted_population)
    selection_probs = np.zeros(n)
    
    for i in range(n):
        selection_probs[i] = (2 - 1.5) / n + 2 * (n - i - 1) * (1.5 - 1) / (n * (n - 1))
    
    # Select parents
    selected_indices = np.random.choice(n, num_parents, p=selection_probs)
    selected_parents = [sorted_population[i] for i in selected_indices]
    
    return selected_parents


def crowding_distance_selection(population: List[Solution], num_select: int) -> List[Solution]:
    """
    Select solutions based on crowding distance.
    
    Args:
        population: List of solutions
        num_select: Number of solutions to select
    
    Returns:
        List of selected solutions
    """
    # Sort by crowding distance (descending)
    sorted_population = sorted(population, key=lambda x: x.cd, reverse=True)
    
    return sorted_population[:num_select]


def pareto_rank_selection(population: List[Solution], num_select: int) -> List[Solution]:
    """
    Select solutions based on Pareto rank.
    
    Args:
        population: List of solutions
        num_select: Number of solutions to select
    
    Returns:
        List of selected solutions
    """
    # Sort by Pareto rank (ascending)
    sorted_population = sorted(population, key=lambda x: x.Pareto_rank)
    
    return sorted_population[:num_select]


def create_offspring_population(parent_population: List[Solution], 
                               population_size: int,
                               crossover_rate: float = 0.9,
                               mutation_rate: float = 0.1) -> List[Solution]:
    """
    Create offspring population using genetic operators.
    
    Args:
        parent_population: Parent population
        population_size: Size of offspring population
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
    
    Returns:
        Offspring population
    """
    offspring_population = []
    
    while len(offspring_population) < population_size:
        # Select parents
        parent1 = binary_tournament_selection(parent_population)
        parent2 = binary_tournament_selection(parent_population)
        
        # Perform crossover
        offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)
        
        # Perform mutation
        offspring1 = mutation(offspring1, mutation_rate)
        offspring2 = mutation(offspring2, mutation_rate)
        
        offspring_population.extend([offspring1, offspring2])
    
    # Trim to exact population size
    return offspring_population[:population_size] 