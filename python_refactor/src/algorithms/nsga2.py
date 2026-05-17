"""
NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.

This module implements the NSGA-II algorithm for multi-objective optimization
of portfolio selection problems.
"""

import numpy as np
from typing import List, Optional
from .solution import Solution
from .operators import create_offspring_population


def fast_non_dominated_sort(population: List[Solution]) -> List[List[Solution]]:
    """
    Perform fast non-dominated sorting to assign Pareto ranks.
    
    Args:
        population: List of solutions
    
    Returns:
        List of fronts, where each front is a list of solutions
    """
    fronts = [[]]  # Initialize first front
    
    # For each solution, calculate domination count and dominated solutions
    domination_count = {}  # Number of solutions that dominate this solution
    dominated_solutions = {}  # Solutions that this solution dominates
    
    for p in population:
        domination_count[p] = 0
        dominated_solutions[p] = []
        
        for q in population:
            if p != q:
                if p.dominates_with_constraints(q):
                    dominated_solutions[p].append(q)
                elif q.dominates_with_constraints(p):
                    domination_count[p] += 1
        
        # If no solution dominates p, it belongs to the first front
        if domination_count[p] == 0:
            p.Pareto_rank = 0
            fronts[0].append(p)
    
    # Generate subsequent fronts
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    q.Pareto_rank = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
    
    return fronts


def calculate_crowding_distance(front: List[Solution]) -> None:
    """
    Calculate crowding distance for solutions in a front.
    
    Args:
        front: List of solutions in the same Pareto front
    """
    if len(front) <= 2:
        # If front has 1 or 2 solutions, set infinite crowding distance
        for solution in front:
            solution.cd = float('inf')
        return
    
    # Initialize crowding distance
    for solution in front:
        solution.cd = 0.0
    
    # Calculate crowding distance for each objective
    objectives = ['ROI', 'risk']
    
    for objective in objectives:
        # Sort front by objective
        if objective == 'ROI':
            front.sort(key=lambda x: x.P.ROI, reverse=True)
        else:  # risk
            front.sort(key=lambda x: x.P.risk)
        
        # Set infinite crowding distance for boundary solutions
        front[0].cd = float('inf')
        front[-1].cd = float('inf')
        
        # Calculate crowding distance for intermediate solutions
        obj_range = front[-1].P.ROI - front[0].P.ROI if objective == 'ROI' else front[-1].P.risk - front[0].P.risk
        
        if obj_range == 0:
            continue
        
        for i in range(1, len(front) - 1):
            if objective == 'ROI':
                prev_obj = front[i-1].P.ROI
                next_obj = front[i+1].P.ROI
                current_obj = front[i].P.ROI
            else:
                prev_obj = front[i-1].P.risk
                next_obj = front[i+1].P.risk
                current_obj = front[i].P.risk
            
            front[i].cd += (next_obj - prev_obj) / obj_range


def sort_by_crowding_distance(front: List[Solution]) -> None:
    """
    Sort solutions in a front by crowding distance (descending).
    
    Args:
        front: List of solutions in the same Pareto front
    """
    front.sort(key=lambda x: x.cd, reverse=True)


def select_next_generation(parent_population: List[Solution], 
                          offspring_population: List[Solution],
                          population_size: int) -> List[Solution]:
    """
    Select the next generation using NSGA-II selection mechanism.
    
    Args:
        parent_population: Parent population
        offspring_population: Offspring population
        population_size: Size of the population
    
    Returns:
        Selected population for next generation
    """
    # Combine parent and offspring populations
    combined_population = parent_population + offspring_population
    
    # Perform fast non-dominated sorting
    fronts = fast_non_dominated_sort(combined_population)
    
    # Select solutions from fronts until population is filled
    next_generation = []
    front_index = 0
    
    while len(next_generation) + len(fronts[front_index]) <= population_size:
        # Add entire front
        next_generation.extend(fronts[front_index])
        front_index += 1
        
        if front_index >= len(fronts):
            break
    
    # If we need more solutions, select from the next front using crowding distance
    if len(next_generation) < population_size and front_index < len(fronts):
        remaining_slots = population_size - len(next_generation)
        current_front = fronts[front_index]
        
        # Calculate crowding distance for the current front
        calculate_crowding_distance(current_front)
        
        # Sort by crowding distance and select the best
        sort_by_crowding_distance(current_front)
        next_generation.extend(current_front[:remaining_slots])
    
    return next_generation


def run_nsga2_generation(population: List[Solution],
                        population_size: int,
                        mutation_rate: float = 0.1,
                        crossover_rate: float = 0.9,
                        current_time: int = -1) -> List[Solution]:
    """
    Run one generation of NSGA-II.
    
    Args:
        population: Current population
        population_size: Size of the population
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        current_time: Current time step for anticipatory learning
    
    Returns:
        New population after one generation
    """
    # Apply anticipatory learning if time step is provided
    if current_time >= 0:
        from .anticipatory_learning import apply_anticipatory_learning_to_algorithm
        apply_anticipatory_learning_to_algorithm(population, current_time, 'nsga2')
    
    # Create offspring population
    offspring_population = create_offspring_population(
        population, population_size, crossover_rate, mutation_rate
    )
    
    # Select next generation
    next_generation = select_next_generation(
        population, offspring_population, population_size
    )
    
    return next_generation


def run_nsga2(num_generations: int,
              population_size: int,
              num_assets: int,
              mutation_rate: float = 0.1,
              crossover_rate: float = 0.9,
              random_seed: Optional[int] = None) -> List[Solution]:
    """
    Run NSGA-II algorithm for portfolio optimization.
    
    Args:
        num_generations: Number of generations to run
        population_size: Size of the population
        num_assets: Number of assets in the portfolio
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        random_seed: Random seed for reproducibility
    
    Returns:
        Final population
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initialize population
    population = [Solution(num_assets) for _ in range(population_size)]
    
    # Run generations
    for generation in range(num_generations):
        population = run_nsga2_generation(
            population, population_size, mutation_rate, crossover_rate, generation
        )
        
        # Optional: Print progress
        if generation % 10 == 0:
            pareto_front = [s for s in population if s.Pareto_rank == 0]
            print(f"Generation {generation}: Pareto front size = {len(pareto_front)}")
    
    return population


def get_pareto_front(population: List[Solution]) -> List[Solution]:
    """
    Extract the Pareto front from a population.
    
    Args:
        population: Population of solutions
    
    Returns:
        Solutions in the Pareto front
    """
    return [solution for solution in population if solution.Pareto_rank == 0]


def evaluate_population_statistics(population: List[Solution]) -> dict:
    """
    Evaluate statistics of the population.
    
    Args:
        population: Population of solutions
    
    Returns:
        Dictionary containing population statistics
    """
    if not population:
        return {}
    
    # Calculate statistics
    rois = [s.P.ROI for s in population]
    risks = [s.P.risk for s in population]
    cardinalities = [s.P.cardinality for s in population]
    pareto_ranks = [s.Pareto_rank for s in population]
    crowding_distances = [s.cd for s in population]
    
    stats = {
        'population_size': len(population),
        'pareto_front_size': len(get_pareto_front(population)),
        'roi_mean': np.mean(rois),
        'roi_std': np.std(rois),
        'roi_min': np.min(rois),
        'roi_max': np.max(rois),
        'risk_mean': np.mean(risks),
        'risk_std': np.std(risks),
        'risk_min': np.min(risks),
        'risk_max': np.max(risks),
        'cardinality_mean': np.mean(cardinalities),
        'cardinality_std': np.std(cardinalities),
        'max_pareto_rank': np.max(pareto_ranks),
        'crowding_distance_mean': np.mean(crowding_distances),
        'crowding_distance_std': np.std(crowding_distances)
    }
    
    return stats 