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

# ===========================================================================
# W15-2: Thesis-faithful operators per Azevedo PhD §7.2.3 (BACKLOG B3+B4)
# ===========================================================================
#
# The pre-W15-2 `crossover` (line 14) uses SBX. The thesis §7.2.3
# (p. 147) verbatim prescribes:
#   "We utilized uniform crossover over the mean DD vectors. For mutation,
#    we randomly choose between (1) modifying the non-zero weights; or
#    (2) adding/removing assets. If operator (1) is selected, then, with
#    probability 1/2, we either increase or decrease the investment on a
#    randomly chosen asset by a uniformly drawn factor from 10 to 50%.
#    If (2) is selected, then, with probability 1/2, we either add or
#    remove a randomly chosen asset. If it is removed, we simply set its
#    weight to zero. If it is added, we randomly set its weight within
#    a ±10% range from an equally-balanced allocation. All modified DD
#    vectors are renormalized."
#
# Plus thesis §7.2.3 (p. 146) cardinality: c_l = 5, c_u = 15.
# Plus thesis §7.2 (p. 141) simplex: u ∈ S^{N-1} (u_i ≥ 0, sum=1).
#
# Both the legacy `crossover` and `mutation` above are kept available
# for backward compatibility but should NOT be used for thesis-faithful
# experiments — use the thesis_uniform_crossover + thesis_dual_mode_mutation
# below.


# Thesis defaults from §7.2.3 p. 146
THESIS_CARDINALITY_MIN = 5
THESIS_CARDINALITY_MAX = 15


def project_to_simplex(weights: np.ndarray,
                        c_l: int = THESIS_CARDINALITY_MIN,
                        c_u: int = THESIS_CARDINALITY_MAX,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Project a weight vector onto the cardinality-constrained simplex.

    Steps (in order, per thesis §7.2 Eq 7.3 + §7.2.3 Constraint Handling):
      1. Clip negatives to 0 (simplex non-negativity per p. 141)
      2. Enforce cardinality upper bound: if `cardinality > c_u`, zero
         out the smallest non-zero weights until cardinality == c_u
      3. Enforce cardinality lower bound: if `cardinality < c_l`,
         randomly activate `c_l - cardinality` zero-weight assets at
         ±10% of equally-balanced allocation (per thesis "Search
         Operators" §7.2.3 p. 147 wording for the add-asset mutation)
      4. Renormalize to sum = 1

    Grounding:
      - Thesis §7.2 p. 141: "u ∈ S^{N-1} denote the proportions of
        wealth to be invested" → u_i ≥ 0, sum=1
      - Thesis §7.2 Eq (7.3) p. 142: "s.t. c_l ≤ c(u_t) ≤ c_u, where
        c(u_t) computes the number of assets in u_t with non-zero weight"
      - Thesis §7.2.3 p. 146: "We considered minimum and maximum
        cardinality of c_l = 5 and c_u = 15 assets."
    """
    if rng is None:
        rng = np.random.default_rng()
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    # Active asset count
    active_mask = w > 1e-12
    n_active = int(active_mask.sum())
    n_assets = len(w)
    # Cap to c_u (zero out smallest non-zero weights)
    if n_active > c_u:
        # Sort non-zero indices by weight ascending; zero out the smallest
        # (n_active - c_u) of them.
        nonzero_idx = np.flatnonzero(active_mask)
        sorted_idx = nonzero_idx[np.argsort(w[nonzero_idx])]
        to_zero = sorted_idx[: n_active - c_u]
        w[to_zero] = 0.0
        active_mask = w > 1e-12
        n_active = int(active_mask.sum())
    # Fill to c_l (add small weights to inactive assets)
    if n_active < c_l and n_assets >= c_l:
        inactive_idx = np.flatnonzero(~active_mask)
        n_to_add = c_l - n_active
        # Sample without replacement
        n_to_add = min(n_to_add, len(inactive_idx))
        chosen = rng.choice(inactive_idx, size=n_to_add, replace=False)
        # ±10% of equal allocation per thesis add-asset mutation rule
        equal_alloc = 1.0 / c_l
        for idx in chosen:
            jitter = rng.uniform(-0.10, 0.10) * equal_alloc
            w[idx] = max(equal_alloc + jitter, 1e-9)
    # Renormalize to sum=1
    total = w.sum()
    if total <= 0:
        # Degenerate fallback: equal allocation over c_l first assets
        w = np.zeros(n_assets)
        idx = rng.choice(n_assets, size=min(c_l, n_assets), replace=False)
        w[idx] = 1.0 / len(idx)
    else:
        w = w / total
    return w


def thesis_uniform_crossover(parent1: Solution, parent2: Solution,
                              p: float = 0.5,
                              c_l: int = THESIS_CARDINALITY_MIN,
                              c_u: int = THESIS_CARDINALITY_MAX,
                              rng: np.random.Generator | None = None
                              ) -> Tuple[Solution, Solution]:
    """Uniform crossover over mean DD (decision) vectors per thesis §7.2.3 p.147.

    For each asset i: with probability `p`, child1 takes parent2's
    weight at i (and child2 takes parent1's); otherwise no swap.
    Renormalize + project to cardinality-constrained simplex.

    Grounding (thesis §7.2.3 p. 147):
      "We utilized uniform crossover over the mean DD vectors."
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(parent1.P.investment)
    swap_mask = rng.random(n) < p
    w1 = parent1.P.investment.copy()
    w2 = parent2.P.investment.copy()
    w1[swap_mask], w2[swap_mask] = w2[swap_mask], w1[swap_mask]
    child1 = Solution(num_assets=n)
    child2 = Solution(num_assets=n)
    child1.P.investment = project_to_simplex(w1, c_l, c_u, rng)
    child2.P.investment = project_to_simplex(w2, c_l, c_u, rng)
    return child1, child2


def thesis_dual_mode_mutation(solution: Solution,
                                p_factor_lo: float = 0.10,
                                p_factor_hi: float = 0.50,
                                add_jitter: float = 0.10,
                                c_l: int = THESIS_CARDINALITY_MIN,
                                c_u: int = THESIS_CARDINALITY_MAX,
                                rng: np.random.Generator | None = None
                                ) -> Solution:
    """Dual-mode mutation per thesis §7.2.3 p.147 (verbatim spec):

    Mode (1): modify non-zero weights — with prob 1/2 increase OR
              decrease investment on a randomly chosen asset by a
              uniformly drawn factor in [p_factor_lo, p_factor_hi]
              = [0.10, 0.50] per thesis.

    Mode (2): add or remove an asset — with prob 1/2 add OR remove
              a randomly chosen asset. Removed: set weight to zero.
              Added: random weight within ±`add_jitter` = ±0.10 of
              equally-balanced allocation per thesis.

    All modified weight vectors are renormalized + projected to
    cardinality-constrained simplex per thesis §7.2 Eq 7.3.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(solution.P.investment)
    w = solution.P.investment.copy()
    mutated = Solution(num_assets=n)
    # Coin flip between mode 1 and mode 2 (per thesis: "we randomly choose between")
    if rng.random() < 0.5:
        # Mode 1: modify a non-zero weight
        active_idx = np.flatnonzero(w > 1e-12)
        if len(active_idx) > 0:
            asset = int(rng.choice(active_idx))
            factor = rng.uniform(p_factor_lo, p_factor_hi)
            if rng.random() < 0.5:
                w[asset] *= (1.0 + factor)  # increase
            else:
                w[asset] *= (1.0 - factor)  # decrease
                w[asset] = max(w[asset], 0.0)
    else:
        # Mode 2: add or remove an asset
        active_idx = np.flatnonzero(w > 1e-12)
        if rng.random() < 0.5 and len(active_idx) > c_l:
            # Remove: only if removing keeps cardinality >= c_l
            asset = int(rng.choice(active_idx))
            w[asset] = 0.0
        else:
            # Add: pick an inactive asset (if any)
            inactive_idx = np.flatnonzero(w <= 1e-12)
            if len(inactive_idx) > 0:
                asset = int(rng.choice(inactive_idx))
                equal_alloc = 1.0 / max(c_l, len(active_idx) + 1)
                jitter = rng.uniform(-add_jitter, add_jitter) * equal_alloc
                w[asset] = max(equal_alloc + jitter, 1e-9)
    mutated.P.investment = project_to_simplex(w, c_l, c_u, rng)
    return mutated
