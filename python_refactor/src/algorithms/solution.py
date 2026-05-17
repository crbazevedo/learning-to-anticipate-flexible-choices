"""
Solution class for genetic algorithms.

This module defines the Solution class that represents a single solution
in the multi-objective optimization algorithms (NSGA-II, SMS-EMOA).
"""

from ..portfolio.portfolio import Portfolio


class Solution:
    """
    Python equivalent of the C++ solution struct.
    
    Represents a single solution in the multi-objective optimization,
    containing a portfolio and various metrics for ranking and selection.
    """
    
    # Static variable for regularization type
    regularization_type: str = "L1"
    
    def __init__(self, num_assets: int):
        """
        Initialize a solution with a portfolio.
        
        Args:
            num_assets: Number of assets in the portfolio
        """
        self.P = Portfolio(num_assets)
        self.P.init()
        
        # Solution metrics
        self.cd: float = 0.0  # Crowding distance
        self.Delta_S: float = 0.0  # Hypervolume contribution
        self.Pareto_rank: int = 0  # Pareto front rank
        self.stability: float = 1.0  # Solution stability
        self.rank_ROI: int = 0  # Rank by ROI
        self.rank_risk: int = 0  # Rank by risk
        self.alpha: float = 0.0  # Prediction confidence
        self.anticipation: bool = False  # Anticipatory learning flag
        self.prediction_error: float = 0.0  # Kalman filter prediction error
        
        # Compute efficiency based on robustness setting (only if data is available)
        if Portfolio.mean_ROI is not None and Portfolio.covariance is not None:
            if Portfolio.robustness:
                Portfolio.compute_robust_efficiency(self.P)
            else:
                Portfolio.compute_efficiency(self.P)
        
        # Evaluate stability (placeholder for now)
        self.evaluate_stability()
    
    def evaluate_stability(self):
        """Evaluate solution stability."""
        # Initialize Kalman filter for the portfolio if not already present
        if self.P.kalman_state is None:
            Portfolio.initialize_kalman_filter(self.P, self.P.ROI, self.P.risk)
        
        # Evaluate stability using Portfolio method
        self.stability = Portfolio.evaluate_stability(self.P)
    
    def dominates_without_constraints(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another without considering constraints.
        
        Args:
            other: Another solution to compare against
        
        Returns:
            True if this solution dominates the other
        """
        if (self.P.ROI < other.P.ROI or self.P.risk > other.P.risk):
            return False
        elif (self.P.ROI > other.P.ROI or self.P.risk < other.P.risk):
            return True
        else:
            return False
    
    def dominates_with_constraints(self, other: 'Solution') -> bool:
        """
        Check if this solution dominates another considering cardinality constraints.
        
        Args:
            other: Another solution to compare against
        
        Returns:
            True if this solution dominates the other
        """
        # Check cardinality constraints
        if (self.P.cardinality > Portfolio.max_cardinality and 
            other.P.cardinality <= Portfolio.max_cardinality):
            return False
        elif (self.P.cardinality <= Portfolio.max_cardinality and 
              other.P.cardinality > Portfolio.max_cardinality):
            return True
        elif (self.P.cardinality > Portfolio.max_cardinality and 
              other.P.cardinality > Portfolio.max_cardinality):
            return self.P.cardinality < other.P.cardinality
        
        # If cardinality constraints are satisfied, check Pareto dominance
        if (self.P.ROI < other.P.ROI or self.P.risk > other.P.risk):
            return False
        elif (self.P.ROI > other.P.ROI or self.P.risk < other.P.risk):
            return True
        else:
            return False
    
    def __lt__(self, other: 'Solution') -> bool:
        """Comparison operator for sorting by Pareto rank and crowding distance."""
        if self.Pareto_rank != other.Pareto_rank:
            return self.Pareto_rank < other.Pareto_rank
        else:
            return self.cd > other.cd  # Higher crowding distance is better
    
    def __repr__(self):
        return (f"Solution(rank={self.Pareto_rank}, cd={self.cd:.4f}, "
                f"ROI={self.P.ROI:.4f}, risk={self.P.risk:.4f})")


def create_random_solution(num_assets: int) -> Solution:
    """
    Create a random solution.
    
    Args:
        num_assets: Number of assets in the portfolio
    
    Returns:
        A randomly initialized solution
    """
    return Solution(num_assets)


# Comparison functions for sorting
def compare_crowding_distance(sol1: Solution, sol2: Solution) -> bool:
    """Compare solutions by crowding distance (descending)."""
    return sol1.cd > sol2.cd


def compare_pareto_rank(sol1: Solution, sol2: Solution) -> bool:
    """Compare solutions by Pareto rank (ascending)."""
    return sol1.Pareto_rank < sol2.Pareto_rank


def compare_ROI(sol1: Solution, sol2: Solution) -> bool:
    """Compare solutions by ROI (descending)."""
    return sol1.P.ROI > sol2.P.ROI


def compare_risk(sol1: Solution, sol2: Solution) -> bool:
    """Compare solutions by risk (ascending)."""
    return sol1.P.risk < sol2.P.risk


def compare_pareto_rank_crowding_distance(sol1: Solution, sol2: Solution) -> bool:
    """Compare solutions by Pareto rank first, then crowding distance."""
    if sol1.Pareto_rank != sol2.Pareto_rank:
        return sol1.Pareto_rank < sol2.Pareto_rank
    else:
        return sol1.cd > sol2.cd 