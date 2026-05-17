"""
Thesis Experimental Parameters Configuration

This module defines the exact experimental parameters as specified in the thesis,
ensuring full alignment with the theoretical framework and experimental setup.
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class ThesisParameters:
    """
    Experimental parameters exactly as specified in the thesis.
    
    This class contains all the parameters used in the thesis experiments,
    ensuring reproducibility and alignment with the theoretical framework.
    """
    
    # =============================================================================
    # ASMS-EMOA Algorithm Parameters (Section 7.2)
    # =============================================================================
    
    # Population and Evolution Parameters
    POPULATION_SIZE: int = 20
    GENERATIONS: int = 30
    MUTATION_RATE: float = 0.3
    CROSSOVER_RATE: float = 0.2
    TOURNAMENT_SIZE: int = 2
    
    # Selection and Ranking Parameters
    NUM_FRONTS: int = 5
    CROWDING_DISTANCE_EPSILON: float = 1e-6
    
    # =============================================================================
    # Portfolio Constraints (Section 6.3)
    # =============================================================================
    
    # Cardinality Constraints
    MIN_CARDINALITY: int = 5
    MAX_CARDINALITY: int = 15
    
    # Feasibility Constraints
    FEASIBILITY_EPSILON: float = 0.99  # ε-feasibility threshold
    REFERENCE_POINT: Tuple[float, float] = (0.2, 0.0)  # (risk, return)
    
    # Weight Constraints
    MIN_WEIGHT: float = 0.01  # Minimum weight for an asset
    MAX_WEIGHT: float = 0.5   # Maximum weight for an asset
    
    # =============================================================================
    # Kalman Filter Parameters (Section 6.2)
    # =============================================================================
    
    # State Space Parameters
    KF_WINDOW_SIZE: int = 20  # K parameter for sliding window
    KF_STATE_DIM: int = 4     # [ROI, risk, ROI_velocity, risk_velocity]
    KF_OBSERVATION_DIM: int = 2  # [ROI, risk]
    
    # Process and Measurement Noise
    KF_PROCESS_NOISE: float = 0.01
    KF_MEASUREMENT_NOISE: float = 0.005
    
    # Monte Carlo Parameters
    MONTE_CARLO_SIMULATIONS: int = 1000
    
    # =============================================================================
    # Dirichlet Dynamical Model Parameters (Section 6.4)
    # =============================================================================
    
    # Sliding Window Parameters
    SLIDING_WINDOW_SIZE: int = 20  # K parameter (same as KF_WINDOW_SIZE)
    CONCENTRATION_SCALING: float = 1.0  # s parameter
    
    # Dirichlet Parameters
    DIRICHLET_SCALE_FACTOR: float = 1.0
    DIRICHLET_CONCENTRATION: float = 20.0  # For MAP updates
    
    # =============================================================================
    # Anticipatory Learning Parameters (Section 6.5)
    # =============================================================================
    
    # Prediction Horizon
    PREDICTION_HORIZON: int = 2  # H parameter
    
    # Learning Rate Parameters
    LEARNING_RATE_COMBINATION: float = 0.5  # Weight for Equation 7.16
    MIN_LEARNING_RATE: float = 0.0
    MAX_LEARNING_RATE: float = 0.5
    
    # TIP Parameters
    TIP_MONTE_CARLO_SAMPLES: int = 1000
    TIP_SIMILARITY_THRESHOLD: float = 0.95
    
    # =============================================================================
    # Experimental Setup Parameters (Section 7.1)
    # =============================================================================
    
    # Time Series Parameters
    HISTORICAL_DAYS: int = 120
    STRIDE_DAYS: int = 30  # Rebalancing frequency
    PREDICTION_STEPS: int = 1  # One-step ahead prediction
    
    # Portfolio Parameters
    INITIAL_WEALTH: float = 100000.0  # R$ 100,000
    TRANSACTION_COST_RATE: float = 0.001  # 0.1% per trade — DEPRECATED;
    # W16-2 (BACKLOG H1) replaces flat-rate with thesis Table 7.1
    # bracket schedule (see THESIS_TABLE_71_BRACKETS below). Kept
    # for backward-compat with config dicts that still reference it.

    # ─── W16-2: Brazilian SEC fee schedule per thesis Table 7.1 (p. 144) ──
    # Bracket = (traded_value_lower_bound, proportional_fee, fixed_fee).
    # Per the thesis verbatim:
    #
    #   Traded value         | Proportional Fee | Fixed Fee
    #   --------------------+------------------+----------
    #   < 135.07            | 0.0%             | 2.70
    #   ≥ 135.08, < 498.62  | 2.0%             | 0.00
    #   ≥ 498.63, < 1514.69 | 1.5%             | 2.49
    #   ≥ 1514.70, < 3029.38| 1.0%             | 10.06
    #   ≥ 3029.39           | 0.5%             | 25.21
    #
    # Brackets listed in ascending-lower-bound order; consumer
    # iterates and picks the bracket whose lower_bound ≤ value <
    # next_bracket_lower_bound (or +∞ for last).
    THESIS_TABLE_71_BRACKETS: list = None  # set in __post_init__
    
    # =============================================================================
    # Decision Maker Parameters (Section 7.3)
    # =============================================================================
    
    # Decision Maker Types
    DECISION_MAKER_TYPES: list = None  # Will be set to ['Hv-DM', 'R-DM', 'M-DM']
    
    # Anticipation Rates (static market assumption)
    STATIC_ANTICIPATION_RATE: float = 1.0  # λ(i)_t = 1 for all i
    
    # =============================================================================
    # Search Operators Parameters (Section 6.6)
    # =============================================================================
    
    # Crossover Parameters
    SBX_ETA: float = 20.0  # Distribution index for SBX
    
    # Mutation Parameters
    MUTATION_STRENGTH: float = 0.1
    MUTATION_PROBABILITY: float = 0.1
    
    # =============================================================================
    # Correspondence Mapping Parameters
    # =============================================================================
    
    # Historical Tracking
    MAX_HISTORY_SIZE: int = 50
    CORRESPONDENCE_SIMILARITY_THRESHOLD: float = 0.95
    
    # =============================================================================
    # Performance and Quality Parameters
    # =============================================================================
    
    # Convergence Criteria
    CONVERGENCE_TOLERANCE: float = 1e-6
    MAX_STAGNATION_GENERATIONS: int = 10
    
    # Quality Metrics
    HYPERVOLUME_REFERENCE_POINT: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Initialize derived parameters after dataclass creation."""
        if self.DECISION_MAKER_TYPES is None:
            self.DECISION_MAKER_TYPES = ['Hv-DM', 'R-DM', 'M-DM']
        # W16-2: Brazilian SEC fee schedule per thesis Table 7.1 (p. 144).
        # Each entry: (lower_bound_inclusive, proportional_fee, fixed_fee).
        if self.THESIS_TABLE_71_BRACKETS is None:
            self.THESIS_TABLE_71_BRACKETS = [
                (0.00,    0.000, 2.70),    # < 135.07
                (135.08,  0.020, 0.00),    # 135.08 ≤ v < 498.62
                (498.63,  0.015, 2.49),    # 498.63 ≤ v < 1514.69
                (1514.70, 0.010, 10.06),   # 1514.70 ≤ v < 3029.38
                (3029.39, 0.005, 25.21),   # ≥ 3029.39
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for easy access."""
        return {
            'population_size': self.POPULATION_SIZE,
            'generations': self.GENERATIONS,
            'mutation_rate': self.MUTATION_RATE,
            'crossover_rate': self.CROSSOVER_RATE,
            'tournament_size': self.TOURNAMENT_SIZE,
            'min_cardinality': self.MIN_CARDINALITY,
            'max_cardinality': self.MAX_CARDINALITY,
            'feasibility_epsilon': self.FEASIBILITY_EPSILON,
            'reference_point': self.REFERENCE_POINT,
            'kf_window_size': self.KF_WINDOW_SIZE,
            'monte_carlo_simulations': self.MONTE_CARLO_SIMULATIONS,
            'sliding_window_size': self.SLIDING_WINDOW_SIZE,
            'concentration_scaling': self.CONCENTRATION_SCALING,
            'dirichlet_scale_factor': self.DIRICHLET_SCALE_FACTOR,
            'dirichlet_concentration': self.DIRICHLET_CONCENTRATION,
            'prediction_horizon': self.PREDICTION_HORIZON,
            'learning_rate_combination': self.LEARNING_RATE_COMBINATION,
            'historical_days': self.HISTORICAL_DAYS,
            'stride_days': self.STRIDE_DAYS,
            'initial_wealth': self.INITIAL_WEALTH,
            'transaction_cost_rate': self.TRANSACTION_COST_RATE,
            'decision_maker_types': self.DECISION_MAKER_TYPES,
            'static_anticipation_rate': self.STATIC_ANTICIPATION_RATE,
            'sbx_eta': self.SBX_ETA,
            'mutation_strength': self.MUTATION_STRENGTH,
            'mutation_probability': self.MUTATION_PROBABILITY,
            'max_history_size': self.MAX_HISTORY_SIZE,
            'correspondence_similarity_threshold': self.CORRESPONDENCE_SIMILARITY_THRESHOLD,
            'convergence_tolerance': self.CONVERGENCE_TOLERANCE,
            'max_stagnation_generations': self.MAX_STAGNATION_GENERATIONS,
            'hypervolume_reference_point': self.HYPERVOLUME_REFERENCE_POINT
        }
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get algorithm-specific configuration."""
        return {
            'population_size': self.POPULATION_SIZE,
            'generations': self.GENERATIONS,
            'mutation_rate': self.MUTATION_RATE,
            'crossover_rate': self.CROSSOVER_RATE,
            'tournament_size': self.TOURNAMENT_SIZE,
            'num_fronts': self.NUM_FRONTS,
            'crowding_distance_epsilon': self.CROWDING_DISTANCE_EPSILON,
            'convergence_tolerance': self.CONVERGENCE_TOLERANCE,
            'max_stagnation_generations': self.MAX_STAGNATION_GENERATIONS
        }
    
    def get_portfolio_config(self) -> Dict[str, Any]:
        """Get portfolio-specific configuration."""
        return {
            'min_cardinality': self.MIN_CARDINALITY,
            'max_cardinality': self.MAX_CARDINALITY,
            'min_weight': self.MIN_WEIGHT,
            'max_weight': self.MAX_WEIGHT,
            'feasibility_epsilon': self.FEASIBILITY_EPSILON,
            'reference_point': self.REFERENCE_POINT,
            'initial_wealth': self.INITIAL_WEALTH,
            'transaction_cost_rate': self.TRANSACTION_COST_RATE
        }
    
    def get_anticipatory_config(self) -> Dict[str, Any]:
        """Get anticipatory learning configuration."""
        return {
            'prediction_horizon': self.PREDICTION_HORIZON,
            'learning_rate_combination': self.LEARNING_RATE_COMBINATION,
            'min_learning_rate': self.MIN_LEARNING_RATE,
            'max_learning_rate': self.MAX_LEARNING_RATE,
            'tip_monte_carlo_samples': self.TIP_MONTE_CARLO_SAMPLES,
            'tip_similarity_threshold': self.TIP_SIMILARITY_THRESHOLD,
            'static_anticipation_rate': self.STATIC_ANTICIPATION_RATE
        }
    
    def get_kalman_config(self) -> Dict[str, Any]:
        """Get Kalman filter configuration."""
        return {
            'window_size': self.KF_WINDOW_SIZE,
            'state_dim': self.KF_STATE_DIM,
            'observation_dim': self.KF_OBSERVATION_DIM,
            'process_noise': self.KF_PROCESS_NOISE,
            'measurement_noise': self.KF_MEASUREMENT_NOISE,
            'monte_carlo_simulations': self.MONTE_CARLO_SIMULATIONS
        }
    
    def get_dirichlet_config(self) -> Dict[str, Any]:
        """Get Dirichlet model configuration."""
        return {
            'sliding_window_size': self.SLIDING_WINDOW_SIZE,
            'concentration_scaling': self.CONCENTRATION_SCALING,
            'dirichlet_scale_factor': self.DIRICHLET_SCALE_FACTOR,
            'dirichlet_concentration': self.DIRICHLET_CONCENTRATION
        }
    
    def get_experimental_config(self) -> Dict[str, Any]:
        """Get experimental setup configuration."""
        return {
            'historical_days': self.HISTORICAL_DAYS,
            'stride_days': self.STRIDE_DAYS,
            'prediction_steps': self.PREDICTION_STEPS,
            'decision_maker_types': self.DECISION_MAKER_TYPES,
            'max_history_size': self.MAX_HISTORY_SIZE,
            'correspondence_similarity_threshold': self.CORRESPONDENCE_SIMILARITY_THRESHOLD
        }
    
    def get_search_operators_config(self) -> Dict[str, Any]:
        """Get search operators configuration."""
        return {
            'sbx_eta': self.SBX_ETA,
            'mutation_strength': self.MUTATION_STRENGTH,
            'mutation_probability': self.MUTATION_PROBABILITY
        }
    
    def validate_parameters(self) -> bool:
        """
        Validate that all parameters are within reasonable ranges.
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        validations = [
            (self.POPULATION_SIZE > 0, "Population size must be positive"),
            (self.GENERATIONS > 0, "Generations must be positive"),
            (0 <= self.MUTATION_RATE <= 1, "Mutation rate must be between 0 and 1"),
            (0 <= self.CROSSOVER_RATE <= 1, "Crossover rate must be between 0 and 1"),
            (self.TOURNAMENT_SIZE > 0, "Tournament size must be positive"),
            (self.MIN_CARDINALITY > 0, "Minimum cardinality must be positive"),
            (self.MAX_CARDINALITY >= self.MIN_CARDINALITY, "Maximum cardinality must be >= minimum"),
            (0 <= self.FEASIBILITY_EPSILON <= 1, "Feasibility epsilon must be between 0 and 1"),
            (self.KF_WINDOW_SIZE > 0, "Kalman filter window size must be positive"),
            (self.MONTE_CARLO_SIMULATIONS > 0, "Monte Carlo simulations must be positive"),
            (self.SLIDING_WINDOW_SIZE > 0, "Sliding window size must be positive"),
            (self.PREDICTION_HORIZON > 0, "Prediction horizon must be positive"),
            (0 <= self.LEARNING_RATE_COMBINATION <= 1, "Learning rate combination must be between 0 and 1"),
            (self.HISTORICAL_DAYS > 0, "Historical days must be positive"),
            (self.STRIDE_DAYS > 0, "Stride days must be positive"),
            (self.INITIAL_WEALTH > 0, "Initial wealth must be positive"),
            (self.TRANSACTION_COST_RATE >= 0, "Transaction cost rate must be non-negative")
        ]
        
        for is_valid, message in validations:
            if not is_valid:
                print(f"Parameter validation failed: {message}")
                return False
        
        return True
    
    def print_parameters(self):
        """Print all parameters in a formatted way."""
        print("=" * 80)
        print("THESIS EXPERIMENTAL PARAMETERS")
        print("=" * 80)
        
        print("\n📊 ASMS-EMOA Algorithm Parameters:")
        print(f"  Population Size: {self.POPULATION_SIZE}")
        print(f"  Generations: {self.GENERATIONS}")
        print(f"  Mutation Rate: {self.MUTATION_RATE}")
        print(f"  Crossover Rate: {self.CROSSOVER_RATE}")
        print(f"  Tournament Size: {self.TOURNAMENT_SIZE}")
        
        print("\n🎯 Portfolio Constraints:")
        print(f"  Cardinality: {self.MIN_CARDINALITY}-{self.MAX_CARDINALITY}")
        print(f"  Feasibility Epsilon: {self.FEASIBILITY_EPSILON}")
        print(f"  Reference Point: {self.REFERENCE_POINT}")
        
        print("\n🔮 Anticipatory Learning:")
        print(f"  Prediction Horizon: {self.PREDICTION_HORIZON}")
        print(f"  Learning Rate Combination: {self.LEARNING_RATE_COMBINATION}")
        print(f"  Static Anticipation Rate: {self.STATIC_ANTICIPATION_RATE}")
        
        print("\n📈 Kalman Filter:")
        print(f"  Window Size: {self.KF_WINDOW_SIZE}")
        print(f"  Monte Carlo Simulations: {self.MONTE_CARLO_SIMULATIONS}")
        
        print("\n🎲 Dirichlet Model:")
        print(f"  Sliding Window Size: {self.SLIDING_WINDOW_SIZE}")
        print(f"  Concentration Scaling: {self.CONCENTRATION_SCALING}")
        
        print("\n🧪 Experimental Setup:")
        print(f"  Historical Days: {self.HISTORICAL_DAYS}")
        print(f"  Stride Days: {self.STRIDE_DAYS}")
        print(f"  Initial Wealth: R$ {self.INITIAL_WEALTH:,.2f}")
        print(f"  Transaction Cost Rate: {self.TRANSACTION_COST_RATE:.3f}")
        
        print("\n👥 Decision Makers:")
        for dm in self.DECISION_MAKER_TYPES:
            print(f"  - {dm}")
        
        print("=" * 80)


# Create default instance factory function
def get_default_thesis_parameters() -> ThesisParameters:
    """Get a fresh instance of default thesis parameters."""
    return ThesisParameters()

# Create default instance
DEFAULT_THESIS_PARAMETERS = get_default_thesis_parameters()


# Alternative parameter sets for different experimental scenarios
class AlternativeParameters:
    """Alternative parameter sets for different experimental scenarios."""
    
    @staticmethod
    def get_small_scale_parameters() -> ThesisParameters:
        """Parameters for small-scale testing."""
        params = ThesisParameters()
        params.POPULATION_SIZE = 10
        params.GENERATIONS = 20
        params.HISTORICAL_DAYS = 60
        params.STRIDE_DAYS = 15
        params.MONTE_CARLO_SIMULATIONS = 500
        return params
    
    @staticmethod
    def get_large_scale_parameters() -> ThesisParameters:
        """Parameters for large-scale experiments."""
        params = ThesisParameters()
        params.POPULATION_SIZE = 50
        params.GENERATIONS = 100
        params.HISTORICAL_DAYS = 240
        params.STRIDE_DAYS = 60
        params.MONTE_CARLO_SIMULATIONS = 2000
        return params
    
    @staticmethod
    def get_high_frequency_parameters() -> ThesisParameters:
        """Parameters for high-frequency trading scenarios."""
        params = ThesisParameters()
        params.STRIDE_DAYS = 5
        params.KF_WINDOW_SIZE = 10
        params.SLIDING_WINDOW_SIZE = 10
        params.PREDICTION_HORIZON = 1
        return params
    
    @staticmethod
    def get_conservative_parameters() -> ThesisParameters:
        """Parameters for conservative investment strategies."""
        params = ThesisParameters()
        params.MIN_CARDINALITY = 10
        params.MAX_CARDINALITY = 20
        params.FEASIBILITY_EPSILON = 0.95
        params.LEARNING_RATE_COMBINATION = 0.3
        return params
