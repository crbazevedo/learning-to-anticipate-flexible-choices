"""
Tests for Anticipatory Learning implementation.
"""

import pytest
import numpy as np
from src.algorithms.anticipatory_learning import (
    AnticipatoryLearning, AnticipativeDistribution, DirichletPredictor
)
from src.portfolio.portfolio import Portfolio
from src.algorithms.solution import Solution


class TestAnticipativeDistribution:
    """Test AnticipativeDistribution class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.current_state = np.array([0.1, 0.05, 0.01, 0.02])
        self.predicted_state = np.array([0.12, 0.06, 0.015, 0.025])
        self.current_covariance = np.eye(4) * 0.01
        self.predicted_covariance = np.eye(4) * 0.015
        
        self.anticipative_dist = AnticipativeDistribution(
            self.current_state, self.predicted_state,
            self.current_covariance, self.predicted_covariance
        )
    
    def test_anticipative_distribution_initialization(self):
        """Test AnticipativeDistribution initialization per paper Eq (15).

        W22-NC12: pre-NC12 the formula was SUM (current + predicted), now
        per Eq (15) it is weighted-sum-of-squared-weights with default
        equal weights w_current = w_predicted = 0.5.
        """
        assert np.array_equal(self.anticipative_dist.current_state, self.current_state)
        assert np.array_equal(self.anticipative_dist.predicted_state, self.predicted_state)
        assert np.array_equal(self.anticipative_dist.current_covariance, self.current_covariance)
        assert np.array_equal(self.anticipative_dist.predicted_covariance, self.predicted_covariance)

        # Per paper Eq (15): Σ_anticipative = w_current² · Σ_current + w_predicted² · Σ_predicted
        # With default w_current = 0.5, w_predicted = 0.5, this is
        # 0.25 · (Σ_current + Σ_predicted).
        expected_combined_cov = (
            0.25 * self.current_covariance + 0.25 * self.predicted_covariance
        )
        np.testing.assert_array_almost_equal(
            self.anticipative_dist.anticipative_covariance, expected_combined_cov
        )

        # Anticipative mean per Eq (15) with equal weights = arithmetic mean.
        expected_mean = (self.current_state + self.predicted_state) / 2.0
        np.testing.assert_array_almost_equal(
            self.anticipative_dist.anticipative_mean, expected_mean
        )

    def test_sample_anticipative_state(self):
        """Test sampling from anticipative distribution."""
        samples = self.anticipative_dist.sample_anticipative_state(num_samples=100)

        assert samples.shape[0] == 4  # 4-dimensional state
        assert samples.shape[1] == 100  # 100 samples

        # Check that samples are reasonable
        assert np.all(np.isfinite(samples))

    def test_compute_anticipative_confidence(self):
        """Test anticipative confidence computation."""
        confidence = self.anticipative_dist.compute_anticipative_confidence()

        assert 0.0 <= confidence <= 1.0
        assert np.isfinite(confidence)


class TestW22NC12CovarianceFusion:
    """W22-NC12 regression: anticipative_covariance must use paper Eq (15)
    weighted-sum-of-squared-weights, NOT the naive sum.

    Pre-NC12 the formula was `current + predicted` (a SUM); this caused
    `_update_solution_state_anticipative` (line ~1412) to blend
    kalman_state.P toward a value ≥ 2× the current P, growing P by factor
    (1 + α) per generation. After ~30 generations P[0,0] grew from 0.0091
    to ~542 (a 60,000× factor consistent with 1.5^28 = 88,000). Once
    P[0,0] is large, TIP MC sampling becomes pure noise → TIP ≈ 0.5
    (saturation), λ_combined ≈ 0.5 uniform across portfolios, anticipation
    arm produces no per-portfolio differentiation.

    Receipt:
      - docs/W22-PROBE-A-KF-PREDICTIVE-ACCURACY-POST-NC7.md: kf_P_diag = [542, 542, 1000, 1000]
      - W22 Probe B preliminary: TIP saturated near 0.5 in first 4 records.

    Per paper Eq (15) / standard Bayesian convex combination of two
    independent Gaussians ẑ = w_c · ẑ_c + w_p · ẑ_p:
      E[ẑ] = w_c · E[ẑ_c] + w_p · E[ẑ_p]
      Cov[ẑ] = w_c² · Cov[ẑ_c] + w_p² · Cov[ẑ_p]
    """

    def test_covariance_uses_squared_weights_not_sum(self):
        """The cornerstone: pre-NC12 SUM ≠ post-NC12 squared-weight combo."""
        from src.algorithms.anticipatory_learning import AnticipativeDistribution
        current_cov = np.eye(4) * 0.01
        predicted_cov = np.eye(4) * 0.02
        ad = AnticipativeDistribution(
            np.zeros(4), np.zeros(4), current_cov, predicted_cov,
        )
        # With default equal weights (0.5 each), expected = 0.25*(0.01 + 0.02)·I = 0.0075·I
        expected = 0.25 * current_cov + 0.25 * predicted_cov
        np.testing.assert_array_almost_equal(ad.anticipative_covariance, expected)
        # The PRE-NC12 (wrong) SUM would have been 0.03·I.
        # Verify we are NOT producing the broken value.
        broken_sum = current_cov + predicted_cov
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_almost_equal(ad.anticipative_covariance, broken_sum)

    def test_extreme_weight_current_recovers_current_covariance(self):
        """With weight_current=1.0, anticipative_covariance == current_covariance."""
        from src.algorithms.anticipatory_learning import AnticipativeDistribution
        current_cov = np.eye(4) * 0.5
        predicted_cov = np.eye(4) * 100  # would explode under SUM
        ad = AnticipativeDistribution(
            np.zeros(4), np.zeros(4), current_cov, predicted_cov,
            weight_current=1.0,
        )
        np.testing.assert_array_almost_equal(ad.anticipative_covariance, current_cov)

    def test_extreme_weight_predicted_recovers_predicted_covariance(self):
        """With weight_current=0.0, anticipative_covariance == predicted_covariance."""
        from src.algorithms.anticipatory_learning import AnticipativeDistribution
        current_cov = np.eye(4) * 100  # would explode under SUM
        predicted_cov = np.eye(4) * 0.5
        ad = AnticipativeDistribution(
            np.zeros(4), np.zeros(4), current_cov, predicted_cov,
            weight_current=0.0,
        )
        np.testing.assert_array_almost_equal(ad.anticipative_covariance, predicted_cov)

    def test_anticipative_cov_never_exceeds_max_input_component(self):
        """Eq (15) invariant: anticipative_covariance ≤ max(current, predicted) in any
        scalar dimension, for any w_current ∈ [0, 1]. The SUM formula VIOLATED this."""
        from src.algorithms.anticipatory_learning import AnticipativeDistribution
        current_cov = np.eye(4) * 0.03
        predicted_cov = np.eye(4) * 0.01
        max_input = max(0.03, 0.01)  # 0.03
        for w in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            ad = AnticipativeDistribution(
                np.zeros(4), np.zeros(4), current_cov, predicted_cov,
                weight_current=w,
            )
            diag_max = float(np.max(np.diag(ad.anticipative_covariance)))
            assert diag_max <= max_input + 1e-12, (
                f"NC12 REGRESSION: at w_current={w}, anticipative_cov max diag "
                f"{diag_max:.6e} exceeds max input {max_input:.6e}; Eq (15) "
                f"invariant violated (this is what the broken SUM did)"
            )


class TestDirichletPredictor:
    """Test DirichletPredictor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.prev_proportions = np.array([0.3, 0.4, 0.3])
        self.current_proportions = np.array([0.25, 0.45, 0.3])
        self.anticipative_rate = 0.5
    
    def test_dirichlet_mean_prediction_vec(self):
        """Test Dirichlet mean prediction vector."""
        predicted = DirichletPredictor.dirichlet_mean_prediction_vec(
            self.prev_proportions, self.current_proportions, self.anticipative_rate
        )
        
        assert len(predicted) == 3
        assert np.all(predicted >= 0.0)
        assert np.all(predicted <= 1.0)
        assert np.abs(np.sum(predicted) - 1.0) < 1e-10
        
        # Check that prediction is between prev and current
        for i in range(3):
            assert min(self.prev_proportions[i], self.current_proportions[i]) <= predicted[i] <= max(self.prev_proportions[i], self.current_proportions[i])
    
    def test_dirichlet_mean_map_update(self):
        """Test Dirichlet MAP update."""
        p_predicted = np.array([0.3, 0.4, 0.3])
        p_obs = np.array([0.25, 0.45, 0.3])
        concentration = 10.0
        
        updated = DirichletPredictor.dirichlet_mean_map_update(p_predicted, p_obs, concentration)
        
        assert len(updated) == 3
        assert np.all(updated >= 0.0)
        assert np.all(updated <= 1.0)
        assert np.abs(np.sum(updated) - 1.0) < 1e-10


class TestAnticipatoryLearning:
    """Test AnticipatoryLearning class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.covariance = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.015],
            [0.01, 0.015, 0.16]
        ])
        Portfolio.robust_covariance = Portfolio.covariance.copy()
        Portfolio.robustness = False
        Portfolio.window_size = 5
        
        # Create synthetic returns data
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        # Create anticipatory learning instance
        self.anticipatory_learning = AnticipatoryLearning(
            learning_rate=0.01,
            prediction_horizon=1,
            monte_carlo_simulations=100,
            window_size=5
        )
        
        # Create test solution
        self.solution = Solution(num_assets=3)
        self.solution.P.ROI = 0.12
        self.solution.P.risk = 0.08
        self.solution.P.investment = np.array([0.3, 0.4, 0.3])
    
    def test_anticipatory_learning_initialization(self):
        """Test AnticipatoryLearning initialization."""
        assert self.anticipatory_learning.base_learning_rate == 0.01
        assert self.anticipatory_learning.prediction_horizon == 1
        assert self.anticipatory_learning.monte_carlo_simulations == 100
        assert self.anticipatory_learning.window_size == 5
        assert self.anticipatory_learning.adaptive_learning is True
        assert len(self.anticipatory_learning.historical_populations) == 0
        assert len(self.anticipatory_learning.historical_anticipative_decisions) == 0
        assert self.anticipatory_learning.predicted_anticipative_decision is None
    
    def test_store_historical_population(self):
        """Test historical population storage."""
        population = [self.solution]
        
        self.anticipatory_learning.store_historical_population(population)
        
        assert len(self.anticipatory_learning.historical_populations) == 1
        assert len(self.anticipatory_learning.historical_populations[0]) == 1
        
        stored_solution = self.anticipatory_learning.historical_populations[0][0]
        assert np.array_equal(stored_solution.P.investment, self.solution.P.investment)
        assert stored_solution.P.ROI == self.solution.P.ROI
        assert stored_solution.P.risk == self.solution.P.risk
    
    def test_store_anticipative_decision(self):
        """Test anticipative decision storage."""
        self.anticipatory_learning.store_anticipative_decision(self.solution)
        
        assert len(self.anticipatory_learning.historical_anticipative_decisions) == 1
        assert self.anticipatory_learning.predicted_anticipative_decision is not None
        
        stored_decision = self.anticipatory_learning.historical_anticipative_decisions[0]
        assert np.array_equal(stored_decision.P.investment, self.solution.P.investment)
    
    def test_compute_anticipatory_learning_rate(self):
        """Test anticipatory learning rate computation."""
        min_error = 0.01
        max_error = 0.1
        min_alpha = 0.5
        max_alpha = 0.9
        current_time = 10
        
        # Set solution attributes
        self.solution.prediction_error = 0.05
        self.solution.alpha = 0.7
        
        rate = self.anticipatory_learning.compute_anticipatory_learning_rate(
            self.solution, min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        assert 0.0 <= rate <= 1.0
        assert np.isfinite(rate)
    
    def test_compute_transaction_cost(self):
        """Test transaction cost computation."""
        current_weights = np.array([0.3, 0.4, 0.3])
        new_weights = np.array([0.25, 0.45, 0.3])
        
        cost = self.anticipatory_learning.compute_transaction_cost(current_weights, new_weights)
        
        assert cost >= 0.0
        assert np.isfinite(cost)
        
        # Cost should be zero for identical weights
        zero_cost = self.anticipatory_learning.compute_transaction_cost(current_weights, current_weights)
        assert zero_cost == 0.0
    
    def test_epsilon_feasibility(self):
        """Test epsilon feasibility computation."""
        # Initialize Kalman state for solution
        from src.algorithms.kalman_filter import create_kalman_params
        self.solution.P.kalman_state = create_kalman_params(0.12, 0.08)
        
        feasibility = self.anticipatory_learning.epsilon_feasibility(self.solution)
        
        assert len(feasibility) == 2
        assert all(0.0 <= p <= 1.0 for p in feasibility)
        assert all(np.isfinite(p) for p in feasibility)
    
    def test_is_epsilon_feasible(self):
        """Test epsilon feasibility checking."""
        # Initialize Kalman state for solution
        from src.algorithms.kalman_filter import create_kalman_params
        self.solution.P.kalman_state = create_kalman_params(0.12, 0.08)
        
        is_feasible = self.anticipatory_learning.is_epsilon_feasible(self.solution)
        
        assert isinstance(is_feasible, bool)
    
    def test_non_dominance_probability(self):
        """Test non-dominance probability computation."""
        # Create two solutions with Kalman states
        from src.algorithms.kalman_filter import create_kalman_params
        
        solution1 = Solution(num_assets=3)
        solution1.P.ROI = 0.12
        solution1.P.risk = 0.08
        solution1.P.kalman_state = create_kalman_params(0.12, 0.08)
        
        solution2 = Solution(num_assets=3)
        solution2.P.ROI = 0.15
        solution2.P.risk = 0.10
        solution2.P.kalman_state = create_kalman_params(0.15, 0.10)
        
        probability = self.anticipatory_learning.non_dominance_probability(solution1, solution2)
        
        assert 0.0 <= probability <= 1.0
        assert np.isfinite(probability)
    
    def test_learn_single_solution(self):
        """Test learning for single solution."""
        # Initialize Kalman state for solution
        from src.algorithms.kalman_filter import create_kalman_params
        self.solution.P.kalman_state = create_kalman_params(0.12, 0.08)
        
        # Test learning
        self.anticipatory_learning.learn_single_solution(self.solution, current_time=10)
        
        # Check that learning occurred
        assert hasattr(self.solution, 'anticipation')
        assert hasattr(self.solution, 'alpha')
        assert hasattr(self.solution, 'prediction_error')
        assert len(self.anticipatory_learning.learning_history) > 0
    
    def test_learn_population(self):
        """Test learning for population."""
        # Create population
        population = [self.solution]
        
        # Initialize Kalman state for solution
        from src.algorithms.kalman_filter import create_kalman_params
        self.solution.P.kalman_state = create_kalman_params(0.12, 0.08)
        
        # Test learning
        self.anticipatory_learning.learn_population(population, current_time=10)
        
        # Check that learning occurred
        assert len(self.anticipatory_learning.historical_populations) > 0
        assert len(self.anticipatory_learning.stochastic_pareto_frontiers) > 0
    
    def test_get_learning_metrics(self):
        """Test learning metrics retrieval."""
        # Add some learning history
        self.anticipatory_learning.learning_history = [
            {'alpha': 0.7, 'prediction_error': 0.05, 'state_quality': 0.8}
        ]
        
        metrics = self.anticipatory_learning.get_learning_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_learning_events' in metrics
        assert 'mean_alpha' in metrics
        assert 'mean_prediction_error' in metrics
        assert 'mean_state_quality' in metrics
    
    def test_reset(self):
        """Test learning system reset."""
        # Add some data
        self.anticipatory_learning.learning_history = [{'test': 'data'}]
        self.anticipatory_learning.historical_populations = [[self.solution]]
        
        # Reset
        self.anticipatory_learning.reset()
        
        # Check that all data is cleared
        assert len(self.anticipatory_learning.learning_history) == 0
        assert len(self.anticipatory_learning.historical_populations) == 0
        assert len(self.anticipatory_learning.stochastic_pareto_frontiers) == 0


class TestAnticipatoryLearningIntegration:
    """Test integration with other components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize Portfolio static variables
        Portfolio.available_assets_size = 3
        Portfolio.mean_ROI = np.array([0.1, 0.15, 0.12])
        Portfolio.median_ROI = np.array([0.1, 0.15, 0.12])  # Same as mean for simplicity
        Portfolio.covariance = np.eye(3) * 0.1
        Portfolio.robust_covariance = np.eye(3) * 0.1
        Portfolio.complete_returns_data = np.random.normal(0, 0.1, (50, 3))
        
        self.anticipatory_learning = AnticipatoryLearning()
    
    def test_integration_with_solution_class(self):
        """Test integration with Solution class."""
        solution = Solution(num_assets=3)
        solution.P.ROI = 0.12
        solution.P.risk = 0.08
        
        # Should not fail
        self.anticipatory_learning.learn_single_solution(solution, current_time=0)
        
        # Check that solution was updated
        assert hasattr(solution, 'anticipation')
    
    def test_integration_with_portfolio_class(self):
        """Test integration with Portfolio class."""
        # Create solution first (it will set up Portfolio static variables)
        solution = Solution(num_assets=3)
        
        # Create a new portfolio and assign it
        portfolio = Portfolio(3)
        portfolio.init()
        solution.P = portfolio
        
        # Initialize Kalman state
        from src.algorithms.kalman_filter import create_kalman_params
        solution.P.kalman_state = create_kalman_params(solution.P.ROI, solution.P.risk)
        
        # Should not fail
        self.anticipatory_learning.learn_single_solution(solution, current_time=0)
        
        # Check that learning occurred
        assert hasattr(solution, 'anticipation')


class TestAnticipatoryLearningEdgeCases:
    """Test edge cases and error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.anticipatory_learning = AnticipatoryLearning()
    
    def test_empty_population(self):
        """Test learning with empty population."""
        empty_population = []
        
        # Should not fail
        self.anticipatory_learning.learn_population(empty_population, current_time=0)
        
        # Should store empty population
        assert len(self.anticipatory_learning.historical_populations) == 1
        assert len(self.anticipatory_learning.historical_populations[0]) == 0
    
    def test_solution_without_kalman_state(self):
        """Test learning with solution without Kalman state."""
        solution = Solution(num_assets=3)
        
        # Should not fail
        self.anticipatory_learning.learn_single_solution(solution, current_time=0)
        
        # Should still mark as anticipated
        assert hasattr(solution, 'anticipation')
    
    def test_negative_time(self):
        """Test learning with negative time."""
        solution = Solution(num_assets=3)
        
        # Should not fail
        self.anticipatory_learning.learn_single_solution(solution, current_time=-1)
        
        # Should still work
        assert hasattr(solution, 'anticipation')
    
    def test_large_time_values(self):
        """Test learning with large time values."""
        solution = Solution(num_assets=3)
        
        # Should not fail
        self.anticipatory_learning.learn_single_solution(solution, current_time=1000000)
        
        # Should still work
        assert hasattr(solution, 'anticipation') 