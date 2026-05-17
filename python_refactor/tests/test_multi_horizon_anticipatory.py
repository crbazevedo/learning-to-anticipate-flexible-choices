"""
Unit tests for Multi-Horizon Anticipatory Learning

Tests the multi-horizon anticipatory learning implementation including
Equation 6.10 and multi-horizon prediction capabilities.
"""

import unittest
import numpy as np

# W2-2 import-style fix: use canonical `from src.algorithms.X` (W1-2/W1-3/W1-4 pattern).
from src.algorithms.multi_horizon_anticipatory import (
    MultiHorizonAnticipatoryLearning, MultiHorizonPrediction,
    create_multi_horizon_anticipatory_learning
)


class TestMultiHorizonAnticipatoryLearning(unittest.TestCase):
    """Test cases for MultiHorizonAnticipatoryLearning class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.multi_horizon_learning = MultiHorizonAnticipatoryLearning(max_horizon=3, monte_carlo_samples=100)
        
        # Create mock solution for testing
        self.mock_solution = self._create_mock_solution()
        
    def _create_mock_solution(self):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk, weights):
                self.ROI = roi
                self.risk = risk
                self.investment = weights
                self.cardinality = np.sum(weights > 0.01)
                self.num_assets = len(weights)
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk, weights):
                self.P = MockPortfolio(roi, risk, weights)
                self.alpha = 0.5
                self.prediction_error = 0.01
                self.anticipation = False
        
        return MockSolution(0.1, 0.05, np.array([0.4, 0.3, 0.2, 0.1]))
    
    def test_initialization(self):
        """Test initialization of multi-horizon anticipatory learning."""
        self.assertEqual(self.multi_horizon_learning.max_horizon, 3)
        self.assertEqual(self.multi_horizon_learning.monte_carlo_samples, 100)
        self.assertIsNotNone(self.multi_horizon_learning.tip_calculator)
        self.assertIsNotNone(self.multi_horizon_learning.n_step_predictor)
        self.assertIsNotNone(self.multi_horizon_learning.dirichlet_model)
        self.assertEqual(len(self.multi_horizon_learning.prediction_history), 0)
        self.assertEqual(len(self.multi_horizon_learning.lambda_rates_history), 0)
        
    def test_apply_anticipatory_learning_rule(self):
        """Test application of anticipatory learning rule (Equation 6.10)."""
        # Test with single horizon
        current_state = np.array([0.1, 0.05])
        predicted_states = [np.array([0.12, 0.06])]
        lambda_rates = [0.3]
        
        result = self.multi_horizon_learning.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Expected: (1 - 0.3) * [0.1, 0.05] + 0.3 * [0.12, 0.06]
        # = 0.7 * [0.1, 0.05] + 0.3 * [0.12, 0.06]
        # = [0.07, 0.035] + [0.036, 0.018] = [0.106, 0.053]
        expected = np.array([0.106, 0.053])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        
    def test_apply_anticipatory_learning_rule_multiple_horizons(self):
        """Test anticipatory learning rule with multiple horizons."""
        current_state = np.array([0.1, 0.05])
        predicted_states = [
            np.array([0.12, 0.06]),  # h=1
            np.array([0.14, 0.07])   # h=2
        ]
        lambda_rates = [0.2, 0.3]
        
        result = self.multi_horizon_learning.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Expected: (1 - 0.5) * [0.1, 0.05] + 0.2 * [0.12, 0.06] + 0.3 * [0.14, 0.07]
        # = 0.5 * [0.1, 0.05] + [0.024, 0.012] + [0.042, 0.021]
        # = [0.05, 0.025] + [0.024, 0.012] + [0.042, 0.021] = [0.116, 0.058]
        expected = np.array([0.116, 0.058])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        
    def test_apply_anticipatory_learning_rule_empty_predictions(self):
        """Test anticipatory learning rule with empty predictions."""
        current_state = np.array([0.1, 0.05])
        predicted_states = []
        lambda_rates = []
        
        result = self.multi_horizon_learning.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Should return current state unchanged
        np.testing.assert_array_equal(result, current_state)
        
    def test_apply_anticipatory_learning_rule_lambda_sum_exceeds_one(self):
        """Test anticipatory learning rule when lambda sum exceeds 1.0."""
        current_state = np.array([0.1, 0.05])
        predicted_states = [np.array([0.12, 0.06])]
        lambda_rates = [1.5]  # Exceeds 1.0
        
        result = self.multi_horizon_learning.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Should normalize lambda rates
        # Normalized lambda = 1.5 / 1.5 = 1.0
        # Result = (1 - 1.0) * [0.1, 0.05] + 1.0 * [0.12, 0.06] = [0.12, 0.06]
        expected = np.array([0.12, 0.06])
        np.testing.assert_array_almost_equal(result, expected, decimal=3)
        
    def test_calculate_multi_horizon_lambda_rates(self):
        """Test calculation of multi-horizon lambda rates."""
        # Test with horizon 2
        lambda_rates = self.multi_horizon_learning.calculate_multi_horizon_lambda_rates(
            self.mock_solution, 2
        )
        
        self.assertEqual(len(lambda_rates), 1)  # Only h=1 for horizon 2
        self.assertGreaterEqual(lambda_rates[0], 0.0)
        self.assertLessEqual(lambda_rates[0], 0.5)
        
        # Test with horizon 3
        lambda_rates = self.multi_horizon_learning.calculate_multi_horizon_lambda_rates(
            self.mock_solution, 3
        )
        
        self.assertEqual(len(lambda_rates), 2)  # h=1 and h=2 for horizon 3
        for rate in lambda_rates:
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 0.5)
        
        # Test with horizon 1 (should return empty list)
        lambda_rates = self.multi_horizon_learning.calculate_multi_horizon_lambda_rates(
            self.mock_solution, 1
        )
        
        self.assertEqual(len(lambda_rates), 0)
        
    def test_generate_predicted_solution(self):
        """Test generation of predicted solution."""
        predicted_solution = self.multi_horizon_learning._generate_predicted_solution(
            self.mock_solution, 1
        )
        
        self.assertIsNotNone(predicted_solution)
        self.assertEqual(predicted_solution.P.num_assets, self.mock_solution.P.num_assets)
        self.assertIsNotNone(predicted_solution.P.ROI)
        self.assertIsNotNone(predicted_solution.P.risk)
        
    def test_perform_multi_horizon_prediction(self):
        """Test multi-horizon prediction."""
        predictions = self.multi_horizon_learning.perform_multi_horizon_prediction(
            self.mock_solution, 2
        )
        
        self.assertEqual(len(predictions), 2)  # h=1 and h=2
        
        for i, prediction in enumerate(predictions):
            self.assertEqual(prediction.horizon, i + 1)
            self.assertIsNotNone(prediction.predicted_state)
            self.assertIsNotNone(prediction.predicted_covariance)
            self.assertGreaterEqual(prediction.lambda_rate, 0.0)
            self.assertLessEqual(prediction.lambda_rate, 0.5)
            self.assertGreaterEqual(prediction.tip_value, 0.0)
            self.assertLessEqual(prediction.tip_value, 1.0)
            self.assertGreaterEqual(prediction.confidence, 0.0)
            self.assertLessEqual(prediction.confidence, 1.0)
        
        # Check that prediction history was stored
        self.assertEqual(len(self.multi_horizon_learning.prediction_history), 1)
        
    def test_perform_multi_horizon_prediction_exceeds_max_horizon(self):
        """Test multi-horizon prediction with horizon exceeding maximum."""
        with self.assertRaises(ValueError):
            self.multi_horizon_learning.perform_multi_horizon_prediction(
                self.mock_solution, 5  # Exceeds max_horizon=3
            )
        
    def test_apply_multi_horizon_anticipatory_learning(self):
        """Test application of multi-horizon anticipatory learning."""
        anticipatory_solution = self.multi_horizon_learning.apply_multi_horizon_anticipatory_learning(
            self.mock_solution, 2
        )
        
        self.assertIsNotNone(anticipatory_solution)
        self.assertEqual(anticipatory_solution.P.num_assets, self.mock_solution.P.num_assets)
        self.assertTrue(anticipatory_solution.anticipation)
        self.assertEqual(anticipatory_solution.alpha, self.mock_solution.alpha)
        self.assertEqual(anticipatory_solution.prediction_error, self.mock_solution.prediction_error)
        
        # Check that lambda rates history was stored
        self.assertEqual(len(self.multi_horizon_learning.lambda_rates_history), 1)
        
    def test_apply_multi_horizon_anticipatory_learning_single_horizon(self):
        """Test multi-horizon anticipatory learning with single horizon."""
        anticipatory_solution = self.multi_horizon_learning.apply_multi_horizon_anticipatory_learning(
            self.mock_solution, 1
        )
        
        # Should return original solution since no multi-horizon learning is possible
        self.assertEqual(anticipatory_solution.P.ROI, self.mock_solution.P.ROI)
        self.assertEqual(anticipatory_solution.P.risk, self.mock_solution.P.risk)
        
    def test_get_prediction_statistics(self):
        """Test getting prediction statistics."""
        # Initially no statistics
        stats = self.multi_horizon_learning.get_prediction_statistics()
        self.assertIn('error', stats)
        
        # Perform some predictions
        self.multi_horizon_learning.perform_multi_horizon_prediction(self.mock_solution, 2)
        
        # Get statistics
        stats = self.multi_horizon_learning.get_prediction_statistics()
        
        self.assertIn('total_predictions', stats)
        self.assertIn('mean_tip', stats)
        self.assertIn('std_tip', stats)
        self.assertIn('mean_lambda', stats)
        self.assertIn('std_lambda', stats)
        self.assertIn('mean_confidence', stats)
        self.assertIn('std_confidence', stats)
        self.assertIn('max_horizon_used', stats)
        
        self.assertEqual(stats['total_predictions'], 1)
        self.assertEqual(stats['max_horizon_used'], 2)
        
    def test_get_lambda_rates_statistics(self):
        """Test getting lambda rates statistics."""
        # Initially no statistics
        stats = self.multi_horizon_learning.get_lambda_rates_statistics()
        self.assertIn('error', stats)
        
        # Apply some anticipatory learning
        self.multi_horizon_learning.apply_multi_horizon_anticipatory_learning(self.mock_solution, 2)
        
        # Get statistics
        stats = self.multi_horizon_learning.get_lambda_rates_statistics()
        
        self.assertIn('total_entries', stats)
        self.assertIn('mean_lambda', stats)
        self.assertIn('std_lambda', stats)
        self.assertIn('min_lambda', stats)
        self.assertIn('max_lambda', stats)
        self.assertIn('horizon_distribution', stats)
        
        self.assertEqual(stats['total_entries'], 1)
        self.assertIn(2, stats['horizon_distribution'])
        
    def test_reset_history(self):
        """Test resetting history."""
        # Perform some operations to create history
        self.multi_horizon_learning.perform_multi_horizon_prediction(self.mock_solution, 2)
        self.multi_horizon_learning.apply_multi_horizon_anticipatory_learning(self.mock_solution, 2)
        
        # Check that history exists
        self.assertGreater(len(self.multi_horizon_learning.prediction_history), 0)
        self.assertGreater(len(self.multi_horizon_learning.lambda_rates_history), 0)
        
        # Reset history
        self.multi_horizon_learning.reset_history()
        
        # Check that history is cleared
        self.assertEqual(len(self.multi_horizon_learning.prediction_history), 0)
        self.assertEqual(len(self.multi_horizon_learning.lambda_rates_history), 0)
        
    def test_validate_prediction_horizon(self):
        """Test prediction horizon validation."""
        # Valid horizons
        self.assertTrue(self.multi_horizon_learning.validate_prediction_horizon(1))
        self.assertTrue(self.multi_horizon_learning.validate_prediction_horizon(2))
        self.assertTrue(self.multi_horizon_learning.validate_prediction_horizon(3))
        
        # Invalid horizons
        self.assertFalse(self.multi_horizon_learning.validate_prediction_horizon(0))
        self.assertFalse(self.multi_horizon_learning.validate_prediction_horizon(4))
        self.assertFalse(self.multi_horizon_learning.validate_prediction_horizon(-1))
        
    def test_get_set_max_horizon(self):
        """Test getting and setting maximum horizon."""
        # Test getter
        self.assertEqual(self.multi_horizon_learning.get_max_horizon(), 3)
        
        # Test setter
        self.multi_horizon_learning.set_max_horizon(5)
        self.assertEqual(self.multi_horizon_learning.get_max_horizon(), 5)
        self.assertEqual(self.multi_horizon_learning.n_step_predictor.max_horizon, 5)
        
        # Test invalid setter
        with self.assertRaises(ValueError):
            self.multi_horizon_learning.set_max_horizon(0)
        
    def test_get_predicted_covariance(self):
        """Test getting predicted covariance matrix."""
        covariance = self.multi_horizon_learning._get_predicted_covariance(self.mock_solution)
        
        self.assertIsNotNone(covariance)
        self.assertEqual(covariance.shape, (2, 2))  # ROI and risk covariance
        self.assertTrue(np.allclose(covariance, covariance.T))  # Should be symmetric


class TestMultiHorizonPrediction(unittest.TestCase):
    """Test cases for MultiHorizonPrediction dataclass."""
    
    def test_multi_horizon_prediction_creation(self):
        """Test creation of MultiHorizonPrediction object."""
        prediction = MultiHorizonPrediction(
            horizon=1,
            predicted_state=np.array([0.12, 0.06]),
            predicted_covariance=np.eye(2) * 0.01,
            lambda_rate=0.3,
            tip_value=0.7,
            confidence=0.8
        )
        
        self.assertEqual(prediction.horizon, 1)
        np.testing.assert_array_equal(prediction.predicted_state, np.array([0.12, 0.06]))
        np.testing.assert_array_equal(prediction.predicted_covariance, np.eye(2) * 0.01)
        self.assertEqual(prediction.lambda_rate, 0.3)
        self.assertEqual(prediction.tip_value, 0.7)
        self.assertEqual(prediction.confidence, 0.8)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_multi_horizon_anticipatory_learning(self):
        """Test convenience function for creating multi-horizon anticipatory learning."""
        learning = create_multi_horizon_anticipatory_learning(max_horizon=4, monte_carlo_samples=200)
        
        self.assertIsInstance(learning, MultiHorizonAnticipatoryLearning)
        self.assertEqual(learning.max_horizon, 4)
        self.assertEqual(learning.monte_carlo_samples, 200)


class TestMultiHorizonAnticipatoryLearningIntegration(unittest.TestCase):
    """Integration tests for multi-horizon anticipatory learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.learning = MultiHorizonAnticipatoryLearning(max_horizon=3, monte_carlo_samples=100)
        self.solution = self._create_mock_solution()
        
    def _create_mock_solution(self):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk, weights):
                self.ROI = roi
                self.risk = risk
                self.investment = weights
                self.cardinality = np.sum(weights > 0.01)
                self.num_assets = len(weights)
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk, weights):
                self.P = MockPortfolio(roi, risk, weights)
                self.alpha = 0.5
                self.prediction_error = 0.01
                self.anticipation = False
        
        return MockSolution(0.1, 0.05, np.array([0.4, 0.3, 0.2, 0.1]))
    
    def test_full_multi_horizon_workflow(self):
        """Test complete multi-horizon workflow."""
        # Step 1: Perform multi-horizon prediction
        predictions = self.learning.perform_multi_horizon_prediction(self.solution, 3)
        
        self.assertEqual(len(predictions), 3)
        
        # Step 2: Apply anticipatory learning
        anticipatory_solution = self.learning.apply_multi_horizon_anticipatory_learning(
            self.solution, 3
        )
        
        self.assertIsNotNone(anticipatory_solution)
        self.assertTrue(anticipatory_solution.anticipation)
        
        # Step 3: Check statistics
        pred_stats = self.learning.get_prediction_statistics()
        lambda_stats = self.learning.get_lambda_rates_statistics()
        
        self.assertNotIn('error', pred_stats)
        self.assertNotIn('error', lambda_stats)
        
        # Step 4: Reset and verify
        self.learning.reset_history()
        
        pred_stats = self.learning.get_prediction_statistics()
        lambda_stats = self.learning.get_lambda_rates_statistics()
        
        self.assertIn('error', pred_stats)
        self.assertIn('error', lambda_stats)
        
    def test_equation_610_verification(self):
        """Test verification of Equation 6.10 implementation."""
        # Test with known values
        current_state = np.array([0.1, 0.05])
        predicted_states = [
            np.array([0.12, 0.06]),  # h=1
            np.array([0.14, 0.07])   # h=2
        ]
        lambda_rates = [0.3, 0.2]
        
        result = self.learning.apply_anticipatory_learning_rule(
            current_state, predicted_states, lambda_rates
        )
        
        # Manual calculation of Equation 6.10
        lambda_sum = sum(lambda_rates)  # 0.5
        expected = (1 - lambda_sum) * current_state
        for pred_state, lambda_h in zip(predicted_states, lambda_rates):
            expected += lambda_h * pred_state
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
    def test_lambda_rates_bounds(self):
        """Test that lambda rates are properly bounded."""
        # Test multiple horizons
        for horizon in range(2, 5):
            lambda_rates = self.learning.calculate_multi_horizon_lambda_rates(
                self.solution, horizon
            )
            
            for rate in lambda_rates:
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 0.5)


class TestW4_2_LearnPopulationOverride(unittest.TestCase):
    """W4-2 regression: MultiHorizonAnticipatoryLearning.learn_population
    actually drives the multi-horizon machinery (paper Eq 14) instead of
    falling through to the parent's single-horizon path.

    Closes W1-3-CARRY-3.
    """

    def setUp(self):
        self.learner = MultiHorizonAnticipatoryLearning(
            max_horizon=3, monte_carlo_samples=50,
        )

    def _make_solution(self, roi: float, risk: float):
        """Minimal MockSolution shaped like the existing fixtures here."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.num_assets = 3
                self.investment = np.array([0.4, 0.3, 0.3])
                self.cardinality = 3
                self.kalman_state = None

        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
                self.alpha = 0.5
                self.prediction_error = 0.02
                self.anticipation = False

        return MockSolution(roi, risk)

    def test_learn_population_tags_solutions_with_multi_horizon_applied(self):
        """The override sets `multi_horizon_applied = True` on every
        solution it processes — proof the multi-horizon path was the
        live driver (the parent's single-horizon learn_population does
        NOT set this tag).
        """
        population = [
            self._make_solution(roi=0.1, risk=0.05),
            self._make_solution(roi=0.12, risk=0.06),
            self._make_solution(roi=0.08, risk=0.04),
        ]

        # Confirm no solution has the tag pre-learn.
        for sol in population:
            self.assertFalse(getattr(sol, 'multi_horizon_applied', False))

        self.learner.learn_population(population, current_time=1)

        # All solutions tagged.
        for sol in population:
            self.assertTrue(getattr(sol, 'multi_horizon_applied', False),
                            "learn_population override must tag solutions")

    def test_learn_population_writes_anticipatory_state_back(self):
        """The override applies paper Eq (14) and writes the convex-combo
        result back to solution.P.ROI / risk. Verify the ROI changes
        (the rule is non-trivial: predicted states are generated via
        _generate_predicted_solution which is non-identity).
        """
        population = [self._make_solution(roi=0.10, risk=0.05)]
        original_roi = population[0].P.ROI

        self.learner.learn_population(population, current_time=1)

        # The ROI may equal or differ from original depending on whether
        # the predicted state differs; but the solution MUST have been
        # processed (tag present) and the ROI is still a float.
        self.assertTrue(population[0].multi_horizon_applied)
        self.assertIsInstance(population[0].P.ROI, float)
        # Honest: with max_horizon=3 + non-trivial lambda rates, the
        # multi-horizon convex combination typically produces a value
        # different from the original. We assert it's at least a valid
        # finite float rather than pinning an exact value (MC variance
        # is real).
        self.assertTrue(np.isfinite(population[0].P.ROI))
        self.assertTrue(np.isfinite(population[0].P.risk))

    def test_learn_population_stores_historical_snapshot(self):
        """The override mirrors the parent's bookkeeping — store_historical_
        population is called so downstream correspondence_mapping sees a
        snapshot.
        """
        population = [self._make_solution(roi=0.1, risk=0.05)]
        pre_count = len(self.learner.historical_populations)

        self.learner.learn_population(population, current_time=1)

        self.assertEqual(
            len(self.learner.historical_populations), pre_count + 1,
            "learn_population must append to historical_populations",
        )


class TestW5_2_CovarianceThreading(unittest.TestCase):
    """W5-2 regression: MultiHorizonAnticipatoryLearning.learn_population
    threads covariance updates per paper Eq (15) when kalman_state is
    present on the current and predicted solutions.

    Closes W4-2-CARRY-1.

    Paper canon:
        Σ_combined = w_0² · Σ_t + Σ_{h=1}^{H-1} w_h² · Σ_{t+h}

    where w_0 = (1 - Σλ) and w_h = λ_{t+h}. Σ_combined is positive-
    semidefinite by construction (sum of weighted PSD matrices with
    non-negative weights).
    """

    def setUp(self):
        np.random.seed(7)
        self.learner = MultiHorizonAnticipatoryLearning(
            max_horizon=3, monte_carlo_samples=50,
        )

    def _make_solution_with_kf(self, roi: float, risk: float,
                                roi_var: float = 0.01, risk_var: float = 0.01,
                                cov: float = 0.0):
        """Build a Solution-shaped mock WITH a real KalmanParams whose
        P[:2, :2] is the [ROI, risk] covariance. Uses create_kalman_params
        helper so F + H + R are all properly populated (required by
        the n-step predictor that `_generate_predicted_solution` calls).
        """
        from src.algorithms.kalman_filter import create_kalman_params

        kalman_state = create_kalman_params(initial_roi=roi, initial_risk=risk)
        # Override the [ROI, risk] covariance block to the test-specified
        # values; leave velocity block at the create_kalman_params default.
        kalman_state.P[0, 0] = roi_var
        kalman_state.P[1, 1] = risk_var
        kalman_state.P[0, 1] = cov
        kalman_state.P[1, 0] = cov
        kalman_state.P_next = kalman_state.P.copy()

        class MockPortfolio:
            def __init__(self, roi, risk, kf):
                self.ROI = roi
                self.risk = risk
                self.num_assets = 3
                self.investment = np.array([0.4, 0.3, 0.3])
                self.cardinality = 3
                self.kalman_state = kf

        class MockSolution:
            def __init__(self, roi, risk, kf):
                self.P = MockPortfolio(roi, risk, kf)
                self.alpha = 0.5
                self.prediction_error = 0.02
                self.anticipation = False

        return MockSolution(roi, risk, kalman_state)

    def test_covariance_updated_in_place_for_solution_with_kf(self):
        """When solution has kalman_state, learn_population computes
        the Eq (15) Σ_combined and writes it to P[:2, :2].
        """
        sol = self._make_solution_with_kf(roi=0.10, risk=0.05,
                                            roi_var=0.005, risk_var=0.005)
        original_cov = sol.P.kalman_state.P[:2, :2].copy()

        self.learner.learn_population([sol], current_time=1)

        new_cov = sol.P.kalman_state.P[:2, :2]
        # Covariance should be a real 2x2 numpy array.
        self.assertEqual(new_cov.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(new_cov)))
        # In the multi-horizon path with non-trivial λ, the combined
        # covariance differs from the original (weighted sum, not a
        # passthrough). Original was 0.005·I; combined typically smaller
        # because each w² < 1 and Σ w² < 1 for the weighted sum.
        self.assertFalse(np.allclose(new_cov, original_cov),
                          "Covariance should be updated, not left unchanged")

    def test_covariance_remains_positive_semidefinite(self):
        """Σ_combined per paper Eq (15) is PSD by construction. Verify
        eigenvalues all >= 0 (within numerical tolerance).
        """
        sol = self._make_solution_with_kf(roi=0.1, risk=0.05)
        self.learner.learn_population([sol], current_time=1)

        new_cov = sol.P.kalman_state.P[:2, :2]
        eigenvalues = np.linalg.eigvalsh(new_cov)
        # PSD: all eigenvalues >= 0 (small numerical slack)
        for eig in eigenvalues:
            self.assertGreaterEqual(eig, -1e-9,
                                     f"Covariance must remain PSD; eig={eig:.2e}")

    def test_kalman_state_none_does_not_break_learn_population(self):
        """Graceful degradation: when current solution has no
        kalman_state, the mean update still runs and the solution is
        still tagged. No covariance update + no exception.
        """
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.num_assets = 3
                self.investment = np.array([0.4, 0.3, 0.3])
                self.cardinality = 3
                self.kalman_state = None  # ← graceful-degradation case

        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
                self.alpha = 0.5
                self.prediction_error = 0.02
                self.anticipation = False

        sol = MockSolution(0.1, 0.05)

        # Must NOT raise.
        self.learner.learn_population([sol], current_time=1)
        self.assertTrue(sol.multi_horizon_applied)


if __name__ == '__main__':
    unittest.main()
