"""
Integration tests for TIP integration with Anticipatory Learning

Tests the integration of Temporal Incomparability Probability (TIP) with
the main anticipatory learning algorithm.
"""

import unittest
import numpy as np

# W1-2 import-style fix: use `from src.algorithms.X` to match the
# pattern that test_kalman_filter.py uses, which collects cleanly under
# pytest. The previous `sys.path.insert` + `from algorithms.X` pattern
# was one of the 17 collection errors flagged in the 2026-05-16 audit
# (the deeper relative-import bug in src/algorithms/solution.py is
# W1-3 scope and is unaffected by this surface fix). After W1-3 lands
# the relative-import refactor, this file can drop the `src.` prefix
# in favour of a fully package-aware layout.
from src.algorithms.anticipatory_learning import TIPIntegratedAnticipatoryLearning
from src.algorithms.temporal_incomparability_probability import TemporalIncomparabilityCalculator


class TestTIPIntegration(unittest.TestCase):
    """Test cases for TIP integration with anticipatory learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.anticipatory_learning = TIPIntegratedAnticipatoryLearning(window_size=5, monte_carlo_samples=100)
        
        # Create mock solutions for testing
        self.current_solution = self._create_mock_solution(0.1, 0.05, 0.5, 0.02)
        self.predicted_solution = self._create_mock_solution(0.12, 0.06, 0.6, 0.03)
        
    def _create_mock_solution(self, roi: float, risk: float, alpha: float, prediction_error: float):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk, alpha, prediction_error):
                self.P = MockPortfolio(roi, risk)
                self.alpha = alpha
                self.prediction_error = prediction_error
        
        return MockSolution(roi, risk, alpha, prediction_error)
    
    def test_initialization(self):
        """Test TIP-integrated anticipatory learning initialization."""
        self.assertIsInstance(self.anticipatory_learning.tip_calculator, TemporalIncomparabilityCalculator)
        self.assertEqual(self.anticipatory_learning.prediction_horizon, 2)
        self.assertEqual(self.anticipatory_learning.window_size, 5)
        
    def test_set_prediction_horizon(self):
        """Test setting prediction horizon."""
        self.anticipatory_learning.set_prediction_horizon(3)
        self.assertEqual(self.anticipatory_learning.prediction_horizon, 3)
        
    def test_enhanced_anticipatory_learning_rate_calculation(self):
        """Test enhanced anticipatory learning rate calculation."""
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        combined_rate, traditional_rate, tip_rate = self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
            self.current_solution, self.predicted_solution,
            min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # All rates should be between 0 and 1
        self.assertGreaterEqual(combined_rate, 0.0)
        self.assertLessEqual(combined_rate, 1.0)
        self.assertGreaterEqual(traditional_rate, 0.0)
        self.assertLessEqual(traditional_rate, 1.0)
        self.assertGreaterEqual(tip_rate, 0.0)
        self.assertLessEqual(tip_rate, 1.0)
        
        # Combined rate should be the average of traditional and TIP rates
        expected_combined = 0.5 * (traditional_rate + tip_rate)
        self.assertAlmostEqual(combined_rate, expected_combined, places=5)
        
    def test_enhanced_learning_rate_with_different_solutions(self):
        """Test enhanced learning rate with different solution characteristics."""
        # Test with solutions that have different dominance relationships
        current_dominates = self._create_mock_solution(0.15, 0.03, 0.5, 0.02)  # Higher ROI, lower risk
        predicted_dominated = self._create_mock_solution(0.10, 0.05, 0.6, 0.03)
        
        current_dominated = self._create_mock_solution(0.10, 0.05, 0.5, 0.02)
        predicted_dominates = self._create_mock_solution(0.15, 0.03, 0.6, 0.03)
        
        current_non_dom = self._create_mock_solution(0.12, 0.04, 0.5, 0.02)  # Higher ROI, higher risk
        predicted_non_dom = self._create_mock_solution(0.10, 0.03, 0.6, 0.03)  # Lower ROI, lower risk
        
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        # Test different scenarios
        scenarios = [
            (current_dominates, predicted_dominated, "current dominates"),
            (current_dominated, predicted_dominates, "predicted dominates"),
            (current_non_dom, predicted_non_dom, "mutually non-dominated")
        ]
        
        for current, predicted, scenario_name in scenarios:
            with self.subTest(scenario=scenario_name):
                combined_rate, traditional_rate, tip_rate = self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
                    current, predicted, min_error, max_error, min_alpha, max_alpha, current_time
                )
                
                # All rates should be valid
                self.assertGreaterEqual(combined_rate, 0.0)
                self.assertLessEqual(combined_rate, 1.0)
                self.assertGreaterEqual(traditional_rate, 0.0)
                self.assertLessEqual(traditional_rate, 1.0)
                self.assertGreaterEqual(tip_rate, 0.0)
                self.assertLessEqual(tip_rate, 1.0)
                
    def test_tip_statistics(self):
        """Test TIP statistics retrieval."""
        # Calculate some learning rates to generate TIP history
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        for _ in range(5):
            self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
                self.current_solution, self.predicted_solution,
                min_error, max_error, min_alpha, max_alpha, current_time
            )
        
        stats = self.anticipatory_learning.get_tip_statistics()
        
        self.assertIn('count', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('trend', stats)
        
        self.assertEqual(stats['count'], 5)
        self.assertGreaterEqual(stats['mean'], 0.0)
        self.assertLessEqual(stats['mean'], 1.0)
        
    def test_reset_tip_history(self):
        """Test TIP history reset functionality."""
        # Calculate some learning rates to generate TIP history
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        for _ in range(3):
            self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
                self.current_solution, self.predicted_solution,
                min_error, max_error, min_alpha, max_alpha, current_time
            )
        
        # Verify history exists
        stats_before = self.anticipatory_learning.get_tip_statistics()
        self.assertEqual(stats_before['count'], 3)
        
        # Reset history
        self.anticipatory_learning.reset_tip_history()
        
        # Verify history is cleared
        stats_after = self.anticipatory_learning.get_tip_statistics()
        self.assertEqual(stats_after['count'], 0)
        
    def test_learning_rate_comparison(self):
        """Test comparison between traditional and enhanced learning rates."""
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        # Calculate enhanced rate
        enhanced_rate, traditional_rate, tip_rate = self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
            self.current_solution, self.predicted_solution,
            min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # Calculate traditional rate directly
        traditional_rate_direct = self.anticipatory_learning._compute_traditional_learning_rate(
            self.current_solution, min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # Traditional rates should be the same
        self.assertAlmostEqual(traditional_rate, traditional_rate_direct, places=5)
        
        # Enhanced rate should be the average of traditional and TIP rates
        expected_enhanced = 0.5 * (traditional_rate + tip_rate)
        self.assertAlmostEqual(enhanced_rate, expected_enhanced, places=5)
        
    def test_edge_cases(self):
        """Test edge cases for TIP integration."""
        # Test with extreme values
        extreme_current = self._create_mock_solution(0.0, 0.0, 0.0, 0.0)
        extreme_predicted = self._create_mock_solution(1.0, 1.0, 1.0, 1.0)
        
        min_error, max_error = 0.0, 1.0
        min_alpha, max_alpha = 0.0, 1.0
        current_time = 0
        
        combined_rate, traditional_rate, tip_rate = self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
            extreme_current, extreme_predicted,
            min_error, max_error, min_alpha, max_alpha, current_time
        )
        
        # All rates should be valid even with extreme values
        self.assertGreaterEqual(combined_rate, 0.0)
        self.assertLessEqual(combined_rate, 1.0)
        self.assertGreaterEqual(traditional_rate, 0.0)
        self.assertLessEqual(traditional_rate, 1.0)
        self.assertGreaterEqual(tip_rate, 0.0)
        self.assertLessEqual(tip_rate, 1.0)
        
    def test_prediction_horizon_effect(self):
        """Test effect of different prediction horizons on learning rates."""
        min_error, max_error = 0.01, 0.05
        min_alpha, max_alpha = 0.3, 0.8
        current_time = 5
        
        horizons = [1, 2, 3, 5]
        rates_by_horizon = {}
        
        for horizon in horizons:
            self.anticipatory_learning.set_prediction_horizon(horizon)
            combined_rate, traditional_rate, tip_rate = self.anticipatory_learning.compute_enhanced_anticipatory_learning_rate(
                self.current_solution, self.predicted_solution,
                min_error, max_error, min_alpha, max_alpha, current_time
            )
            
            rates_by_horizon[horizon] = {
                'combined': combined_rate,
                'traditional': traditional_rate,
                'tip': tip_rate
            }
        
        # Traditional rate should be the same for all horizons
        traditional_rates = [rates_by_horizon[h]['traditional'] for h in horizons]
        self.assertTrue(all(abs(tr - traditional_rates[0]) < 1e-10 for tr in traditional_rates))
        
        # TIP rate should vary with horizon (horizon=1 should give 0)
        self.assertEqual(rates_by_horizon[1]['tip'], 0.0)
        
        # For other horizons, TIP rate should be positive
        for horizon in [2, 3, 5]:
            self.assertGreater(rates_by_horizon[horizon]['tip'], 0.0)


class TestW1_2_TIPWiring(unittest.TestCase):
    """W1-2 regression tests for the live-path TIP wiring.

    Anchors:
      * Paper Eq (12) — TIP definition
      * Paper Eq (13) — λ^(H) from binary entropy of TIP
      * Thesis Eq 7.16 (= paper-only-extension) — (1/2)(λ^(H) + λ^(K)) blend

    These tests pin two pre-W1-2 bugs against regression:
      1. `TIPIntegratedAnticipatoryLearning(window_size=10)` silently
         set `base_learning_rate = 10.0` because `super().__init__` was
         called with `window_size` as a positional arg in a position
         the parent treats as `learning_rate`.
      2. `compute_anticipatory_learning_rate` had the TIP arm wired
         correctly internally but `anticipatory_learning_obj_space`
         never threaded `tip_calculator` through, so the live published
         λ always fell back to the KF-residuals-only path.
    """

    def test_super_init_does_not_miswire_window_size_as_learning_rate(self):
        """W1-2 fix: super().__init__ uses keyword `window_size`, not positional."""
        learner = TIPIntegratedAnticipatoryLearning(window_size=10, monte_carlo_samples=50)

        # Before W1-2, `super().__init__(window_size)` set
        # base_learning_rate = 10.0 (because the parent's first positional
        # arg is `learning_rate: float = 0.01`).
        # After W1-2 (keyword `window_size=window_size`), base_learning_rate
        # should be the parent's default of 0.01.
        self.assertNotEqual(
            learner.base_learning_rate, 10.0,
            "TIPIntegratedAnticipatoryLearning(window_size=10) must not set "
            "base_learning_rate=10.0 (the pre-W1-2 super()-miswire bug)."
        )
        self.assertAlmostEqual(learner.base_learning_rate, 0.01, places=6)
        # window_size should still be the requested value.
        self.assertEqual(learner.window_size, 10)

    def test_tip_arm_fires_when_tip_calculator_is_threaded(self):
        """Pin: passing tip_calculator + predicted_solution makes the
        combined rate differ from the traditional (KF-only) rate.

        This proves the TIP arm (paper Eq 13) actually fires in the
        path used by `anticipatory_learning_obj_space` — closing the
        audit's "wired but divergent" finding for TIP integration.
        """
        from src.algorithms.anticipatory_learning import AnticipatoryLearning

        np.random.seed(42)

        # Build two solutions whose ROI/risk differ enough that the
        # TIP calculator (with covariance fallback) returns a value
        # away from 0.5, making λ^(H) non-zero per paper Eq (13).
        # Use the same MockSolution pattern as the existing tests above.
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.kalman_state = None  # forces MC fallback path

        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
                self.alpha = 0.5
                self.prediction_error = 0.02

        current = MockSolution(roi=0.10, risk=0.05)
        predicted = MockSolution(roi=0.30, risk=0.20)  # very different → TIP near 0

        base = AnticipatoryLearning(window_size=20, prediction_horizon=2)
        # Drive current_time>0 path so the rate isn't trivially zero.
        traditional = base.compute_anticipatory_learning_rate(
            current, min_error=0.0, max_error=0.1,
            min_alpha=0.0, max_alpha=1.0, current_time=5,
        )

        tip_calc = TemporalIncomparabilityCalculator(monte_carlo_samples=200)
        tip_integrated = TIPIntegratedAnticipatoryLearning(
            window_size=20, monte_carlo_samples=200,
        )
        combined = tip_integrated.compute_anticipatory_learning_rate(
            current, min_error=0.0, max_error=0.1,
            min_alpha=0.0, max_alpha=1.0, current_time=5,
            tip_calculator=tip_calc, predicted_solution=predicted, horizon=2,
        )

        # When TIP arm fires with a non-degenerate predicted_solution,
        # the combined rate differs from the traditional rate. If the
        # threading were broken (pre-W1-2 behaviour), both calls would
        # produce identical values via the KF-only fallback.
        self.assertNotAlmostEqual(
            combined, traditional, places=6,
            msg="Combined rate (paper Eq 7.16 blend) must differ from "
                "the traditional KF-only rate when tip_calculator is "
                "threaded — otherwise the TIP arm never fires."
        )


if __name__ == '__main__':
    unittest.main()
