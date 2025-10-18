"""
Unit tests for Belief Coefficient Self-Adjustment

Tests the belief coefficient self-adjustment implementation including
Equation 6.30 and TIP-based confidence calculation.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.belief_coefficient import (
    BeliefCoefficientCalculator, BeliefCoefficientResult,
    create_belief_coefficient_calculator
)


class TestBeliefCoefficientCalculator(unittest.TestCase):
    """Test cases for BeliefCoefficientCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = BeliefCoefficientCalculator(monte_carlo_samples=100)
        
        # Create mock solutions for testing
        self.current_solution = self._create_mock_solution(0.1, 0.05)
        self.predicted_solution = self._create_mock_solution(0.12, 0.06)
        
    def _create_mock_solution(self, roi: float, risk: float):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.num_assets = 4
                self.investment = np.array([0.4, 0.3, 0.2, 0.1])
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
                self.alpha = 0.5
                self.prediction_error = 0.01
                self.anticipation = False
        
        return MockSolution(roi, risk)
    
    def test_initialization(self):
        """Test initialization of belief coefficient calculator."""
        self.assertEqual(self.calculator.monte_carlo_samples, 100)
        self.assertIsNotNone(self.calculator.tip_calculator)
        self.assertEqual(len(self.calculator.belief_coefficient_history), 0)
        self.assertEqual(len(self.calculator.tip_history), 0)
        self.assertEqual(len(self.calculator.entropy_history), 0)
        
    def test_binary_entropy(self):
        """Test binary entropy calculation."""
        # Test maximum entropy (p = 0.5)
        entropy = self.calculator._binary_entropy(0.5)
        self.assertAlmostEqual(entropy, 1.0, places=5)
        
        # Test minimum entropy (p = 0 or p = 1)
        entropy_0 = self.calculator._binary_entropy(0.0)
        entropy_1 = self.calculator._binary_entropy(1.0)
        self.assertAlmostEqual(entropy_0, 0.0, places=5)
        self.assertAlmostEqual(entropy_1, 0.0, places=5)
        
        # Test intermediate values
        entropy_0_25 = self.calculator._binary_entropy(0.25)
        entropy_0_75 = self.calculator._binary_entropy(0.75)
        self.assertGreater(entropy_0_25, 0.0)
        self.assertLess(entropy_0_25, 1.0)
        self.assertAlmostEqual(entropy_0_25, entropy_0_75, places=5)  # Should be symmetric
        
        # Test edge cases
        entropy_negative = self.calculator._binary_entropy(-0.1)
        entropy_above_1 = self.calculator._binary_entropy(1.1)
        self.assertAlmostEqual(entropy_negative, 0.0, places=5)
        self.assertAlmostEqual(entropy_above_1, 0.0, places=5)
        
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        # Test with TIP = 0.5 (maximum uncertainty)
        confidence = self.calculator._calculate_confidence(0.5, 1.0)
        self.assertLess(confidence, 0.5)  # Low confidence for high uncertainty
        
        # Test with TIP = 0.0 or 1.0 (minimum uncertainty)
        confidence_0 = self.calculator._calculate_confidence(0.0, 0.0)
        confidence_1 = self.calculator._calculate_confidence(1.0, 0.0)
        self.assertGreater(confidence_0, 0.5)  # High confidence for low uncertainty
        self.assertGreater(confidence_1, 0.5)  # High confidence for low uncertainty
        
        # Test bounds
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
    def test_calculate_belief_coefficient(self):
        """Test belief coefficient calculation (Equation 6.30)."""
        result = self.calculator.calculate_belief_coefficient(
            self.current_solution, self.predicted_solution
        )
        
        # Check result structure
        self.assertIsInstance(result, BeliefCoefficientResult)
        self.assertIsNotNone(result.belief_coefficient)
        self.assertIsNotNone(result.tip_value)
        self.assertIsNotNone(result.binary_entropy)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.timestamp)
        
        # Check bounds
        self.assertGreaterEqual(result.belief_coefficient, 0.5)
        self.assertLessEqual(result.belief_coefficient, 1.0)
        self.assertGreaterEqual(result.tip_value, 0.0)
        self.assertLessEqual(result.tip_value, 1.0)
        self.assertGreaterEqual(result.binary_entropy, 0.0)
        self.assertLessEqual(result.binary_entropy, 1.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
        
        # Check that history was stored
        self.assertEqual(len(self.calculator.belief_coefficient_history), 1)
        self.assertEqual(len(self.calculator.tip_history), 1)
        self.assertEqual(len(self.calculator.entropy_history), 1)
        
    def test_equation_630_verification(self):
        """Test verification of Equation 6.30 implementation."""
        result = self.calculator.calculate_belief_coefficient(
            self.current_solution, self.predicted_solution
        )
        
        # Manual calculation of Equation 6.30
        expected_belief_coefficient = 1.0 - 0.5 * result.binary_entropy
        
        # Check that the calculation matches Equation 6.30
        self.assertAlmostEqual(result.belief_coefficient, expected_belief_coefficient, places=6)
        
        # Test with different TIP values
        test_tips = [0.0, 0.25, 0.5, 0.75, 1.0]
        for tip in test_tips:
            entropy = self.calculator._binary_entropy(tip)
            expected_bc = 1.0 - 0.5 * entropy
            expected_bc = max(0.5, min(1.0, expected_bc))
            
            # Verify the formula
            self.assertAlmostEqual(expected_bc, 1.0 - 0.5 * entropy, places=6)
            
    def test_calculate_adaptive_belief_coefficient(self):
        """Test adaptive belief coefficient calculation."""
        # First, create some history
        for i in range(15):
            result = self.calculator.calculate_belief_coefficient(
                self.current_solution, self.predicted_solution
            )
        
        # Test adaptive calculation
        adaptive_result = self.calculator.calculate_adaptive_belief_coefficient(
            self.current_solution, self.predicted_solution, historical_window=10
        )
        
        self.assertIsInstance(adaptive_result, BeliefCoefficientResult)
        self.assertGreaterEqual(adaptive_result.belief_coefficient, 0.5)
        self.assertLessEqual(adaptive_result.belief_coefficient, 1.0)
        
    def test_calculate_trend(self):
        """Test trend calculation."""
        # Test increasing trend
        increasing_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        trend = self.calculator._calculate_trend(increasing_values)
        self.assertGreater(trend, 0.0)
        
        # Test decreasing trend
        decreasing_values = [0.5, 0.4, 0.3, 0.2, 0.1]
        trend = self.calculator._calculate_trend(decreasing_values)
        self.assertLess(trend, 0.0)
        
        # Test constant values
        constant_values = [0.3, 0.3, 0.3, 0.3, 0.3]
        trend = self.calculator._calculate_trend(constant_values)
        self.assertAlmostEqual(trend, 0.0, places=5)
        
        # Test single value
        single_value = [0.5]
        trend = self.calculator._calculate_trend(single_value)
        self.assertAlmostEqual(trend, 0.0, places=5)
        
    def test_calculate_belief_coefficient_for_horizon(self):
        """Test belief coefficient calculation for specific horizon."""
        result = self.calculator.calculate_belief_coefficient_for_horizon(
            self.current_solution, self.predicted_solution, horizon=2
        )
        
        self.assertIsInstance(result, BeliefCoefficientResult)
        self.assertGreaterEqual(result.belief_coefficient, 0.5)
        self.assertLessEqual(result.belief_coefficient, 1.0)
        
        # Test different horizons
        for horizon in range(1, 4):
            result = self.calculator.calculate_belief_coefficient_for_horizon(
                self.current_solution, self.predicted_solution, horizon=horizon
            )
            self.assertGreaterEqual(result.belief_coefficient, 0.5)
            self.assertLessEqual(result.belief_coefficient, 1.0)
            
    def test_get_belief_coefficient_statistics(self):
        """Test getting belief coefficient statistics."""
        # Initially no statistics
        stats = self.calculator.get_belief_coefficient_statistics()
        self.assertIn('error', stats)
        
        # Create some history
        for i in range(5):
            self.calculator.calculate_belief_coefficient(
                self.current_solution, self.predicted_solution
            )
        
        # Get statistics
        stats = self.calculator.get_belief_coefficient_statistics()
        
        self.assertIn('total_calculations', stats)
        self.assertIn('mean_belief_coefficient', stats)
        self.assertIn('std_belief_coefficient', stats)
        self.assertIn('min_belief_coefficient', stats)
        self.assertIn('max_belief_coefficient', stats)
        self.assertIn('mean_tip', stats)
        self.assertIn('mean_entropy', stats)
        self.assertIn('mean_confidence', stats)
        self.assertIn('recent_trend', stats)
        
        self.assertEqual(stats['total_calculations'], 5)
        
    def test_get_recent_belief_coefficients(self):
        """Test getting recent belief coefficient results."""
        # Initially empty
        recent = self.calculator.get_recent_belief_coefficients(5)
        self.assertEqual(len(recent), 0)
        
        # Create some history
        for i in range(10):
            self.calculator.calculate_belief_coefficient(
                self.current_solution, self.predicted_solution
            )
        
        # Get recent results
        recent = self.calculator.get_recent_belief_coefficients(5)
        self.assertEqual(len(recent), 5)
        
        # Get more than available
        recent = self.calculator.get_recent_belief_coefficients(15)
        self.assertEqual(len(recent), 10)
        
    def test_reset_history(self):
        """Test resetting history."""
        # Create some history
        for i in range(5):
            self.calculator.calculate_belief_coefficient(
                self.current_solution, self.predicted_solution
            )
        
        # Check that history exists
        self.assertGreater(len(self.calculator.belief_coefficient_history), 0)
        self.assertGreater(len(self.calculator.tip_history), 0)
        self.assertGreater(len(self.calculator.entropy_history), 0)
        
        # Reset history
        self.calculator.reset_history()
        
        # Check that history is cleared
        self.assertEqual(len(self.calculator.belief_coefficient_history), 0)
        self.assertEqual(len(self.calculator.tip_history), 0)
        self.assertEqual(len(self.calculator.entropy_history), 0)
        
    def test_validate_belief_coefficient(self):
        """Test belief coefficient validation."""
        # Valid values
        self.assertTrue(self.calculator.validate_belief_coefficient(0.5))
        self.assertTrue(self.calculator.validate_belief_coefficient(0.7))
        self.assertTrue(self.calculator.validate_belief_coefficient(1.0))
        
        # Invalid values
        self.assertFalse(self.calculator.validate_belief_coefficient(0.4))
        self.assertFalse(self.calculator.validate_belief_coefficient(1.1))
        self.assertFalse(self.calculator.validate_belief_coefficient(-0.1))
        
    def test_get_equation_630_verification(self):
        """Test Equation 6.30 verification method."""
        verification = self.calculator.get_equation_630_verification(
            self.current_solution, self.predicted_solution
        )
        
        self.assertIn('tip_value', verification)
        self.assertIn('binary_entropy', verification)
        self.assertIn('belief_coefficient', verification)
        self.assertIn('expected_belief_coefficient', verification)
        self.assertIn('equation_630_verified', verification)
        self.assertIn('formula', verification)
        self.assertIn('calculation', verification)
        
        self.assertTrue(verification['equation_630_verified'])
        self.assertAlmostEqual(
            verification['belief_coefficient'], 
            verification['expected_belief_coefficient'], 
            places=6
        )


class TestBeliefCoefficientResult(unittest.TestCase):
    """Test cases for BeliefCoefficientResult dataclass."""
    
    def test_belief_coefficient_result_creation(self):
        """Test creation of BeliefCoefficientResult object."""
        result = BeliefCoefficientResult(
            belief_coefficient=0.8,
            tip_value=0.7,
            binary_entropy=0.4,
            confidence=0.9,
            timestamp=1234567890.0
        )
        
        self.assertEqual(result.belief_coefficient, 0.8)
        self.assertEqual(result.tip_value, 0.7)
        self.assertEqual(result.binary_entropy, 0.4)
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.timestamp, 1234567890.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_belief_coefficient_calculator(self):
        """Test convenience function for creating belief coefficient calculator."""
        calculator = create_belief_coefficient_calculator(monte_carlo_samples=200)
        
        self.assertIsInstance(calculator, BeliefCoefficientCalculator)
        self.assertEqual(calculator.monte_carlo_samples, 200)


class TestBeliefCoefficientIntegration(unittest.TestCase):
    """Integration tests for belief coefficient calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = BeliefCoefficientCalculator(monte_carlo_samples=100)
        self.current_solution = self._create_mock_solution(0.1, 0.05)
        self.predicted_solution = self._create_mock_solution(0.12, 0.06)
        
    def _create_mock_solution(self, roi: float, risk: float):
        """Create a mock solution for testing."""
        class MockPortfolio:
            def __init__(self, roi, risk):
                self.ROI = roi
                self.risk = risk
                self.num_assets = 4
                self.investment = np.array([0.4, 0.3, 0.2, 0.1])
                self.kalman_state = None
        
        class MockSolution:
            def __init__(self, roi, risk):
                self.P = MockPortfolio(roi, risk)
                self.alpha = 0.5
                self.prediction_error = 0.01
                self.anticipation = False
        
        return MockSolution(roi, risk)
    
    def test_full_belief_coefficient_workflow(self):
        """Test complete belief coefficient workflow."""
        # Step 1: Calculate belief coefficient
        result = self.calculator.calculate_belief_coefficient(
            self.current_solution, self.predicted_solution
        )
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.belief_coefficient, 0.5)
        self.assertLessEqual(result.belief_coefficient, 1.0)
        
        # Step 2: Get statistics
        stats = self.calculator.get_belief_coefficient_statistics()
        self.assertNotIn('error', stats)
        self.assertEqual(stats['total_calculations'], 1)
        
        # Step 3: Test adaptive calculation
        adaptive_result = self.calculator.calculate_adaptive_belief_coefficient(
            self.current_solution, self.predicted_solution
        )
        
        self.assertIsNotNone(adaptive_result)
        
        # Step 4: Test horizon-specific calculation
        horizon_result = self.calculator.calculate_belief_coefficient_for_horizon(
            self.current_solution, self.predicted_solution, horizon=2
        )
        
        self.assertIsNotNone(horizon_result)
        
        # Step 5: Verify Equation 6.30
        verification = self.calculator.get_equation_630_verification(
            self.current_solution, self.predicted_solution
        )
        
        self.assertTrue(verification['equation_630_verified'])
        
    def test_equation_630_mathematical_verification(self):
        """Test mathematical verification of Equation 6.30."""
        # Test with known values
        test_cases = [
            (0.0, 0.0, 1.0),  # TIP=0, Entropy=0, BC=1.0
            (1.0, 0.0, 1.0),  # TIP=1, Entropy=0, BC=1.0
            (0.5, 1.0, 0.5),  # TIP=0.5, Entropy=1.0, BC=0.5
        ]
        
        for tip, expected_entropy, expected_bc in test_cases:
            entropy = self.calculator._binary_entropy(tip)
            belief_coefficient = 1.0 - 0.5 * entropy
            
            self.assertAlmostEqual(entropy, expected_entropy, places=5)
            self.assertAlmostEqual(belief_coefficient, expected_bc, places=5)
            
    def test_belief_coefficient_bounds(self):
        """Test that belief coefficients are properly bounded."""
        # Test multiple calculations
        for i in range(10):
            result = self.calculator.calculate_belief_coefficient(
                self.current_solution, self.predicted_solution
            )
            
            self.assertGreaterEqual(result.belief_coefficient, 0.5)
            self.assertLessEqual(result.belief_coefficient, 1.0)
            self.assertGreaterEqual(result.tip_value, 0.0)
            self.assertLessEqual(result.tip_value, 1.0)
            self.assertGreaterEqual(result.binary_entropy, 0.0)
            self.assertLessEqual(result.binary_entropy, 1.0)
            self.assertGreaterEqual(result.confidence, 0.0)
            self.assertLessEqual(result.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()



