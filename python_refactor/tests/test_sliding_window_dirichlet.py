"""
Unit tests for Sliding Window Dirichlet Model

Tests the implementation of Equations 6.24-6.27 and related functionality.
"""

import unittest
import numpy as np

# W2-2 import-style fix: use canonical `from src.algorithms.X` (W1-2/W1-3/W1-4 pattern).
from src.algorithms.sliding_window_dirichlet import SlidingWindowDirichlet


class TestSlidingWindowDirichlet(unittest.TestCase):
    """Test cases for SlidingWindowDirichlet class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.window_size = 5
        self.scaling_factor = 1.0
        self.model = SlidingWindowDirichlet(self.window_size, self.scaling_factor)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.get_window_size(), 5)
        self.assertEqual(self.model.get_scaling_factor(), 1.0)
        self.assertIsNone(self.model.get_current_concentration())
        
    def test_equation_624_accumulating_phase(self):
        """Test Equation 6.24: Accumulating observations (t < K)."""
        # Test t=0 (initialization)
        u_0 = np.array([0.3, 0.4, 0.3])
        alpha_0 = self.model.update_concentration(0, u_0)
        
        expected_alpha_0 = self.scaling_factor * np.ones_like(u_0) / len(u_0)
        np.testing.assert_array_almost_equal(alpha_0, expected_alpha_0)
        
        # Test t=1 (Equation 6.24)
        u_1 = np.array([0.2, 0.5, 0.3])
        alpha_1 = self.model.update_concentration(1, u_1)
        
        expected_alpha_1 = alpha_0 + self.scaling_factor * u_1
        np.testing.assert_array_almost_equal(alpha_1, expected_alpha_1)
        
        # Test t=2 (Equation 6.24)
        u_2 = np.array([0.4, 0.3, 0.3])
        alpha_2 = self.model.update_concentration(2, u_2)
        
        expected_alpha_2 = alpha_1 + self.scaling_factor * u_2
        np.testing.assert_array_almost_equal(alpha_2, expected_alpha_2)
        
    def test_equation_625_first_full_window(self):
        """Test Equation 6.25: First time window is full (t = K)."""
        # Build up to t=K
        for t in range(self.window_size):
            u_t = np.array([0.3, 0.4, 0.3]) + 0.1 * t * np.array([1, -1, 0])
            self.model.update_concentration(t, u_t)
        
        # Test t=K (Equation 6.25)
        u_K = np.array([0.5, 0.2, 0.3])
        alpha_K = self.model.update_concentration(self.window_size, u_K)
        
        alpha_K_minus_1 = self.model.alpha_history[-2]
        alpha_0 = self.model.alpha_0
        expected_alpha_K = alpha_K_minus_1 + self.scaling_factor * u_K - alpha_0
        
        np.testing.assert_array_almost_equal(alpha_K, expected_alpha_K)
        
    def test_equation_626_sliding_window(self):
        """Test Equation 6.26: Sliding window (t > K)."""
        # Build up to t=K+1
        for t in range(self.window_size + 2):
            u_t = np.array([0.3, 0.4, 0.3]) + 0.1 * t * np.array([1, -1, 0])
            self.model.update_concentration(t, u_t)
        
        # Test t=K+1 (Equation 6.26)
        u_K_plus_1 = np.array([0.6, 0.1, 0.3])
        
        # Store the value that will be removed (the oldest element in the window, u_{t-K-1}) before calling update_concentration
        u_t_minus_K_minus_1 = self.model.u_history[0]  # This is u_{t-K-1}, the oldest element being removed
        
        alpha_K_plus_1 = self.model.update_concentration(self.window_size + 1, u_K_plus_1)
        
        alpha_K = self.model.alpha_history[-2]
        expected_alpha_K_plus_1 = alpha_K + self.scaling_factor * u_K_plus_1 - self.scaling_factor * u_t_minus_K_minus_1
        
        # Apply the same constraint as in the implementation
        expected_alpha_K_plus_1 = np.maximum(expected_alpha_K_plus_1, 1e-10)
        
        np.testing.assert_array_almost_equal(alpha_K_plus_1, expected_alpha_K_plus_1)
        
    def test_velocity_calculation(self):
        """Test velocity calculation (Equation 6.28)."""
        # Add some concentration history
        for t in range(3):
            u_t = np.array([0.3, 0.4, 0.3]) + 0.1 * t * np.array([1, -1, 0])
            self.model.update_concentration(t, u_t)
        
        velocity = self.model.calculate_velocity(2)
        expected_velocity = self.model.alpha_history[-1] - self.model.alpha_history[-2]
        
        np.testing.assert_array_almost_equal(velocity, expected_velocity)
        
    def test_future_prediction(self):
        """Test future concentration prediction."""
        # Add some concentration history
        for t in range(3):
            u_t = np.array([0.3, 0.4, 0.3]) + 0.1 * t * np.array([1, -1, 0])
            self.model.update_concentration(t, u_t)
        
        predicted_alpha = self.model.predict_future_concentration(2, horizon=2)
        current_alpha = self.model.alpha_history[-1]
        velocity = self.model.calculate_velocity(2)
        expected_alpha = current_alpha + 2 * velocity
        
        np.testing.assert_array_almost_equal(predicted_alpha, expected_alpha)
        
    def test_dirichlet_mean(self):
        """Test Dirichlet mean calculation."""
        alpha = np.array([2.0, 3.0, 1.0])
        mean = self.model.dirichlet_mean(alpha)
        expected_mean = alpha / np.sum(alpha)
        
        np.testing.assert_array_almost_equal(mean, expected_mean)
        
    def test_dirichlet_variance(self):
        """Test Dirichlet variance calculation."""
        alpha = np.array([2.0, 3.0, 1.0])
        variance = self.model.dirichlet_variance(alpha)
        
        # Manual calculation
        alpha_sum = np.sum(alpha)
        factor = alpha_sum * alpha_sum * alpha_sum + alpha_sum * alpha_sum
        alpha_square = alpha * alpha
        expected_variance = (alpha_sum * alpha - alpha_square) / factor
        
        np.testing.assert_array_almost_equal(variance, expected_variance)
        
    def test_dirichlet_variance_from_proportions(self):
        """Test Dirichlet variance from proportions."""
        proportions = np.array([0.3, 0.5, 0.2])
        concentration = 10.0
        variance = self.model.dirichlet_variance_from_proportions(proportions, concentration)
        
        alpha = concentration * proportions
        expected_variance = self.model.dirichlet_variance(alpha)
        
        np.testing.assert_array_almost_equal(variance, expected_variance)
        
    def test_dirichlet_mean_map_estimate(self):
        """Test MAP estimate calculation."""
        alpha = np.array([3.0, 4.0, 2.0])
        map_estimate = self.model.dirichlet_mean_map_estimate(alpha)
        
        expected_map = (alpha - 1.0) / (np.sum(alpha) - len(alpha))
        
        np.testing.assert_array_almost_equal(map_estimate, expected_map)
        
    def test_sampling(self):
        """Test sampling from Dirichlet distribution."""
        alpha = np.array([2.0, 3.0, 1.0])
        samples = self.model.sample_from_dirichlet(alpha, size=100)
        
        # Check shape
        self.assertEqual(samples.shape, (100, 3))
        
        # Check that samples sum to 1
        row_sums = np.sum(samples, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(100))
        
        # Check that all values are positive
        self.assertTrue(np.all(samples >= 0))
        
    def test_reset(self):
        """Test model reset functionality."""
        # Add some data
        for t in range(3):
            u_t = np.array([0.3, 0.4, 0.3])
            self.model.update_concentration(t, u_t)
        
        # Reset
        self.model.reset()
        
        # Check that everything is cleared
        self.assertEqual(len(self.model.alpha_history), 0)
        self.assertEqual(len(self.model.u_history), 0)
        self.assertIsNone(self.model.alpha_0)
        self.assertIsNone(self.model.get_current_concentration())
        
    def test_positive_concentration_constraint(self):
        """Test that concentration parameters remain positive."""
        # Test with very small values that could become negative
        u_t = np.array([0.001, 0.001, 0.998])
        alpha_t = self.model.update_concentration(0, u_t)
        
        self.assertTrue(np.all(alpha_t > 0))
        
    def test_normalization_handling(self):
        """Test handling of unnormalized input vectors."""
        # Test with unnormalized vector
        u_unnormalized = np.array([0.3, 0.4, 0.2])  # Sum = 0.9
        alpha_t = self.model.update_concentration(0, u_unnormalized)
        
        # Should still work (normalization is handled internally)
        self.assertTrue(np.all(alpha_t > 0))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with zero vector
        u_zero = np.array([0.0, 0.0, 0.0])
        alpha_t = self.model.update_concentration(0, u_zero)
        
        # Should still produce valid concentration parameters
        self.assertTrue(np.all(alpha_t > 0))
        
        # Test with single element
        model_single = SlidingWindowDirichlet(3, 1.0)
        u_single = np.array([1.0])
        alpha_single = model_single.update_concentration(0, u_single)
        
        self.assertEqual(len(alpha_single), 1)
        self.assertTrue(alpha_single[0] > 0)


if __name__ == '__main__':
    unittest.main()
