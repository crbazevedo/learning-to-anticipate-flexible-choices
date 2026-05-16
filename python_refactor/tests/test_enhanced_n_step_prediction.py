"""
Unit tests for Enhanced N-Step Prediction Integration

Tests the enhanced N-step prediction with anticipatory learning integration,
belief coefficient usage, and conditional expected hypervolume calculation.
"""

import unittest
import numpy as np

# W2-2 import-style fix: use canonical `from src.algorithms.X` (W1-2/W1-3/W1-4 pattern).
from src.algorithms.enhanced_n_step_prediction import (
    EnhancedNStepPredictor, EnhancedPredictionResult,
    create_enhanced_n_step_predictor
)


class TestEnhancedNStepPredictor(unittest.TestCase):
    """Test cases for EnhancedNStepPredictor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = EnhancedNStepPredictor(max_horizon=3)
        
        # Create mock solutions for testing
        self.pareto_frontier = [
            self._create_mock_solution(0.1, 0.05),
            self._create_mock_solution(0.12, 0.06),
            self._create_mock_solution(0.08, 0.04)
        ]
        
        # Create mock predictions
        self.kalman_predictions = {
            'step_1': {
                'state': np.array([0.11, 0.055, 0.001, 0.001]),
                'covariance': np.eye(4) * 0.01,
                'horizon': 1
            },
            'step_2': {
                'state': np.array([0.12, 0.06, 0.002, 0.002]),
                'covariance': np.eye(4) * 0.02,
                'horizon': 2
            },
            'step_3': {
                'state': np.array([0.13, 0.065, 0.003, 0.003]),
                'covariance': np.eye(4) * 0.03,
                'horizon': 3
            }
        }
        
        self.dirichlet_predictions = {
            'step_1': {
                'dirichlet_params': np.array([0.4, 0.3, 0.2, 0.1]),
                'mean_prediction': np.array([0.4, 0.3, 0.2, 0.1]),
                'horizon': 1
            },
            'step_2': {
                'dirichlet_params': np.array([0.35, 0.35, 0.2, 0.1]),
                'mean_prediction': np.array([0.35, 0.35, 0.2, 0.1]),
                'horizon': 2
            },
            'step_3': {
                'dirichlet_params': np.array([0.3, 0.4, 0.2, 0.1]),
                'mean_prediction': np.array([0.3, 0.4, 0.2, 0.1]),
                'horizon': 3
            }
        }
        
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
                self.hypervolume_contribution = 0.1  # Add hypervolume contribution
        
        return MockSolution(roi, risk)
    
    def test_initialization(self):
        """Test initialization of enhanced N-step predictor."""
        self.assertEqual(self.predictor.max_horizon, 3)
        self.assertIsNone(self.predictor.anticipatory_learning)
        self.assertIsNotNone(self.predictor.belief_calculator)
        self.assertEqual(len(self.predictor.enhanced_prediction_history), 0)
        
    def test_set_anticipatory_learning(self):
        """Test setting anticipatory learning reference."""
        # Create mock anticipatory learning
        mock_anticipatory = type('MockAnticipatory', (), {})()
        
        self.predictor.set_anticipatory_learning(mock_anticipatory)
        self.assertEqual(self.predictor.anticipatory_learning, mock_anticipatory)
        
    def test_compute_conditional_expected_hypervolume(self):
        """Test enhanced conditional expected hypervolume computation."""
        selected_solution = 0
        horizon = 1
        
        result = self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, selected_solution,
            self.kalman_predictions, self.dirichlet_predictions, horizon
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('solution_0', result)
        self.assertIn('solution_1', result)
        self.assertIn('solution_2', result)
        
        # Check selected solution
        selected_result = result['solution_0']
        self.assertTrue(selected_result['is_selected'])
        self.assertIn('conditional_expected_hypervolume', selected_result)
        self.assertIn('belief_coefficient', selected_result)
        self.assertIn('tip_value', selected_result)
        self.assertIn('confidence', selected_result)
        
        # Check other solutions
        for i in range(1, 3):
            other_result = result[f'solution_{i}']
            self.assertFalse(other_result['is_selected'])
            self.assertIn('conditional_expected_hypervolume', other_result)
            self.assertIn('belief_coefficient', other_result)
        
        # Check that history was stored
        self.assertEqual(len(self.predictor.enhanced_prediction_history), 1)
        
    def test_compute_conditional_expected_hypervolume_invalid_solution(self):
        """Test conditional expected hypervolume with invalid solution index."""
        with self.assertRaises(ValueError):
            self.predictor.compute_conditional_expected_hypervolume(
                self.pareto_frontier, 5,  # Invalid index
                self.kalman_predictions, self.dirichlet_predictions, 1
            )
        
    def test_compute_enhanced_expected_future_hypervolume(self):
        """Test enhanced expected future hypervolume computation."""
        horizon = 1
        
        result = self.predictor.compute_enhanced_expected_future_hypervolume(
            self.pareto_frontier, self.kalman_predictions,
            self.dirichlet_predictions, horizon
        )
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn('solution_0', result)
        self.assertIn('solution_1', result)
        self.assertIn('solution_2', result)
        
        # Check each solution result
        for i in range(3):
            solution_result = result[f'solution_{i}']
            self.assertIn('enhanced_expected_hypervolume', solution_result)
            self.assertIn('base_expected_hypervolume', solution_result)
            self.assertIn('belief_coefficient', solution_result)
            self.assertIn('anticipatory_adjustment', solution_result)
            self.assertIn('confidence', solution_result)
            
            # Check bounds
            self.assertGreaterEqual(solution_result['enhanced_expected_hypervolume'], 0.0)
            self.assertGreaterEqual(solution_result['belief_coefficient'], 0.5)
            self.assertLessEqual(solution_result['belief_coefficient'], 1.0)
            self.assertGreaterEqual(solution_result['anticipatory_adjustment'], 0.8)
            self.assertLessEqual(solution_result['anticipatory_adjustment'], 1.2)
        
    def test_compute_enhanced_expected_future_hypervolume_invalid_horizon(self):
        """Test enhanced expected future hypervolume with invalid horizon."""
        with self.assertRaises(ValueError):
            self.predictor.compute_enhanced_expected_future_hypervolume(
                self.pareto_frontier, self.kalman_predictions,
                self.dirichlet_predictions, 5  # Invalid horizon
            )
        
    def test_get_anticipatory_adjustment(self):
        """Test anticipatory adjustment calculation."""
        solution = self.pareto_frontier[0]
        horizon = 1
        
        # Test without anticipatory learning
        adjustment = self.predictor._get_anticipatory_adjustment(solution, horizon)
        self.assertEqual(adjustment, 1.0)
        
        # Test with mock anticipatory learning
        mock_anticipatory = type('MockAnticipatory', (), {
            'compute_anticipatory_learning_rate': lambda self, sol, h: 0.5
        })()
        
        self.predictor.set_anticipatory_learning(mock_anticipatory)
        adjustment = self.predictor._get_anticipatory_adjustment(solution, horizon)
        self.assertGreaterEqual(adjustment, 0.8)
        self.assertLessEqual(adjustment, 1.2)
        
    def test_calculate_prediction_confidence(self):
        """Test prediction confidence calculation."""
        # Create mock conditional hypervolumes
        conditional_hypervolumes = {
            'solution_0': {'conditional_expected_hypervolume': 0.1},
            'solution_1': {'conditional_expected_hypervolume': 0.12},
            'solution_2': {'conditional_expected_hypervolume': 0.08}
        }
        
        belief_coefficients = {
            'solution_0': 0.8,
            'solution_1': 0.9,
            'solution_2': 0.7
        }
        
        confidence = self.predictor._calculate_prediction_confidence(
            conditional_hypervolumes, belief_coefficients
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with empty data
        confidence = self.predictor._calculate_prediction_confidence({}, {})
        self.assertEqual(confidence, 0.0)
        
    def test_get_enhanced_prediction_statistics(self):
        """Test getting enhanced prediction statistics."""
        # Initially no statistics
        stats = self.predictor.get_enhanced_prediction_statistics()
        self.assertIn('error', stats)
        
        # Create some history
        self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Get statistics
        stats = self.predictor.get_enhanced_prediction_statistics()
        
        self.assertIn('total_predictions', stats)
        self.assertIn('mean_confidence', stats)
        self.assertIn('std_confidence', stats)
        self.assertIn('min_confidence', stats)
        self.assertIn('max_confidence', stats)
        self.assertIn('mean_horizon', stats)
        self.assertIn('std_horizon', stats)
        self.assertIn('horizon_distribution', stats)
        
        self.assertEqual(stats['total_predictions'], 1)
        
    def test_get_enhanced_prediction_for_horizon(self):
        """Test getting enhanced prediction for specific horizon."""
        # Initially no predictions
        result = self.predictor.get_enhanced_prediction_for_horizon(1)
        self.assertIsNone(result)
        
        # Create some history
        self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Get prediction for horizon 1
        result = self.predictor.get_enhanced_prediction_for_horizon(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.horizon, 1)
        
        # Get prediction for horizon 2 (should return None)
        result = self.predictor.get_enhanced_prediction_for_horizon(2)
        self.assertIsNone(result)
        
    def test_reset_enhanced_prediction_history(self):
        """Test resetting enhanced prediction history."""
        # Create some history
        self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Check that history exists
        self.assertGreater(len(self.predictor.enhanced_prediction_history), 0)
        
        # Reset history
        self.predictor.reset_enhanced_prediction_history()
        
        # Check that history is cleared
        self.assertEqual(len(self.predictor.enhanced_prediction_history), 0)
        
    def test_validate_enhanced_prediction(self):
        """Test enhanced prediction validation."""
        # Create a valid result
        valid_result = EnhancedPredictionResult(
            conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
            belief_coefficients={'solution_0': 0.8},
            anticipatory_adjustments={'solution_0': 1.0},
            prediction_confidence=0.9,
            horizon=1,
            timestamp=1234567890.0
        )
        
        self.assertTrue(self.predictor.validate_enhanced_prediction(valid_result))
        
        # Test invalid confidence
        invalid_result = EnhancedPredictionResult(
            conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
            belief_coefficients={'solution_0': 0.8},
            anticipatory_adjustments={'solution_0': 1.0},
            prediction_confidence=1.5,  # Invalid confidence
            horizon=1,
            timestamp=1234567890.0
        )
        
        self.assertFalse(self.predictor.validate_enhanced_prediction(invalid_result))
        
        # Test invalid horizon
        invalid_result = EnhancedPredictionResult(
            conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
            belief_coefficients={'solution_0': 0.8},
            anticipatory_adjustments={'solution_0': 1.0},
            prediction_confidence=0.9,
            horizon=5,  # Invalid horizon
            timestamp=1234567890.0
        )
        
        self.assertFalse(self.predictor.validate_enhanced_prediction(invalid_result))
        
        # Test invalid belief coefficient
        invalid_result = EnhancedPredictionResult(
            conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
            belief_coefficients={'solution_0': 0.3},  # Invalid belief coefficient
            anticipatory_adjustments={'solution_0': 1.0},
            prediction_confidence=0.9,
            horizon=1,
            timestamp=1234567890.0
        )
        
        self.assertFalse(self.predictor.validate_enhanced_prediction(invalid_result))
        
    def test_get_enhanced_prediction_summary(self):
        """Test getting enhanced prediction summary."""
        # Initially no summary
        summary = self.predictor.get_enhanced_prediction_summary()
        self.assertIn('error', summary)
        
        # Create some history
        self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Get summary
        summary = self.predictor.get_enhanced_prediction_summary()
        
        self.assertIn('latest_prediction', summary)
        self.assertIn('total_predictions', summary)
        self.assertIn('horizon_coverage', summary)
        self.assertIn('confidence_trend', summary)
        self.assertIn('belief_coefficient_summary', summary)
        
        self.assertEqual(summary['total_predictions'], 1)
        self.assertEqual(summary['latest_prediction']['horizon'], 1)
        
    def test_calculate_confidence_trend(self):
        """Test confidence trend calculation."""
        # Initially no trend
        trend = self.predictor._calculate_confidence_trend()
        self.assertEqual(trend, 0.0)
        
        # Create some history with varying confidence
        for i in range(5):
            # Mock different confidence values
            result = EnhancedPredictionResult(
                conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
                belief_coefficients={'solution_0': 0.8},
                anticipatory_adjustments={'solution_0': 1.0},
                prediction_confidence=0.5 + 0.1 * i,  # Increasing confidence
                horizon=1,
                timestamp=1234567890.0 + i
            )
            self.predictor.enhanced_prediction_history.append(result)
        
        # Calculate trend
        trend = self.predictor._calculate_confidence_trend()
        self.assertGreater(trend, 0.0)  # Should be positive (increasing)
        
    def test_get_belief_coefficient_summary(self):
        """Test belief coefficient summary."""
        summary = self.predictor._get_belief_coefficient_summary()
        
        # Initially no data
        self.assertIn('error', summary)
        
        # Create some belief coefficient history
        self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Get summary
        summary = self.predictor._get_belief_coefficient_summary()
        
        self.assertIn('mean_belief_coefficient', summary)
        self.assertIn('std_belief_coefficient', summary)
        self.assertIn('total_calculations', summary)
        
        self.assertGreater(summary['total_calculations'], 0)


class TestEnhancedPredictionResult(unittest.TestCase):
    """Test cases for EnhancedPredictionResult dataclass."""
    
    def test_enhanced_prediction_result_creation(self):
        """Test creation of EnhancedPredictionResult object."""
        result = EnhancedPredictionResult(
            conditional_hypervolumes={'solution_0': {'conditional_expected_hypervolume': 0.1}},
            belief_coefficients={'solution_0': 0.8},
            anticipatory_adjustments={'solution_0': 1.0},
            prediction_confidence=0.9,
            horizon=1,
            timestamp=1234567890.0
        )
        
        self.assertEqual(result.prediction_confidence, 0.9)
        self.assertEqual(result.horizon, 1)
        self.assertEqual(result.timestamp, 1234567890.0)
        self.assertIn('solution_0', result.conditional_hypervolumes)
        self.assertIn('solution_0', result.belief_coefficients)
        self.assertIn('solution_0', result.anticipatory_adjustments)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_enhanced_n_step_predictor(self):
        """Test convenience function for creating enhanced N-step predictor."""
        predictor = create_enhanced_n_step_predictor(max_horizon=5)
        
        self.assertIsInstance(predictor, EnhancedNStepPredictor)
        self.assertEqual(predictor.max_horizon, 5)


class TestEnhancedNStepPredictionIntegration(unittest.TestCase):
    """Integration tests for enhanced N-step prediction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = EnhancedNStepPredictor(max_horizon=3)
        self.pareto_frontier = [
            self._create_mock_solution(0.1, 0.05),
            self._create_mock_solution(0.12, 0.06),
            self._create_mock_solution(0.08, 0.04)
        ]
        
        self.kalman_predictions = {
            'step_1': {
                'state': np.array([0.11, 0.055, 0.001, 0.001]),
                'covariance': np.eye(4) * 0.01,
                'horizon': 1
            }
        }
        
        self.dirichlet_predictions = {
            'step_1': {
                'dirichlet_params': np.array([0.4, 0.3, 0.2, 0.1]),
                'mean_prediction': np.array([0.4, 0.3, 0.2, 0.1]),
                'horizon': 1
            }
        }
        
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
                self.hypervolume_contribution = 0.1
        
        return MockSolution(roi, risk)
    
    def test_full_enhanced_prediction_workflow(self):
        """Test complete enhanced prediction workflow."""
        # Step 1: Compute conditional expected hypervolume
        conditional_result = self.predictor.compute_conditional_expected_hypervolume(
            self.pareto_frontier, 0, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        self.assertIsNotNone(conditional_result)
        self.assertIn('solution_0', conditional_result)
        
        # Step 2: Compute enhanced expected future hypervolume
        enhanced_result = self.predictor.compute_enhanced_expected_future_hypervolume(
            self.pareto_frontier, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        self.assertIsNotNone(enhanced_result)
        self.assertIn('solution_0', enhanced_result)
        
        # Step 3: Get statistics
        stats = self.predictor.get_enhanced_prediction_statistics()
        self.assertNotIn('error', stats)
        self.assertEqual(stats['total_predictions'], 1)
        
        # Step 4: Get summary
        summary = self.predictor.get_enhanced_prediction_summary()
        self.assertNotIn('error', summary)
        self.assertEqual(summary['total_predictions'], 1)
        
        # Step 5: Validate results
        latest_result = self.predictor.enhanced_prediction_history[-1]
        self.assertTrue(self.predictor.validate_enhanced_prediction(latest_result))
        
    def test_enhanced_prediction_with_anticipatory_learning(self):
        """Test enhanced prediction with anticipatory learning integration."""
        # Create mock anticipatory learning
        mock_anticipatory = type('MockAnticipatory', (), {
            'compute_anticipatory_learning_rate': lambda self, sol, h: 0.5
        })()
        
        self.predictor.set_anticipatory_learning(mock_anticipatory)
        
        # Compute enhanced prediction
        result = self.predictor.compute_enhanced_expected_future_hypervolume(
            self.pareto_frontier, self.kalman_predictions,
            self.dirichlet_predictions, 1
        )
        
        # Check that anticipatory adjustments were applied
        for i in range(3):
            solution_result = result[f'solution_{i}']
            self.assertIn('anticipatory_adjustment', solution_result)
            self.assertGreaterEqual(solution_result['anticipatory_adjustment'], 0.8)
            self.assertLessEqual(solution_result['anticipatory_adjustment'], 1.2)
        
    def test_enhanced_prediction_bounds_validation(self):
        """Test that enhanced predictions respect bounds."""
        # Test multiple predictions
        for horizon in range(1, 4):
            result = self.predictor.compute_enhanced_expected_future_hypervolume(
                self.pareto_frontier, self.kalman_predictions,
                self.dirichlet_predictions, horizon
            )
            
            for i in range(3):
                solution_result = result[f'solution_{i}']
                
                # Check bounds
                self.assertGreaterEqual(solution_result['enhanced_expected_hypervolume'], 0.0)
                self.assertGreaterEqual(solution_result['belief_coefficient'], 0.5)
                self.assertLessEqual(solution_result['belief_coefficient'], 1.0)
                self.assertGreaterEqual(solution_result['anticipatory_adjustment'], 0.8)
                self.assertLessEqual(solution_result['anticipatory_adjustment'], 1.2)
                self.assertGreaterEqual(solution_result['confidence'], 0.0)
                self.assertLessEqual(solution_result['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
