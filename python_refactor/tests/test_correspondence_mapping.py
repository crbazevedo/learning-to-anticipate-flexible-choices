"""
Unit tests for Correspondence Mapping Implementation

Tests the correspondence mapping functionality for tracking individual
solutions across time periods.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.correspondence_mapping import CorrespondenceMapping


class TestCorrespondenceMapping(unittest.TestCase):
    """Test cases for CorrespondenceMapping class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.correspondence_mapping = CorrespondenceMapping(max_history_size=10)
        
        # Create mock solutions for testing
        self.mock_solutions = self._create_mock_solutions(5)
        
    def _create_mock_solutions(self, num_solutions: int):
        """Create mock solutions for testing."""
        solutions = []
        
        for i in range(num_solutions):
            solution = self._create_mock_solution(
                roi=0.1 + i * 0.02,
                risk=0.05 + i * 0.01,
                weights=np.random.dirichlet(np.ones(3))
            )
            solutions.append(solution)
        
        return solutions
    
    def _create_mock_solution(self, roi: float, risk: float, weights: np.ndarray):
        """Create a single mock solution."""
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
                self.cd = np.random.random()
                self.Delta_S = np.random.random()
                self.Pareto_rank = 0
                self.stability = 1.0
                self.rank_ROI = 0
                self.rank_risk = 0
                self.alpha = np.random.random()
                self.anticipation = False
                self.prediction_error = np.random.random() * 0.1
        
        return MockSolution(roi, risk, weights)
    
    def test_initialization(self):
        """Test correspondence mapping initialization."""
        self.assertEqual(self.correspondence_mapping.max_history_size, 10)
        self.assertEqual(len(self.correspondence_mapping.historical_populations), 0)
        self.assertEqual(len(self.correspondence_mapping.historical_anticipative_decisions), 0)
        self.assertIsNone(self.correspondence_mapping.predicted_anticipative_decision)
        
    def test_store_population(self):
        """Test storing population snapshots."""
        current_time = 0
        metadata = {'test': 'data'}
        
        self.correspondence_mapping.store_population(
            self.mock_solutions, current_time, metadata
        )
        
        self.assertEqual(len(self.correspondence_mapping.historical_populations), 1)
        self.assertEqual(len(self.correspondence_mapping.population_metadata), 1)
        
        stored_population = self.correspondence_mapping.historical_populations[0]
        self.assertEqual(len(stored_population), 5)
        
        stored_metadata = self.correspondence_mapping.population_metadata[0]
        self.assertEqual(stored_metadata['current_time'], current_time)
        self.assertEqual(stored_metadata['population_size'], 5)
        self.assertEqual(stored_metadata['metadata'], metadata)
        
    def test_get_historical_solution(self):
        """Test retrieving historical solutions."""
        # Store a population
        self.correspondence_mapping.store_population(self.mock_solutions, 0)
        
        # Test valid retrieval
        solution = self.correspondence_mapping.get_historical_solution(0, 0)
        self.assertIsNotNone(solution)
        self.assertEqual(solution.P.ROI, self.mock_solutions[0].P.ROI)
        
        # Test invalid indices
        self.assertIsNone(self.correspondence_mapping.get_historical_solution(-1, 0))
        self.assertIsNone(self.correspondence_mapping.get_historical_solution(10, 0))
        self.assertIsNone(self.correspondence_mapping.get_historical_solution(0, -1))
        self.assertIsNone(self.correspondence_mapping.get_historical_solution(0, 10))
        
    def test_track_solution_evolution(self):
        """Test tracking solution evolution across time steps."""
        # Store multiple populations
        for t in range(3):
            solutions = self._create_mock_solutions(3)
            self.correspondence_mapping.store_population(solutions, t)
        
        # Track evolution of solution 0
        evolution = self.correspondence_mapping.track_solution_evolution(0, 0, 2)
        
        self.assertEqual(len(evolution), 3)
        for i, solution in enumerate(evolution):
            self.assertIsNotNone(solution)
            self.assertEqual(solution.P.ROI, self.correspondence_mapping.historical_populations[i][0].P.ROI)
        
    def test_find_corresponding_solution(self):
        """Test finding corresponding solutions across time steps."""
        # Create populations with similar solutions
        population1 = self._create_mock_solutions(3)
        population2 = self._create_mock_solutions(3)
        
        # Make solution 0 in population2 very similar to solution 0 in population1
        population2[0].P.investment = population1[0].P.investment.copy()
        
        # Store populations
        self.correspondence_mapping.store_population(population1, 0)
        self.correspondence_mapping.store_population(population2, 1)
        
        # Find corresponding solution
        corresponding = self.correspondence_mapping.find_corresponding_solution(
            population1[0], 0, 1, similarity_threshold=0.9
        )
        
        self.assertIsNotNone(corresponding)
        self.assertEqual(corresponding.P.ROI, population2[0].P.ROI)
        
    def test_calculate_weight_similarity(self):
        """Test weight similarity calculation."""
        weights1 = np.array([0.5, 0.3, 0.2])
        weights2 = np.array([0.5, 0.3, 0.2])
        weights3 = np.array([0.2, 0.3, 0.5])
        
        # Identical weights should have similarity 1.0
        similarity1 = self.correspondence_mapping._calculate_weight_similarity(weights1, weights2)
        self.assertAlmostEqual(similarity1, 1.0, places=5)
        
        # Different weights should have lower similarity
        similarity2 = self.correspondence_mapping._calculate_weight_similarity(weights1, weights3)
        self.assertLess(similarity2, 1.0)
        self.assertGreaterEqual(similarity2, 0.0)
        
    def test_store_anticipative_decision(self):
        """Test storing anticipative decisions."""
        solution = self.mock_solutions[0]
        current_time = 5
        
        self.correspondence_mapping.store_anticipative_decision(solution, current_time)
        
        self.assertEqual(len(self.correspondence_mapping.historical_anticipative_decisions), 1)
        stored_decision = self.correspondence_mapping.historical_anticipative_decisions[0]
        self.assertEqual(stored_decision.P.ROI, solution.P.ROI)
        
    def test_predicted_anticipative_decision(self):
        """Test predicted anticipative decision functionality."""
        solution = self.mock_solutions[0]
        
        # Set predicted decision
        self.correspondence_mapping.set_predicted_anticipative_decision(solution)
        
        # Get predicted decision
        predicted = self.correspondence_mapping.get_predicted_anticipative_decision()
        self.assertIsNotNone(predicted)
        self.assertEqual(predicted.P.ROI, solution.P.ROI)
        
    def test_get_population_statistics(self):
        """Test population statistics calculation."""
        # Store a population
        self.correspondence_mapping.store_population(self.mock_solutions, 0)
        
        # Get statistics
        stats = self.correspondence_mapping.get_population_statistics(0)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['time_step'], 0)
        self.assertEqual(stats['population_size'], 5)
        self.assertIn('roi_mean', stats)
        self.assertIn('risk_mean', stats)
        self.assertIn('alpha_mean', stats)
        self.assertIn('prediction_error_mean', stats)
        
        # Test invalid time step
        self.assertIsNone(self.correspondence_mapping.get_population_statistics(-1))
        self.assertIsNone(self.correspondence_mapping.get_population_statistics(10))
        
    def test_get_evolution_statistics(self):
        """Test evolution statistics calculation."""
        # Store multiple populations
        for t in range(3):
            solutions = self._create_mock_solutions(3)
            self.correspondence_mapping.store_population(solutions, t)
        
        # Get evolution statistics
        stats = self.correspondence_mapping.get_evolution_statistics(0, 0, 2)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['solution_index'], 0)
        self.assertEqual(stats['start_time'], 0)
        self.assertEqual(stats['end_time'], 2)
        self.assertEqual(stats['evolution_length'], 3)
        self.assertIn('roi_evolution', stats)
        self.assertIn('risk_evolution', stats)
        self.assertIn('alpha_evolution', stats)
        self.assertIn('roi_trend', stats)
        self.assertIn('risk_trend', stats)
        self.assertIn('alpha_trend', stats)
        
    def test_max_history_size(self):
        """Test maximum history size constraint."""
        # Store more populations than max_history_size
        for t in range(15):  # More than max_history_size (10)
            solutions = self._create_mock_solutions(3)
            self.correspondence_mapping.store_population(solutions, t)
        
        # Should only keep the last max_history_size populations
        self.assertEqual(len(self.correspondence_mapping.historical_populations), 10)
        self.assertEqual(len(self.correspondence_mapping.population_metadata), 10)
        
        # The first population should be from time step 5 (15 - 10)
        first_metadata = self.correspondence_mapping.population_metadata[0]
        self.assertEqual(first_metadata['current_time'], 5)
        
    def test_clear_history(self):
        """Test clearing all historical data."""
        # Store some data
        self.correspondence_mapping.store_population(self.mock_solutions, 0)
        self.correspondence_mapping.store_anticipative_decision(self.mock_solutions[0], 0)
        self.correspondence_mapping.set_predicted_anticipative_decision(self.mock_solutions[0])
        
        # Clear history
        self.correspondence_mapping.clear_history()
        
        # Verify everything is cleared
        self.assertEqual(len(self.correspondence_mapping.historical_populations), 0)
        self.assertEqual(len(self.correspondence_mapping.historical_anticipative_decisions), 0)
        self.assertEqual(len(self.correspondence_mapping.population_metadata), 0)
        self.assertIsNone(self.correspondence_mapping.predicted_anticipative_decision)
        
    def test_get_history_summary(self):
        """Test history summary generation."""
        # Initially empty
        summary = self.correspondence_mapping.get_history_summary()
        self.assertEqual(summary['num_historical_populations'], 0)
        self.assertEqual(summary['num_anticipative_decisions'], 0)
        self.assertFalse(summary['has_predicted_decision'])
        
        # Add some data
        self.correspondence_mapping.store_population(self.mock_solutions, 0)
        self.correspondence_mapping.store_anticipative_decision(self.mock_solutions[0], 0)
        self.correspondence_mapping.set_predicted_anticipative_decision(self.mock_solutions[0])
        
        # Check updated summary
        summary = self.correspondence_mapping.get_history_summary()
        self.assertEqual(summary['num_historical_populations'], 1)
        self.assertEqual(summary['num_anticipative_decisions'], 1)
        self.assertTrue(summary['has_predicted_decision'])
        self.assertEqual(summary['time_range']['start'], 0)
        self.assertEqual(summary['time_range']['end'], 0)
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty population
        self.correspondence_mapping.store_population([], 0)
        self.assertEqual(len(self.correspondence_mapping.historical_populations), 1)
        self.assertEqual(len(self.correspondence_mapping.historical_populations[0]), 0)
        
        # Test with single solution
        single_solution = [self.mock_solutions[0]]
        self.correspondence_mapping.store_population(single_solution, 1)
        
        # Test evolution with single time step
        evolution = self.correspondence_mapping.track_solution_evolution(0, 1, 1)
        self.assertEqual(len(evolution), 1)
        
        # Test statistics with single solution
        stats = self.correspondence_mapping.get_population_statistics(1)
        self.assertIsNotNone(stats)
        self.assertEqual(stats['population_size'], 1)


if __name__ == '__main__':
    unittest.main()



