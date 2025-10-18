"""
Integration tests for Thesis-Aligned Experiment

Tests the integration of thesis parameters with experiment execution.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.experiment_config import get_experiment_config
from experiments.thesis_aligned_experiment import ThesisAlignedExperiment


class TestThesisAlignedExperiment(unittest.TestCase):
    """Test cases for ThesisAlignedExperiment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic returns data
        np.random.seed(42)
        n_assets = 5
        n_days = 200
        
        # Generate realistic returns data
        returns_data = pd.DataFrame(
            np.random.normal(0.001, 0.02, (n_days, n_assets)),
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )
        
        self.returns_data = returns_data
        self.experiment = ThesisAlignedExperiment('small_scale')  # Use small scale for testing
        
    def test_initialization(self):
        """Test experiment initialization."""
        self.assertIsNotNone(self.experiment.config)
        self.assertEqual(self.experiment.experiment_metadata['config_name'], 'small_scale')
        self.assertIn('start_time', self.experiment.experiment_metadata)
        self.assertIn('parameters', self.experiment.experiment_metadata)
        
    def test_config_retrieval(self):
        """Test configuration retrieval methods."""
        # Test time series config
        time_config = self.experiment.config.get_time_series_config()
        self.assertIn('historical_days', time_config)
        self.assertIn('stride_days', time_config)
        self.assertIn('initial_wealth', time_config)
        
        # Test algorithm config
        algo_config = self.experiment.config.get_asmsoa_config()
        self.assertIn('population_size', algo_config)
        self.assertIn('generations', algo_config)
        self.assertIn('mutation_rate', algo_config)
        
        # Test portfolio config
        portfolio_config = self.experiment.config.get_portfolio_config()
        self.assertIn('min_cardinality', portfolio_config)
        self.assertIn('max_cardinality', portfolio_config)
        self.assertIn('feasibility_epsilon', portfolio_config)
        
        # Test anticipatory config
        anticipatory_config = self.experiment.config.get_anticipatory_config()
        self.assertIn('prediction_horizon', anticipatory_config)
        self.assertIn('learning_rate_combination', anticipatory_config)
        
    def test_decision_maker_config(self):
        """Test decision maker configuration."""
        # Test all decision maker types
        for dm_type in self.experiment.config.parameters.DECISION_MAKER_TYPES:
            dm_config = self.experiment.config.get_decision_maker_config(dm_type)
            
            self.assertIn('static_anticipation_rate', dm_config)
            self.assertIn('prediction_horizon', dm_config)
            self.assertIn('learning_rate_combination', dm_config)
            self.assertIn('selection_criterion', dm_config)
            self.assertIn('optimization_target', dm_config)
            
            # Check specific criteria
            if dm_type == 'Hv-DM':
                self.assertEqual(dm_config['selection_criterion'], 'hypervolume')
                self.assertEqual(dm_config['optimization_target'], 'expected_hypervolume')
            elif dm_type == 'R-DM':
                self.assertEqual(dm_config['selection_criterion'], 'random')
                self.assertEqual(dm_config['optimization_target'], 'pareto_frontier')
            elif dm_type == 'M-DM':
                self.assertEqual(dm_config['selection_criterion'], 'median')
                self.assertEqual(dm_config['optimization_target'], 'weight_median')
    
    def test_decision_maker_solution_selection(self):
        """Test decision maker solution selection."""
        # Create mock population
        class MockSolution:
            def __init__(self, roi, risk, delta_s, pareto_rank):
                self.P = type('Portfolio', (), {'ROI': roi, 'risk': risk, 'investment': np.random.dirichlet(np.ones(3))})()
                self.Delta_S = delta_s
                self.Pareto_rank = pareto_rank
                self.alpha = 0.5
                self.prediction_error = 0.01
        
        population = [
            MockSolution(0.1, 0.05, 0.8, 0),  # Pareto front
            MockSolution(0.12, 0.06, 0.9, 0),  # Pareto front
            MockSolution(0.08, 0.04, 0.7, 1),  # Second front
            MockSolution(0.15, 0.08, 0.95, 0),  # Pareto front
        ]
        
        # Test Hv-DM selection
        hv_config = self.experiment.config.get_decision_maker_config('Hv-DM')
        hv_solution = self.experiment._select_decision_maker_solution(population, 'Hv-DM', hv_config)
        self.assertEqual(hv_solution.Delta_S, 0.95)  # Should select highest Delta_S
        
        # Test R-DM selection (random, so just check it returns a solution)
        r_config = self.experiment.config.get_decision_maker_config('R-DM')
        r_solution = self.experiment._select_decision_maker_solution(population, 'R-DM', r_config)
        self.assertIsNotNone(r_solution)
        
        # Test M-DM selection
        m_config = self.experiment.config.get_decision_maker_config('M-DM')
        m_solution = self.experiment._select_decision_maker_solution(population, 'M-DM', m_config)
        self.assertIsNotNone(m_solution)
        
    def test_period_performance_calculation(self):
        """Test period performance calculation."""
        # Create mock solution
        class MockSolution:
            def __init__(self):
                self.P = type('Portfolio', (), {
                    'ROI': 0.1,
                    'risk': 0.05,
                    'investment': np.array([0.4, 0.3, 0.2, 0.1])
                })()
                self.alpha = 0.6
                self.prediction_error = 0.02
        
        solution = MockSolution()
        
        # Create mock period data
        period_data = pd.DataFrame({
            'Asset_0': [0.01, 0.02, -0.01],
            'Asset_1': [0.015, 0.01, 0.005],
            'Asset_2': [0.008, 0.012, 0.01],
            'Asset_3': [0.02, 0.01, 0.015]
        })
        
        initial_wealth = 100000.0
        transaction_cost_rate = 0.001
        
        performance = self.experiment._calculate_period_performance(
            solution, period_data, initial_wealth, transaction_cost_rate
        )
        
        # Check required fields
        required_fields = ['period', 'initial_wealth', 'final_wealth', 'portfolio_return',
                          'transaction_cost', 'roi', 'risk', 'portfolio_weights',
                          'solution_alpha', 'solution_prediction_error']
        
        for field in required_fields:
            self.assertIn(field, performance)
        
        # Check values
        self.assertEqual(performance['initial_wealth'], initial_wealth)
        self.assertEqual(performance['solution_alpha'], 0.6)
        self.assertEqual(performance['solution_prediction_error'], 0.02)
        self.assertGreater(performance['final_wealth'], 0)
        
    def test_summary_statistics_calculation(self):
        """Test summary statistics calculation."""
        # Create mock period results
        period_results = [
            {
                'roi': 0.05,
                'risk': 0.03,
                'solution_alpha': 0.6,
                'solution_prediction_error': 0.02,
                'final_wealth': 105000,
                'transaction_cost': 100
            },
            {
                'roi': 0.03,
                'risk': 0.04,
                'solution_alpha': 0.7,
                'solution_prediction_error': 0.015,
                'final_wealth': 108150,
                'transaction_cost': 150
            },
            {
                'roi': 0.04,
                'risk': 0.035,
                'solution_alpha': 0.65,
                'solution_prediction_error': 0.018,
                'final_wealth': 112476,
                'transaction_cost': 120
            }
        ]
        
        # Set initial wealth for first period
        period_results[0]['initial_wealth'] = 100000
        
        summary = self.experiment._calculate_summary_statistics(period_results)
        
        # Check required fields
        required_fields = ['total_return', 'mean_roi', 'std_roi', 'mean_risk', 'std_risk',
                          'mean_alpha', 'std_alpha', 'mean_prediction_error', 'std_prediction_error',
                          'num_periods', 'final_wealth', 'total_transaction_costs']
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check values
        self.assertEqual(summary['num_periods'], 3)
        self.assertEqual(summary['final_wealth'], 112476)
        self.assertEqual(summary['total_transaction_costs'], 370)
        self.assertAlmostEqual(summary['mean_roi'], 0.04, places=2)
        self.assertAlmostEqual(summary['mean_alpha'], 0.65, places=2)
        
    def test_experiment_summary(self):
        """Test experiment summary generation."""
        # Create mock experiment results
        self.experiment.experiment_results = {
            'run_0': {
                'Hv-DM': {
                    'summary_statistics': {
                        'total_return': 0.12,
                        'final_wealth': 112000,
                        'mean_alpha': 0.6,
                        'mean_prediction_error': 0.02
                    }
                },
                'R-DM': {
                    'summary_statistics': {
                        'total_return': 0.08,
                        'final_wealth': 108000,
                        'mean_alpha': 0.5,
                        'mean_prediction_error': 0.025
                    }
                },
                'M-DM': {
                    'summary_statistics': {
                        'total_return': 0.10,
                        'final_wealth': 110000,
                        'mean_alpha': 0.55,
                        'mean_prediction_error': 0.022
                    }
                }
            }
        }
        
        summary = self.experiment.get_experiment_summary()
        
        # Check structure
        self.assertIn('experiment_metadata', summary)
        self.assertIn('decision_maker_summary', summary)
        self.assertIn('config_used', summary)
        
        # Check decision maker summary
        dm_summary = summary['decision_maker_summary']
        self.assertIn('Hv-DM', dm_summary)
        self.assertIn('R-DM', dm_summary)
        self.assertIn('M-DM', dm_summary)
        
        # Check Hv-DM results
        hv_results = dm_summary['Hv-DM']
        self.assertEqual(hv_results['mean_total_return'], 0.12)
        self.assertEqual(hv_results['mean_final_wealth'], 112000)
        self.assertEqual(hv_results['mean_alpha'], 0.6)
        
    def test_empty_experiment_summary(self):
        """Test experiment summary with no results."""
        # Clear results
        self.experiment.experiment_results = {}
        
        summary = self.experiment.get_experiment_summary()
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], 'No experiment results available')
        
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration should pass
        self.assertTrue(self.experiment.config.validate_configuration())
        
        # Test with different configurations
        thesis_config = get_experiment_config('thesis')
        self.assertTrue(thesis_config.validate_configuration())
        
        small_scale_config = get_experiment_config('small_scale')
        self.assertTrue(small_scale_config.validate_configuration())
        
        large_scale_config = get_experiment_config('large_scale')
        self.assertTrue(large_scale_config.validate_configuration())


class TestThesisAlignedExperimentIntegration(unittest.TestCase):
    """Integration tests for thesis-aligned experiment."""
    
    def test_configuration_consistency(self):
        """Test that configurations are consistent across different instances."""
        # Create multiple experiment instances
        exp1 = ThesisAlignedExperiment('thesis')
        exp2 = ThesisAlignedExperiment('thesis')
        exp3 = ThesisAlignedExperiment('small_scale')
        
        # Thesis configurations should be identical
        config1 = exp1.config.parameters.to_dict()
        config2 = exp2.config.parameters.to_dict()
        
        # Check key parameters
        key_params = ['population_size', 'generations', 'mutation_rate', 'crossover_rate']
        for param in key_params:
            self.assertEqual(config1[param], config2[param])
        
        # Small scale should be different
        config3 = exp3.config.parameters.to_dict()
        self.assertNotEqual(config1['population_size'], config3['population_size'])
        
    def test_parameter_alignment(self):
        """Test that parameters are aligned with thesis specifications."""
        thesis_exp = ThesisAlignedExperiment('thesis')
        params = thesis_exp.config.parameters
        
        # Check ASMS-EMOA parameters
        self.assertEqual(params.POPULATION_SIZE, 20)
        self.assertEqual(params.GENERATIONS, 30)
        self.assertEqual(params.MUTATION_RATE, 0.3)
        self.assertEqual(params.CROSSOVER_RATE, 0.2)
        self.assertEqual(params.TOURNAMENT_SIZE, 2)
        
        # Check portfolio constraints
        self.assertEqual(params.MIN_CARDINALITY, 5)
        self.assertEqual(params.MAX_CARDINALITY, 15)
        self.assertEqual(params.FEASIBILITY_EPSILON, 0.99)
        
        # Check anticipatory learning parameters
        self.assertEqual(params.PREDICTION_HORIZON, 2)
        self.assertEqual(params.LEARNING_RATE_COMBINATION, 0.5)
        self.assertEqual(params.STATIC_ANTICIPATION_RATE, 1.0)
        
        # Check experimental setup
        self.assertEqual(params.HISTORICAL_DAYS, 120)
        self.assertEqual(params.STRIDE_DAYS, 30)
        self.assertEqual(params.INITIAL_WEALTH, 100000.0)
        self.assertEqual(params.TRANSACTION_COST_RATE, 0.001)


if __name__ == '__main__':
    unittest.main()



