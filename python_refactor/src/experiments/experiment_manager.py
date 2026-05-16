"""
Experiment Manager Module

Main orchestrator for portfolio optimization experiments with comprehensive
configuration management, execution control, and result aggregation.
"""

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from .logger import ExperimentLogger
from .metrics_collector import MetricsCollector
from .data_loader import DataLoader
from .portfolio_evaluator import PortfolioEvaluator

class ExperimentManager:
    """Main experiment manager for portfolio optimization experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment manager.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        
        # Initialize components
        self.logger = ExperimentLogger(self.experiment_id)
        self.metrics_collector = MetricsCollector(self.experiment_id)
        self.data_loader = DataLoader()
        self.portfolio_evaluator = PortfolioEvaluator()
        
        # Experiment state
        self.current_experiment = None
        self.results = {}
        self.experiment_history = []
        
        # Log experiment initialization
        self.logger.log('experiment', 'INFO', 'Experiment manager initialized', {
            'experiment_id': self.experiment_id,
            'config': config
        })
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def run_experiment(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with the given configuration.
        
        Args:
            experiment_config: Configuration for this specific experiment
            
        Returns:
            Dictionary containing experiment results
        """
        experiment_name = experiment_config.get('name', 'unnamed_experiment')
        
        self.logger.log('experiment', 'INFO', f'Starting experiment: {experiment_name}', {
            'experiment_config': experiment_config
        })
        
        try:
            # Load data
            data = self._load_experiment_data(experiment_config)
            
            # Run algorithm
            algorithm_results = self._run_algorithm(experiment_config, data)
            
            # Evaluate portfolio
            portfolio_results = self._evaluate_portfolio(experiment_config, algorithm_results, data)
            
            # Collect final metrics
            final_metrics = self._collect_final_metrics(experiment_config, algorithm_results, portfolio_results)
            
            # Save results
            results = {
                'experiment_name': experiment_name,
                'experiment_config': experiment_config,
                'algorithm_results': algorithm_results,
                'portfolio_results': portfolio_results,
                'final_metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self._save_experiment_results(results)
            
            self.logger.log('experiment', 'INFO', f'Experiment {experiment_name} completed successfully', {
                'final_metrics': final_metrics
            })
            
            return results
            
        except Exception as e:
            self.logger.log_error(e, {'experiment_config': experiment_config})
            raise
    
    def run_experiment_suite(self, suite_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a suite of experiments with different configurations.
        
        Args:
            suite_config: Configuration for the experiment suite
            
        Returns:
            Dictionary containing all experiment results
        """
        suite_name = suite_config.get('name', 'unnamed_suite')
        
        self.logger.log('experiment', 'INFO', f'Starting experiment suite: {suite_name}', {
            'suite_config': suite_config
        })
        
        experiments = suite_config.get('experiments', [])
        suite_results = {
            'suite_name': suite_name,
            'experiments': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for i, experiment_config in enumerate(experiments):
            self.logger.log('experiment', 'INFO', f'Running experiment {i+1}/{len(experiments)}', {
                'experiment_name': experiment_config.get('name', f'experiment_{i+1}')
            })
            
            try:
                experiment_result = self.run_experiment(experiment_config)
                suite_results['experiments'].append(experiment_result)
                
            except Exception as e:
                self.logger.log_error(e, {'experiment_index': i, 'experiment_config': experiment_config})
                # Continue with next experiment
                continue
        
        # Generate suite summary
        suite_results['summary'] = self._generate_suite_summary(suite_results['experiments'])
        
        # Save suite results
        self._save_suite_results(suite_results)
        
        self.logger.log('experiment', 'INFO', f'Experiment suite {suite_name} completed', {
            'total_experiments': len(experiments),
            'successful_experiments': len(suite_results['experiments']),
            'summary': suite_results['summary']
        })
        
        return suite_results
    
    def _load_experiment_data(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data for the experiment."""
        data_config = experiment_config.get('data', {})
        
        self.logger.log('data', 'INFO', 'Loading experiment data', {
            'data_config': data_config
        })
        
        # Load asset data
        asset_data = self.data_loader.load_asset_data(
            data_config.get('asset_files', []),
            data_config.get('date_range', {}),
            data_config.get('assets', [])
        )
        
        # Load market data
        market_data = self.data_loader.load_market_data(
            data_config.get('market_files', []),
            data_config.get('date_range', {})
        )
        
        data = {
            'assets': asset_data,
            'market': market_data,
            'config': data_config
        }
        
        self.logger.log('data', 'INFO', 'Data loading completed', {
            'num_assets': len(asset_data) if asset_data else 0,
            'data_period': data_config.get('date_range', {})
        })
        
        return data
    
    def _run_algorithm(self, experiment_config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the optimization algorithm."""
        algorithm_config = experiment_config.get('algorithm', {})
        algorithm_name = algorithm_config.get('name', 'unknown')
        
        self.logger.log('algorithm', 'INFO', f'Starting algorithm: {algorithm_name}', {
            'algorithm_config': algorithm_config
        })
        
        start_time = time.time()
        
        try:
            # Import algorithm dynamically
            if algorithm_name == 'nsga2':
                from ..algorithms.nsga2 import NSGA2
                algorithm = NSGA2(**algorithm_config.get('parameters', {}))
            elif algorithm_name == 'sms_emoa':
                from ..algorithms.sms_emoa import SMSEMOA
                algorithm = SMSEMOA(**algorithm_config.get('parameters', {}))
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Setup anticipatory learning if enabled. Three opt-in tiers
            # ordered from strongest superset to base:
            #   1. learning.use_multi_horizon (W1-3): MultiHorizonAnticipatoryLearning
            #      — multi-horizon convex combo per paper Eq (14); also
            #      has TIP built-in (composes with W1-2's obj_space
            #      threading). Constructor: max_horizon + monte_carlo_samples.
            #   2. learning.use_tip (W1-2): TIPIntegratedAnticipatoryLearning
            #      — TIP arm via paper Eq (13) + thesis Eq 7.16 blend.
            #      Constructor: window_size + monte_carlo_samples.
            #   3. (default): base AnticipatoryLearning — KF-residuals
            #      only (the pre-W1-2 default for every live run).
            # Higher-tier classes inherit from AnticipatoryLearning so
            # set_learning(learning) accepts any of them.
            learning_config = experiment_config.get('learning', {})
            if learning_config.get('enabled', False):
                if learning_config.get('use_multi_horizon', False):
                    from ..algorithms.multi_horizon_anticipatory import (
                        MultiHorizonAnticipatoryLearning,
                    )
                    learning = MultiHorizonAnticipatoryLearning(
                        **learning_config.get('parameters', {})
                    )
                elif learning_config.get('use_tip', False):
                    from ..algorithms.anticipatory_learning import (
                        TIPIntegratedAnticipatoryLearning,
                    )
                    learning = TIPIntegratedAnticipatoryLearning(
                        **learning_config.get('parameters', {})
                    )
                else:
                    from ..algorithms.anticipatory_learning import AnticipatoryLearning
                    learning = AnticipatoryLearning(**learning_config.get('parameters', {}))
                algorithm.set_learning(learning)
            
            # Run algorithm
            population = algorithm.run(data)
            
            execution_time = time.time() - start_time
            
            # Collect algorithm metrics
            algorithm_metrics = self.metrics_collector.collect_optimization_metrics(
                population=population,
                generation=algorithm_config.get('generations', 0),
                pareto_front=algorithm.get_pareto_front(),
                hypervolume=algorithm.get_hypervolume()
            )
            
            # Collect computational metrics
            computational_metrics = self.metrics_collector.collect_computational_metrics(
                execution_time=execution_time,
                memory_usage=self._get_memory_usage(),
                function_evaluations=algorithm.get_function_evaluations()
            )
            
            results = {
                'algorithm_name': algorithm_name,
                'population': population,
                'pareto_front': algorithm.get_pareto_front(),
                'hypervolume': algorithm.get_hypervolume(),
                'algorithm_metrics': algorithm_metrics,
                'computational_metrics': computational_metrics,
                'execution_time': execution_time
            }
            
            self.logger.log('algorithm', 'INFO', f'Algorithm {algorithm_name} completed', {
                'execution_time': execution_time,
                'population_size': len(population),
                'pareto_front_size': len(algorithm.get_pareto_front()),
                'hypervolume': algorithm.get_hypervolume()
            })
            
            return results
            
        except Exception as e:
            self.logger.log_error(e, {'algorithm_config': algorithm_config})
            raise
    
    def _evaluate_portfolio(self, experiment_config: Dict[str, Any], 
                          algorithm_results: Dict[str, Any], 
                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate portfolio performance."""
        self.logger.log('portfolio', 'INFO', 'Starting portfolio evaluation')
        
        try:
            # Get best solution from Pareto front
            pareto_front = algorithm_results['pareto_front']
            if not pareto_front:
                raise ValueError("No solutions in Pareto front")
            
            # Select solution based on criteria
            selection_criteria = experiment_config.get('portfolio_selection', 'hypervolume')
            selected_solution = self._select_solution(pareto_front, selection_criteria)
            
            # Evaluate portfolio performance
            portfolio_results = self.portfolio_evaluator.evaluate_portfolio(
                solution=selected_solution,
                data=data,
                evaluation_period=experiment_config.get('evaluation_period', 'full')
            )
            
            # Collect portfolio metrics
            portfolio_metrics = self.metrics_collector.collect_portfolio_metrics(
                portfolio_weights=portfolio_results['weights'],
                asset_returns=data['assets'],
                portfolio_value=portfolio_results['final_value']
            )
            
            results = {
                'selected_solution': selected_solution,
                'portfolio_weights': portfolio_results['weights'],
                'portfolio_performance': portfolio_results['performance'],
                'portfolio_metrics': portfolio_metrics,
                'selection_criteria': selection_criteria
            }
            
            self.logger.log('portfolio', 'INFO', 'Portfolio evaluation completed', {
                'final_value': portfolio_results['final_value'],
                'total_return': portfolio_results['performance']['total_return'],
                'sharpe_ratio': portfolio_metrics['sharpe_ratio']
            })
            
            return results
            
        except Exception as e:
            self.logger.log_error(e, {'algorithm_results': algorithm_results})
            raise
    
    def _select_solution(self, pareto_front: List, criteria: str):
        """Select solution from Pareto front based on criteria."""
        if not pareto_front:
            return None
        
        if criteria == 'hypervolume':
            # Select solution with highest hypervolume contribution
            return max(pareto_front, key=lambda x: x.hypervolume_contribution if hasattr(x, 'hypervolume_contribution') else 0)
        
        elif criteria == 'return':
            # Select solution with highest return
            return max(pareto_front, key=lambda x: x.P.ROI if hasattr(x, 'P') and hasattr(x.P, 'ROI') else 0)
        
        elif criteria == 'risk':
            # Select solution with lowest risk
            return min(pareto_front, key=lambda x: x.P.risk if hasattr(x, 'P') and hasattr(x.P, 'risk') else float('inf'))
        
        else:
            # Default: return first solution
            return pareto_front[0]
    
    def _collect_final_metrics(self, experiment_config: Dict[str, Any],
                             algorithm_results: Dict[str, Any],
                             portfolio_results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect final metrics for the experiment."""
        # Combine all metrics
        final_metrics = {
            'algorithm': algorithm_results['algorithm_metrics'],
            'computational': algorithm_results['computational_metrics'],
            'portfolio': portfolio_results['portfolio_metrics'],
            'learning': self.metrics_collector.learning_metrics
        }
        
        # Calculate summary statistics
        summary = {
            'total_execution_time': algorithm_results['execution_time'],
            'final_hypervolume': algorithm_results['hypervolume'],
            'pareto_front_size': len(algorithm_results['pareto_front']),
            'final_portfolio_value': portfolio_results['portfolio_performance']['final_value'],
            'total_return': portfolio_results['portfolio_metrics']['cumulative_return'],
            'sharpe_ratio': portfolio_results['portfolio_metrics']['sharpe_ratio'],
            'max_drawdown': portfolio_results['portfolio_metrics']['max_drawdown']
        }
        
        final_metrics['summary'] = summary
        
        return final_metrics
    
    def _save_experiment_results(self, results: Dict[str, Any]):
        """Save experiment results to files."""
        results_dir = Path(f"experiments/results/{self.experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = results_dir / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save metrics
        self.metrics_collector.save_metrics()
        
        # Save configuration
        config_file = results_dir / "experiment_config.json"
        with open(config_file, 'w') as f:
            json.dump(results['experiment_config'], f, indent=2)
    
    def _save_suite_results(self, suite_results: Dict[str, Any]):
        """Save suite results to files."""
        suite_dir = Path(f"experiments/suites/{self.experiment_id}")
        suite_dir.mkdir(parents=True, exist_ok=True)
        
        # Save suite results
        suite_file = suite_dir / "suite_results.json"
        with open(suite_file, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        # Save individual experiment results
        for i, experiment_result in enumerate(suite_results['experiments']):
            exp_file = suite_dir / f"experiment_{i+1}_results.json"
            with open(exp_file, 'w') as f:
                json.dump(experiment_result, f, indent=2, default=str)
    
    def _generate_suite_summary(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics for experiment suite."""
        if not experiments:
            return {}
        
        # Collect key metrics
        execution_times = [exp['final_metrics']['summary']['total_execution_time'] for exp in experiments]
        hypervolumes = [exp['final_metrics']['summary']['final_hypervolume'] for exp in experiments]
        returns = [exp['final_metrics']['summary']['total_return'] for exp in experiments]
        sharpe_ratios = [exp['final_metrics']['summary']['sharpe_ratio'] for exp in experiments]
        
        summary = {
            'total_experiments': len(experiments),
            'execution_time': {
                'mean': np.mean(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'hypervolume': {
                'mean': np.mean(hypervolumes),
                'std': np.std(hypervolumes),
                'min': np.min(hypervolumes),
                'max': np.max(hypervolumes)
            },
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios)
            }
        }
        
        return summary
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def cleanup(self):
        """Clean up experiment manager resources."""
        self.logger.cleanup()
        self.logger.log_experiment_end({
            'total_experiments': len(self.experiment_history),
            'experiment_id': self.experiment_id
        }) 