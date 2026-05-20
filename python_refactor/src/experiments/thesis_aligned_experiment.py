"""
Thesis-Aligned Experiment

This module demonstrates how to use the thesis parameters configuration
in experiments, ensuring full alignment with the theoretical framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import logging
from datetime import datetime

# W2-2: relative imports (was `from algorithms.X import Y` which only
# worked under a `sys.path.insert(..., src)` hack and was one of the
# pre-W1-3 collection errors). thesis_aligned_experiment lives in
# src/experiments/, so the parent-package relative `..algorithms` /
# `..config` resolves correctly when imported as part of `src`.
from ..config.experiment_config import get_experiment_config
from ..algorithms.sms_emoa import SMSEMOA
from ..algorithms.anticipatory_learning import TIPIntegratedAnticipatoryLearning

logger = logging.getLogger(__name__)


class ThesisAlignedExperiment:
    """
    Experiment class that uses thesis-aligned parameters.
    
    This class demonstrates how to run experiments using the exact
    parameters specified in the thesis, ensuring reproducibility
    and theoretical alignment.
    """
    
    def __init__(self, config_name: str = 'thesis'):
        """
        Initialize thesis-aligned experiment.
        
        Args:
            config_name: Configuration name ('thesis', 'small_scale', etc.)
        """
        self.config = get_experiment_config(config_name)
        self.experiment_results = {}
        self.experiment_metadata = {
            'config_name': config_name,
            'start_time': datetime.now().isoformat(),
            'parameters': self.config.parameters.to_dict()
        }
        
        logger.info(f"Initialized thesis-aligned experiment with config: {config_name}")
        
    def run_thesis_experiment(self, returns_data: pd.DataFrame, 
                            num_runs: int = 1) -> Dict[str, Any]:
        """
        Run experiment with thesis-aligned parameters.
        
        Args:
            returns_data: Historical returns data
            num_runs: Number of experimental runs
            
        Returns:
            Experiment results
        """
        logger.info(f"Starting thesis experiment with {num_runs} runs")
        
        # Get experimental setup parameters
        time_series_config = self.config.get_time_series_config()
        algorithm_config = self.config.get_asmsoa_config()
        portfolio_config = self.config.get_portfolio_config()
        anticipatory_config = self.config.get_anticipatory_config()
        
        # Calculate experiment periods
        historical_days = time_series_config['historical_days']
        stride_days = time_series_config['stride_days']
        total_days = len(returns_data)
        n_periods = max(1, (total_days - historical_days) // stride_days)
        
        logger.info(f"Experiment setup: {n_periods} periods, {stride_days}-day rebalancing")
        
        all_results = {}
        
        for run in range(num_runs):
            logger.info(f"Starting run {run + 1}/{num_runs}")
            
            run_results = {}
            
            # Test each decision maker type
            for dm_type in self.config.parameters.DECISION_MAKER_TYPES:
                logger.info(f"Testing decision maker: {dm_type}")
                
                dm_results = self._run_decision_maker_experiment(
                    returns_data, dm_type, time_series_config, 
                    algorithm_config, portfolio_config, anticipatory_config,
                    n_periods
                )
                
                run_results[dm_type] = dm_results
            
            all_results[f'run_{run}'] = run_results
        
        # Store results
        self.experiment_results = all_results
        self.experiment_metadata['end_time'] = datetime.now().isoformat()
        self.experiment_metadata['num_runs'] = num_runs
        self.experiment_metadata['num_periods'] = n_periods
        
        logger.info("Thesis experiment completed successfully")
        
        return {
            'results': all_results,
            'metadata': self.experiment_metadata,
            'config': self.config.get_full_config()
        }
    
    def _run_decision_maker_experiment(self, returns_data: pd.DataFrame, 
                                     dm_type: str, time_series_config: Dict[str, Any],
                                     algorithm_config: Dict[str, Any], 
                                     portfolio_config: Dict[str, Any],
                                     anticipatory_config: Dict[str, Any],
                                     n_periods: int) -> Dict[str, Any]:
        """
        Run experiment for a specific decision maker type.
        
        Args:
            returns_data: Historical returns data
            dm_type: Decision maker type
            time_series_config: Time series configuration
            algorithm_config: Algorithm configuration
            portfolio_config: Portfolio configuration
            anticipatory_config: Anticipatory learning configuration
            n_periods: Number of periods
            
        Returns:
            Decision maker results
        """
        # Initialize components with thesis parameters
        anticipatory_learning = TIPIntegratedAnticipatoryLearning(
            window_size=anticipatory_config['kf_window_size'],
            monte_carlo_samples=anticipatory_config['tip_monte_carlo_samples']
        )
        
        # Initialize SMS-EMOA with thesis parameters
        sms_emoa = SMSEMOA(
            population_size=algorithm_config['population_size'],
            generations=algorithm_config['generations'],
            mutation_rate=algorithm_config['mutation_rate'],
            crossover_rate=algorithm_config['crossover_rate'],
            tournament_size=algorithm_config['tournament_size']
        )
        
        # Get decision maker configuration
        dm_config = self.config.get_decision_maker_config(dm_type)
        
        # Run experiment periods
        period_results = []
        accumulated_wealth = time_series_config['initial_wealth']
        
        for period in range(n_periods):
            logger.debug(f"Running period {period + 1}/{n_periods} for {dm_type}")
            
            # Get data for current period
            start_idx = period * time_series_config['stride_days']
            end_idx = start_idx + time_series_config['historical_days']
            
            if end_idx >= len(returns_data):
                break
            
            period_data = returns_data.iloc[start_idx:end_idx]
            
            # Run optimization
            population = sms_emoa.optimize(period_data)
            
            # Apply anticipatory learning
            anticipatory_learning.store_population_snapshot(population, period)
            
            # Select decision based on DM type
            selected_solution = self._select_decision_maker_solution(
                population, dm_type, dm_config
            )
            
            # Calculate period performance
            period_performance = self._calculate_period_performance(
                selected_solution, period_data, accumulated_wealth,
                time_series_config['transaction_cost_rate']
            )
            
            period_results.append(period_performance)
            accumulated_wealth = period_performance['final_wealth']
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics(period_results)
        
        return {
            'period_results': period_results,
            'summary_statistics': summary_stats,
            'decision_maker_config': dm_config,
            'final_wealth': accumulated_wealth
        }
    
    def _select_decision_maker_solution(self, population: List, dm_type: str, 
                                      dm_config: Dict[str, Any]) -> Any:
        """
        Select solution based on decision maker type.
        
        Args:
            population: Population of solutions
            dm_type: Decision maker type
            dm_config: Decision maker configuration
            
        Returns:
            Selected solution
        """
        if dm_type == 'Hv-DM':
            # Select solution with maximum expected hypervolume
            return max(population, key=lambda s: s.Delta_S)

        elif dm_type == 'Hv-DM-AMFC':
            # W22-NC30: operator-correct AMFC (Anticipative Maximal Flexible Choice).
            # argmax_{s in P_t}  E[ HV-contribution(s_{t+h} in forecast F_{t+h}) ]
            # See docs/W22-NC30-CONTRACT.md and src/algorithms/amfc_selector.py.
            #
            # NC30 v1 (Option A): independent KF forecasts per candidate.
            # NC30 b (optional via dm_config['derive_zref']=True): data-derive
            #   z_ref from population extremes (closes Inspection-6 z_ref
            #   ambiguity).
            # NC30 d (optional via dm_config['tie_break_by_variance']=True):
            #   when top-1/top-2 expected contributions are within
            #   ``tie_epsilon`` (default 5%), pick the candidate with the
            #   lowest forecast-variance trace (most certain forecast wins).
            from ..algorithms.amfc_selector import select_amfc
            horizon = int(dm_config.get('horizon', 1))
            n_mc = int(dm_config.get('n_mc', 200))
            # If R1/R2 omitted AND derive_zref=True, the selector derives them.
            R1 = dm_config.get('R1', None)
            R2 = dm_config.get('R2', None)
            # W22-NC30 b STRUCTURAL FIX (operator directive 2026-05-19):
            # default derive_zref=True. Pre-fix used hard-coded (0.0, 0.05),
            # which created the z_ref ambiguity flagged by Inspection 6.
            derive_zref = bool(dm_config.get('derive_zref', True))
            zref_margin = float(dm_config.get('zref_margin', 0.0))
            tie_break = bool(dm_config.get('tie_break_by_variance', False))
            tie_epsilon = float(dm_config.get('tie_epsilon', 0.05))
            collect_telemetry = bool(dm_config.get('collect_telemetry', False))
            # W22-NC30 c: continuous variance-aware contribution discount.
            variance_penalty = float(dm_config.get('variance_penalty', 0.0))
            seed = dm_config.get('seed', None)
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            return select_amfc(
                population,
                horizon=horizon,
                n_mc=n_mc,
                R1=float(R1) if R1 is not None else None,
                R2=float(R2) if R2 is not None else None,
                pareto_only=True,
                rng=rng,
                derive_zref=derive_zref,
                zref_margin=zref_margin,
                tie_break_by_variance=tie_break,
                tie_epsilon=tie_epsilon,
                collect_telemetry=collect_telemetry,
                variance_penalty=variance_penalty,
            )

        elif dm_type == 'R-DM':
            # Random selection from Pareto frontier
            pareto_front = [s for s in population if s.Pareto_rank == 0]
            if pareto_front:
                return np.random.choice(pareto_front)
            else:
                return np.random.choice(population)

        elif dm_type == 'M-DM':
            # Median portfolio by weight vector
            # Sort by first objective (ROI) and select median
            sorted_pop = sorted(population, key=lambda s: s.P.ROI)
            median_idx = len(sorted_pop) // 2
            return sorted_pop[median_idx]

        else:
            raise ValueError(f"Unknown decision maker type: {dm_type}")
    
    def _calculate_period_performance(self, solution: Any, period_data: pd.DataFrame,
                                    initial_wealth: float, transaction_cost_rate: float) -> Dict[str, Any]:
        """
        Calculate performance for a single period.
        
        Args:
            solution: Selected solution
            period_data: Period returns data
            initial_wealth: Initial wealth for the period
            transaction_cost_rate: Transaction cost rate
            
        Returns:
            Period performance metrics
        """
        # Calculate portfolio returns
        portfolio_weights = solution.P.investment
        period_returns = period_data.mean()
        portfolio_return = np.dot(portfolio_weights, period_returns)
        
        # Calculate transaction costs (simplified)
        transaction_cost = initial_wealth * transaction_cost_rate * np.sum(np.abs(portfolio_weights))
        
        # Calculate final wealth
        gross_return = initial_wealth * (1 + portfolio_return)
        final_wealth = gross_return - transaction_cost
        
        # Calculate metrics
        roi = (final_wealth - initial_wealth) / initial_wealth
        risk = solution.P.risk
        
        return {
            'period': len(self.experiment_results) + 1,
            'initial_wealth': initial_wealth,
            'final_wealth': final_wealth,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'roi': roi,
            'risk': risk,
            'portfolio_weights': portfolio_weights.tolist(),
            'solution_alpha': solution.alpha,
            'solution_prediction_error': solution.prediction_error
        }
    
    def _calculate_summary_statistics(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics for the experiment.
        
        Args:
            period_results: List of period results
            
        Returns:
            Summary statistics
        """
        if not period_results:
            return {}
        
        # Extract metrics
        rois = [p['roi'] for p in period_results]
        risks = [p['risk'] for p in period_results]
        alphas = [p['solution_alpha'] for p in period_results]
        prediction_errors = [p['solution_prediction_error'] for p in period_results]
        
        # Calculate statistics
        total_return = (period_results[-1]['final_wealth'] - period_results[0]['initial_wealth']) / period_results[0]['initial_wealth']
        
        return {
            'total_return': total_return,
            'mean_roi': np.mean(rois),
            'std_roi': np.std(rois),
            'mean_risk': np.mean(risks),
            'std_risk': np.std(risks),
            'mean_alpha': np.mean(alphas),
            'std_alpha': np.std(alphas),
            'mean_prediction_error': np.mean(prediction_errors),
            'std_prediction_error': np.std(prediction_errors),
            'num_periods': len(period_results),
            'final_wealth': period_results[-1]['final_wealth'],
            'total_transaction_costs': sum(p['transaction_cost'] for p in period_results)
        }
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary.
        
        Returns:
            Experiment summary
        """
        if not self.experiment_results:
            return {'error': 'No experiment results available'}
        
        # Calculate overall statistics
        all_dm_results = {}
        for run_key, run_results in self.experiment_results.items():
            for dm_type, dm_results in run_results.items():
                if dm_type not in all_dm_results:
                    all_dm_results[dm_type] = []
                all_dm_results[dm_type].append(dm_results['summary_statistics'])
        
        # Calculate averages across runs
        summary = {}
        for dm_type, results_list in all_dm_results.items():
            if results_list:
                summary[dm_type] = {
                    'mean_total_return': np.mean([r['total_return'] for r in results_list]),
                    'std_total_return': np.std([r['total_return'] for r in results_list]),
                    'mean_final_wealth': np.mean([r['final_wealth'] for r in results_list]),
                    'std_final_wealth': np.std([r['final_wealth'] for r in results_list]),
                    'mean_alpha': np.mean([r['mean_alpha'] for r in results_list]),
                    'mean_prediction_error': np.mean([r['mean_prediction_error'] for r in results_list])
                }
        
        return {
            'experiment_metadata': self.experiment_metadata,
            'decision_maker_summary': summary,
            'config_used': self.config.parameters.to_dict()
        }
    
    def print_experiment_summary(self):
        """Print experiment summary."""
        summary = self.get_experiment_summary()
        
        print("=" * 80)
        print("THESIS-ALIGNED EXPERIMENT SUMMARY")
        print("=" * 80)
        
        print(f"\n📋 Experiment Configuration: {summary['experiment_metadata']['config_name']}")
        print(f"⏱️  Duration: {summary['experiment_metadata']['start_time']} to {summary['experiment_metadata']['end_time']}")
        print(f"🔄 Number of Runs: {summary['experiment_metadata']['num_runs']}")
        print(f"📅 Number of Periods: {summary['experiment_metadata']['num_periods']}")
        
        print("\n📊 Decision Maker Performance:")
        for dm_type, stats in summary['decision_maker_summary'].items():
            print(f"\n  {dm_type}:")
            print(f"    Total Return: {stats['mean_total_return']:.4f} ± {stats['std_total_return']:.4f}")
            print(f"    Final Wealth: R$ {stats['mean_final_wealth']:,.2f} ± R$ {stats['std_final_wealth']:,.2f}")
            print(f"    Mean Alpha: {stats['mean_alpha']:.4f}")
            print(f"    Mean Prediction Error: {stats['mean_prediction_error']:.4f}")
        
        print("\n⚙️  Key Parameters Used:")
        config = summary['config_used']
        print(f"    Population Size: {config['population_size']}")
        print(f"    Generations: {config['generations']}")
        print(f"    Mutation Rate: {config['mutation_rate']}")
        print(f"    Crossover Rate: {config['crossover_rate']}")
        print(f"    Historical Days: {config['historical_days']}")
        print(f"    Stride Days: {config['stride_days']}")
        print(f"    Prediction Horizon: {config['prediction_horizon']}")
        
        print("=" * 80)


def run_thesis_aligned_experiment(returns_data: pd.DataFrame, 
                                config_name: str = 'thesis',
                                num_runs: int = 1) -> Dict[str, Any]:
    """
    Convenience function to run thesis-aligned experiment.
    
    Args:
        returns_data: Historical returns data
        config_name: Configuration name
        num_runs: Number of experimental runs
        
    Returns:
        Experiment results
    """
    experiment = ThesisAlignedExperiment(config_name)
    results = experiment.run_thesis_experiment(returns_data, num_runs)
    experiment.print_experiment_summary()
    
    return results


if __name__ == '__main__':
    # Example usage
    print("Thesis-Aligned Experiment Module")
    print("This module provides thesis-aligned experiment functionality.")
    print("Use run_thesis_aligned_experiment() to run experiments with thesis parameters.")
