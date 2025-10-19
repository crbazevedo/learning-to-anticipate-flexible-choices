#!/usr/bin/env python3
"""
EPIC 1.5: Parameter Tuning Experiment

This experiment implements Phase 1 parameter tuning for the hybrid online learning system:
1. Tune Adaptive Mirror Descent parameters for better regret bounds
2. Optimize UCB confidence for better regime detection stability
3. Fine-tune learning rates for all components

Expected to improve regret bounds and system stability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from algorithms.hybrid_online_learning_system import HybridOnlineLearningSystem, HybridSystemExperiment
from algorithms.contextual_bandits_regime_detection import ContextualBanditUCB
from algorithms.adaptive_mirror_descent import AdaptiveMirrorDescent
from algorithms.online_newton_step import OnlineNewtonStep

class ParameterTuningExperiment:
    """Experiment for parameter tuning of hybrid online learning system"""
    
    def __init__(self):
        self.results = {}
        self.tuning_history = []
        
    def run_parameter_tuning_experiment(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run comprehensive parameter tuning experiment"""
        
        print("🔧 Starting EPIC 1.5 Parameter Tuning Experiment")
        print("=" * 70)
        
        # Define parameter ranges for tuning
        print("\n📊 Defining Parameter Ranges...")
        parameter_ranges = self._define_parameter_ranges()
        
        # Run tuning experiments
        print("\n🎯 Running Parameter Tuning Experiments...")
        tuning_results = self._run_tuning_experiments(data, parameter_ranges, test_periods)
        
        # Find optimal parameters
        print("\n🔍 Finding Optimal Parameters...")
        optimal_parameters = self._find_optimal_parameters(tuning_results)
        
        # Validate optimal parameters
        print("\n✅ Validating Optimal Parameters...")
        validation_results = self._validate_optimal_parameters(data, optimal_parameters, test_periods)
        
        # Generate tuning report
        print("\n📋 Generating Tuning Report...")
        tuning_report = self._generate_tuning_report(tuning_results, optimal_parameters, validation_results)
        
        # Create visualizations
        print("\n🎨 Creating Parameter Tuning Visualizations...")
        self._create_tuning_visualizations(tuning_results, optimal_parameters, validation_results)
        
        return {
            'parameter_ranges': parameter_ranges,
            'tuning_results': tuning_results,
            'optimal_parameters': optimal_parameters,
            'validation_results': validation_results,
            'tuning_report': tuning_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _define_parameter_ranges(self) -> Dict:
        """Define parameter ranges for tuning"""
        
        parameter_ranges = {
            'adaptive_mirror_descent': {
                'learning_rate': [0.001, 0.005, 0.01, 0.02, 0.05],
                'beta': [0.8, 0.85, 0.9, 0.95, 0.99],
                'epsilon': [1e-6, 1e-5, 1e-4, 1e-3]
            },
            'ucb_confidence': {
                'confidence': [0.90, 0.92, 0.95, 0.97, 0.99],
                'alpha': [1.0, 1.5, 2.0, 2.5, 3.0]
            },
            'learning_rates': {
                'regime_detection': [0.005, 0.01, 0.02, 0.05],
                'parameter_adaptation': [0.005, 0.01, 0.02, 0.05],
                'kalman_optimization': [0.005, 0.01, 0.02, 0.05]
            }
        }
        
        return parameter_ranges
    
    def _run_tuning_experiments(self, data: np.ndarray, parameter_ranges: Dict, test_periods: int) -> Dict:
        """Run parameter tuning experiments"""
        
        tuning_results = {
            'adaptive_mirror_descent': [],
            'ucb_confidence': [],
            'learning_rates': []
        }
        
        # 1. Tune Adaptive Mirror Descent
        print("  🔧 Tuning Adaptive Mirror Descent...")
        for lr in parameter_ranges['adaptive_mirror_descent']['learning_rate']:
            for beta in parameter_ranges['adaptive_mirror_descent']['beta']:
                for epsilon in parameter_ranges['adaptive_mirror_descent']['epsilon']:
                    result = self._test_adaptive_mirror_descent(data, lr, beta, epsilon, test_periods)
                    tuning_results['adaptive_mirror_descent'].append(result)
        
        # 2. Tune UCB Confidence
        print("  🎯 Tuning UCB Confidence...")
        for confidence in parameter_ranges['ucb_confidence']['confidence']:
            for alpha in parameter_ranges['ucb_confidence']['alpha']:
                result = self._test_ucb_confidence(data, confidence, alpha, test_periods)
                tuning_results['ucb_confidence'].append(result)
        
        # 3. Tune Learning Rates
        print("  ⚙️ Tuning Learning Rates...")
        for lr_regime in parameter_ranges['learning_rates']['regime_detection']:
            for lr_param in parameter_ranges['learning_rates']['parameter_adaptation']:
                for lr_kalman in parameter_ranges['learning_rates']['kalman_optimization']:
                    result = self._test_learning_rates(data, lr_regime, lr_param, lr_kalman, test_periods)
                    tuning_results['learning_rates'].append(result)
        
        return tuning_results
    
    def _test_adaptive_mirror_descent(self, data: np.ndarray, lr: float, beta: float, epsilon: float, test_periods: int) -> Dict:
        """Test Adaptive Mirror Descent parameters"""
        
        # Initialize system with specific parameters
        system = HybridOnlineLearningSystem(
            learning_rates={
                'regime_detection': 0.01,
                'parameter_adaptation': lr,
                'kalman_optimization': 0.01
            },
            confidence=0.95,
            window_size=10
        )
        
        # Override Adaptive Mirror Descent parameters
        system.param_system.optimizer.eta = lr
        system.param_system.optimizer.beta = beta
        system.param_system.optimizer.epsilon = epsilon
        
        # Run experiment
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            observation = observations[i] if i < len(observations) else None
            reward = rewards[i] if i < len(rewards) else None
            
            system.process_financial_data(window_data, observation, reward)
        
        # Get performance metrics
        status = system.get_system_status()
        param_perf = status['parameter_adaptation']['optimizer_performance']
        
        return {
            'parameters': {'learning_rate': lr, 'beta': beta, 'epsilon': epsilon},
            'regret_bound': param_perf['regret_bound'],
            'parameter_stability': param_perf['parameter_stability'],
            'learning_rate_adaptation': param_perf['learning_rate_adaptation'],
            'total_updates': param_perf['total_updates'],
            'performance_score': self._calculate_performance_score(param_perf)
        }
    
    def _test_ucb_confidence(self, data: np.ndarray, confidence: float, alpha: float, test_periods: int) -> Dict:
        """Test UCB confidence parameters"""
        
        # Initialize system with specific parameters
        system = HybridOnlineLearningSystem(
            learning_rates={
                'regime_detection': 0.01,
                'parameter_adaptation': 0.01,
                'kalman_optimization': 0.01
            },
            confidence=confidence,
            window_size=10
        )
        
        # Override UCB parameters
        system.regime_system.bandit.confidence = confidence
        system.regime_system.bandit.alpha = alpha
        
        # Run experiment
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            observation = observations[i] if i < len(observations) else None
            reward = rewards[i] if i < len(rewards) else None
            
            system.process_financial_data(window_data, observation, reward)
        
        # Get performance metrics
        status = system.get_system_status()
        regime_perf = status['regime_detection']['bandit_performance']
        
        return {
            'parameters': {'confidence': confidence, 'alpha': alpha},
            'regret_bound': status['regime_detection']['regime_detection']['regret_bound'],
            'exploration_rate': regime_perf['exploration_rate'],
            'regime_distribution': regime_perf['regime_distribution'],
            'total_pulls': regime_perf['total_pulls'],
            'performance_score': self._calculate_regime_performance_score(regime_perf)
        }
    
    def _test_learning_rates(self, data: np.ndarray, lr_regime: float, lr_param: float, lr_kalman: float, test_periods: int) -> Dict:
        """Test learning rates for all components"""
        
        # Initialize system with specific learning rates
        system = HybridOnlineLearningSystem(
            learning_rates={
                'regime_detection': lr_regime,
                'parameter_adaptation': lr_param,
                'kalman_optimization': lr_kalman
            },
            confidence=0.95,
            window_size=10
        )
        
        # Run experiment
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            observation = observations[i] if i < len(observations) else None
            reward = rewards[i] if i < len(rewards) else None
            
            system.process_financial_data(window_data, observation, reward)
        
        # Get performance metrics
        status = system.get_system_status()
        
        return {
            'parameters': {
                'regime_detection': lr_regime,
                'parameter_adaptation': lr_param,
                'kalman_optimization': lr_kalman
            },
            'regret_bounds': status['regret_bounds'],
            'system_performance': status['system_overview']['average_performance'],
            'performance_score': self._calculate_system_performance_score(status)
        }
    
    def _generate_rewards(self, data: np.ndarray) -> np.ndarray:
        """Generate reward signals based on data characteristics"""
        rewards = []
        
        for i in range(len(data)):
            roi, risk = data[i]
            
            # Reward based on ROI and risk
            if roi > 0.02 and risk < 0.15:  # Good performance
                reward = 1.0
            elif roi > 0.01 and risk < 0.20:  # Moderate performance
                reward = 0.5
            elif roi > -0.01 and risk < 0.25:  # Poor performance
                reward = 0.0
            else:  # Very poor performance
                reward = -0.5
            
            rewards.append(reward)
        
        return np.array(rewards)
    
    def _calculate_performance_score(self, param_perf: Dict) -> float:
        """Calculate performance score for parameter adaptation"""
        # Lower regret is better, higher stability is better
        regret_score = 1.0 / (1.0 + param_perf['regret_bound'])
        stability_score = param_perf['parameter_stability']
        
        return (regret_score + stability_score) / 2.0
    
    def _calculate_regime_performance_score(self, regime_perf: Dict) -> float:
        """Calculate performance score for regime detection"""
        # Lower exploration rate is better (more exploitation)
        exploration_score = 1.0 - regime_perf['exploration_rate']
        
        # Balanced regime distribution is better
        distribution = regime_perf['regime_distribution']
        balance_score = 1.0 - np.std(list(distribution.values()))
        
        return (exploration_score + balance_score) / 2.0
    
    def _calculate_system_performance_score(self, status: Dict) -> float:
        """Calculate overall system performance score"""
        # Lower total regret is better
        total_regret = sum(status['regret_bounds'].values())
        regret_score = 1.0 / (1.0 + total_regret)
        
        # Higher system performance is better
        system_score = max(0, status['system_overview']['average_performance'])
        
        return (regret_score + system_score) / 2.0
    
    def _find_optimal_parameters(self, tuning_results: Dict) -> Dict:
        """Find optimal parameters from tuning results"""
        
        optimal_parameters = {}
        
        # Find optimal Adaptive Mirror Descent parameters
        if tuning_results['adaptive_mirror_descent']:
            best_amd = max(tuning_results['adaptive_mirror_descent'], 
                          key=lambda x: x['performance_score'])
            optimal_parameters['adaptive_mirror_descent'] = best_amd['parameters']
            optimal_parameters['adaptive_mirror_descent_performance'] = best_amd['performance_score']
        
        # Find optimal UCB confidence parameters
        if tuning_results['ucb_confidence']:
            best_ucb = max(tuning_results['ucb_confidence'], 
                          key=lambda x: x['performance_score'])
            optimal_parameters['ucb_confidence'] = best_ucb['parameters']
            optimal_parameters['ucb_confidence_performance'] = best_ucb['performance_score']
        
        # Find optimal learning rates
        if tuning_results['learning_rates']:
            best_lr = max(tuning_results['learning_rates'], 
                         key=lambda x: x['performance_score'])
            optimal_parameters['learning_rates'] = best_lr['parameters']
            optimal_parameters['learning_rates_performance'] = best_lr['performance_score']
        
        return optimal_parameters
    
    def _validate_optimal_parameters(self, data: np.ndarray, optimal_parameters: Dict, test_periods: int) -> Dict:
        """Validate optimal parameters with comprehensive testing"""
        
        # Create system with optimal parameters
        system = HybridOnlineLearningSystem(
            learning_rates=optimal_parameters.get('learning_rates', {
                'regime_detection': 0.01,
                'parameter_adaptation': 0.01,
                'kalman_optimization': 0.01
            }),
            confidence=optimal_parameters.get('ucb_confidence', {}).get('confidence', 0.95),
            window_size=10
        )
        
        # Apply optimal parameters
        if 'adaptive_mirror_descent' in optimal_parameters:
            amd_params = optimal_parameters['adaptive_mirror_descent']
            system.param_system.optimizer.eta = amd_params['learning_rate']
            system.param_system.optimizer.beta = amd_params['beta']
            system.param_system.optimizer.epsilon = amd_params['epsilon']
        
        if 'ucb_confidence' in optimal_parameters:
            ucb_params = optimal_parameters['ucb_confidence']
            system.regime_system.bandit.confidence = ucb_params['confidence']
            system.regime_system.bandit.alpha = ucb_params['alpha']
        
        # Run validation experiment
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            observation = observations[i] if i < len(observations) else None
            reward = rewards[i] if i < len(rewards) else None
            
            system.process_financial_data(window_data, observation, reward)
        
        # Get final performance metrics
        status = system.get_system_status()
        
        return {
            'system_status': status,
            'optimal_parameters': optimal_parameters,
            'validation_performance': self._calculate_system_performance_score(status)
        }
    
    def _generate_tuning_report(self, tuning_results: Dict, optimal_parameters: Dict, validation_results: Dict) -> str:
        """Generate comprehensive tuning report"""
        
        report = []
        report.append("EPIC 1.5: Parameter Tuning Experiment Report")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # Adaptive Mirror Descent Results
        report.append("🔧 ADAPTIVE MIRROR DESCENT TUNING")
        report.append("-" * 40)
        
        if tuning_results['adaptive_mirror_descent']:
            amd_results = tuning_results['adaptive_mirror_descent']
            best_amd = max(amd_results, key=lambda x: x['performance_score'])
            
            report.append(f"Best Parameters: {best_amd['parameters']}")
            report.append(f"Performance Score: {best_amd['performance_score']:.4f}")
            report.append(f"Regret Bound: {best_amd['regret_bound']:.4f}")
            report.append(f"Parameter Stability: {best_amd['parameter_stability']:.4f}")
            
            # Show improvement
            baseline_amd = min(amd_results, key=lambda x: x['performance_score'])
            improvement = (best_amd['performance_score'] - baseline_amd['performance_score']) / baseline_amd['performance_score'] * 100
            report.append(f"Improvement: {improvement:.1f}%")
        
        # UCB Confidence Results
        report.append("\n🎯 UCB CONFIDENCE TUNING")
        report.append("-" * 40)
        
        if tuning_results['ucb_confidence']:
            ucb_results = tuning_results['ucb_confidence']
            best_ucb = max(ucb_results, key=lambda x: x['performance_score'])
            
            report.append(f"Best Parameters: {best_ucb['parameters']}")
            report.append(f"Performance Score: {best_ucb['performance_score']:.4f}")
            report.append(f"Exploration Rate: {best_ucb['exploration_rate']:.2%}")
            report.append(f"Regime Distribution: {best_ucb['regime_distribution']}")
            
            # Show improvement
            baseline_ucb = min(ucb_results, key=lambda x: x['performance_score'])
            improvement = (best_ucb['performance_score'] - baseline_ucb['performance_score']) / baseline_ucb['performance_score'] * 100
            report.append(f"Improvement: {improvement:.1f}%")
        
        # Learning Rates Results
        report.append("\n⚙️ LEARNING RATES TUNING")
        report.append("-" * 40)
        
        if tuning_results['learning_rates']:
            lr_results = tuning_results['learning_rates']
            best_lr = max(lr_results, key=lambda x: x['performance_score'])
            
            report.append(f"Best Parameters: {best_lr['parameters']}")
            report.append(f"Performance Score: {best_lr['performance_score']:.4f}")
            report.append(f"Regret Bounds: {best_lr['regret_bounds']}")
            report.append(f"System Performance: {best_lr['system_performance']:.4f}")
            
            # Show improvement
            baseline_lr = min(lr_results, key=lambda x: x['performance_score'])
            improvement = (best_lr['performance_score'] - baseline_lr['performance_score']) / baseline_lr['performance_score'] * 100
            report.append(f"Improvement: {improvement:.1f}%")
        
        # Validation Results
        report.append("\n✅ VALIDATION RESULTS")
        report.append("-" * 40)
        
        validation_perf = validation_results['validation_performance']
        report.append(f"Validation Performance: {validation_perf:.4f}")
        
        status = validation_results['system_status']
        report.append(f"System Performance: {status['system_overview']['average_performance']:.4f}")
        report.append(f"Total Updates: {status['system_overview']['total_updates']}")
        
        # Regret bounds
        regret_bounds = status['regret_bounds']
        report.append(f"Regret Bounds:")
        report.append(f"  Regime Detection: {regret_bounds['regime_detection']:.4f}")
        report.append(f"  Parameter Adaptation: {regret_bounds['parameter_adaptation']:.4f}")
        report.append(f"  Kalman Optimization: {regret_bounds['kalman_optimization']:.4f}")
        report.append(f"  Total Regret: {regret_bounds['total_regret']:.4f}")
        
        # Optimal Parameters Summary
        report.append("\n🏆 OPTIMAL PARAMETERS SUMMARY")
        report.append("-" * 40)
        
        for component, params in optimal_parameters.items():
            if isinstance(params, dict) and 'performance' not in component:
                report.append(f"{component}: {params}")
        
        report.append("\n" + "=" * 60)
        report.append("End of Parameter Tuning Experiment Report")
        
        return "\n".join(report)
    
    def _create_tuning_visualizations(self, tuning_results: Dict, optimal_parameters: Dict, validation_results: Dict):
        """Create parameter tuning visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Adaptive Mirror Descent Parameter Tuning
        plt.subplot(3, 3, 1)
        if tuning_results['adaptive_mirror_descent']:
            amd_results = tuning_results['adaptive_mirror_descent']
            learning_rates = [r['parameters']['learning_rate'] for r in amd_results]
            performance_scores = [r['performance_score'] for r in amd_results]
            
            plt.scatter(learning_rates, performance_scores, alpha=0.7, s=50)
            plt.xlabel('Learning Rate')
            plt.ylabel('Performance Score')
            plt.title('Adaptive Mirror Descent: Learning Rate vs Performance')
            plt.grid(True, alpha=0.3)
        
        # 2. UCB Confidence Parameter Tuning
        plt.subplot(3, 3, 2)
        if tuning_results['ucb_confidence']:
            ucb_results = tuning_results['ucb_confidence']
            confidences = [r['parameters']['confidence'] for r in ucb_results]
            performance_scores = [r['performance_score'] for r in ucb_results]
            
            plt.scatter(confidences, performance_scores, alpha=0.7, s=50, color='green')
            plt.xlabel('Confidence Level')
            plt.ylabel('Performance Score')
            plt.title('UCB Confidence: Confidence vs Performance')
            plt.grid(True, alpha=0.3)
        
        # 3. Learning Rates Parameter Tuning
        plt.subplot(3, 3, 3)
        if tuning_results['learning_rates']:
            lr_results = tuning_results['learning_rates']
            regime_lrs = [r['parameters']['regime_detection'] for r in lr_results]
            performance_scores = [r['performance_score'] for r in lr_results]
            
            plt.scatter(regime_lrs, performance_scores, alpha=0.7, s=50, color='red')
            plt.xlabel('Regime Detection Learning Rate')
            plt.ylabel('Performance Score')
            plt.title('Learning Rates: Regime Detection vs Performance')
            plt.grid(True, alpha=0.3)
        
        # 4. Parameter Stability Analysis
        plt.subplot(3, 3, 4)
        if tuning_results['adaptive_mirror_descent']:
            amd_results = tuning_results['adaptive_mirror_descent']
            learning_rates = [r['parameters']['learning_rate'] for r in amd_results]
            stabilities = [r['parameter_stability'] for r in amd_results]
            
            plt.scatter(learning_rates, stabilities, alpha=0.7, s=50, color='blue')
            plt.xlabel('Learning Rate')
            plt.ylabel('Parameter Stability')
            plt.title('Parameter Stability vs Learning Rate')
            plt.grid(True, alpha=0.3)
        
        # 5. Regret Bounds Analysis
        plt.subplot(3, 3, 5)
        if tuning_results['adaptive_mirror_descent']:
            amd_results = tuning_results['adaptive_mirror_descent']
            learning_rates = [r['parameters']['learning_rate'] for r in amd_results]
            regret_bounds = [r['regret_bound'] for r in amd_results]
            
            plt.scatter(learning_rates, regret_bounds, alpha=0.7, s=50, color='orange')
            plt.xlabel('Learning Rate')
            plt.ylabel('Regret Bound')
            plt.title('Regret Bound vs Learning Rate')
            plt.grid(True, alpha=0.3)
        
        # 6. Exploration Rate Analysis
        plt.subplot(3, 3, 6)
        if tuning_results['ucb_confidence']:
            ucb_results = tuning_results['ucb_confidence']
            confidences = [r['parameters']['confidence'] for r in ucb_results]
            exploration_rates = [r['exploration_rate'] for r in ucb_results]
            
            plt.scatter(confidences, exploration_rates, alpha=0.7, s=50, color='purple')
            plt.xlabel('Confidence Level')
            plt.ylabel('Exploration Rate')
            plt.title('Exploration Rate vs Confidence')
            plt.grid(True, alpha=0.3)
        
        # 7. System Performance Comparison
        plt.subplot(3, 3, 7)
        if tuning_results['learning_rates']:
            lr_results = tuning_results['learning_rates']
            param_lrs = [r['parameters']['parameter_adaptation'] for r in lr_results]
            system_performances = [r['system_performance'] for r in lr_results]
            
            plt.scatter(param_lrs, system_performances, alpha=0.7, s=50, color='brown')
            plt.xlabel('Parameter Adaptation Learning Rate')
            plt.ylabel('System Performance')
            plt.title('System Performance vs Parameter Learning Rate')
            plt.grid(True, alpha=0.3)
        
        # 8. Optimal Parameters Summary
        plt.subplot(3, 3, 8)
        if optimal_parameters:
            components = list(optimal_parameters.keys())
            performances = [optimal_parameters.get(f"{comp}_performance", 0) for comp in components]
            
            bars = plt.bar(components, performances, alpha=0.7, color=['blue', 'green', 'red'])
            plt.xlabel('Component')
            plt.ylabel('Performance Score')
            plt.title('Optimal Parameters Performance')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, performances):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 9. Validation Results
        plt.subplot(3, 3, 9)
        if validation_results:
            status = validation_results['system_status']
            regret_bounds = status['regret_bounds']
            
            components = ['Regime Detection', 'Parameter Adaptation', 'Kalman Optimization']
            regret_values = [
                regret_bounds['regime_detection'],
                regret_bounds['parameter_adaptation'],
                regret_bounds['kalman_optimization']
            ]
            
            bars = plt.bar(components, regret_values, alpha=0.7, color=['blue', 'green', 'red'])
            plt.xlabel('Component')
            plt.ylabel('Regret Bound')
            plt.title('Validation: Final Regret Bounds')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, regret_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_parameter_tuning_experiment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Parameter tuning visualization saved as: {filename}")
        
        plt.show()

def load_test_data() -> np.ndarray:
    """Load test data for parameter tuning experiment"""
    # Generate synthetic financial data with regime changes
    np.random.seed(42)
    n_periods = 200
    
    # Generate regime-specific data
    data = []
    
    # Bull market (periods 0-60)
    for i in range(60):
        roi = 0.02 + np.random.normal(0, 0.01)
        risk = 0.10 + np.random.normal(0, 0.02)
        data.append([roi, risk])
    
    # Bear market (periods 60-120)
    for i in range(60):
        roi = -0.01 + np.random.normal(0, 0.02)
        risk = 0.20 + np.random.normal(0, 0.03)
        data.append([roi, risk])
    
    # Sideways market (periods 120-200)
    for i in range(80):
        roi = 0.005 + np.random.normal(0, 0.015)
        risk = 0.15 + np.random.normal(0, 0.025)
        data.append([roi, risk])
    
    return np.array(data)

def main():
    """Run parameter tuning experiment"""
    
    print("🔧 Starting EPIC 1.5 Parameter Tuning Experiment")
    print("=" * 70)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run parameter tuning experiment
    experiment = ParameterTuningExperiment()
    
    try:
        results = experiment.run_parameter_tuning_experiment(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_parameter_tuning_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Parameter tuning experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("PARAMETER TUNING EXPERIMENT REPORT")
        print("=" * 70)
        print(results['tuning_report'])
        
    except Exception as e:
        print(f"❌ Error running parameter tuning experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
