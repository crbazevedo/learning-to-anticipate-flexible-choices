#!/usr/bin/env python3
"""
EPIC 1.5: Optimized System Comparison Experiment

This experiment compares the optimized hybrid online learning system
with the original system to validate the improvements from parameter tuning.

Expected to show significant improvements in:
- Regret bounds
- Parameter stability
- System performance
- Overall efficiency
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

from algorithms.hybrid_online_learning_system import HybridOnlineLearningSystem
from algorithms.optimized_hybrid_system import OptimizedHybridOnlineLearningSystem

class OptimizedSystemComparisonExperiment:
    """Experiment for comparing optimized vs original hybrid systems"""
    
    def __init__(self):
        self.results = {}
        self.comparison_history = []
        
    def run_comparison_experiment(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run comprehensive comparison experiment"""
        
        print("🔍 Starting EPIC 1.5 Optimized System Comparison Experiment")
        print("=" * 70)
        
        # Initialize both systems
        print("\n🔧 Initializing Systems...")
        original_system = HybridOnlineLearningSystem(
            learning_rates={
                'regime_detection': 0.01,
                'parameter_adaptation': 0.01,
                'kalman_optimization': 0.01
            },
            confidence=0.95,
            window_size=10
        )
        
        optimized_system = OptimizedHybridOnlineLearningSystem(window_size=10)
        
        # Run experiments
        print("\n📊 Running Original System Experiment...")
        original_results = self._run_system_experiment(original_system, data, test_periods, "Original")
        
        print("\n📊 Running Optimized System Experiment...")
        optimized_results = self._run_system_experiment(optimized_system, data, test_periods, "Optimized")
        
        # Compare results
        print("\n🔍 Comparing Results...")
        comparison_results = self._compare_systems(original_results, optimized_results)
        
        # Generate comparison report
        print("\n📋 Generating Comparison Report...")
        comparison_report = self._generate_comparison_report(comparison_results)
        
        # Create visualizations
        print("\n🎨 Creating Comparison Visualizations...")
        self._create_comparison_visualizations(original_results, optimized_results, comparison_results)
        
        return {
            'original_results': original_results,
            'optimized_results': optimized_results,
            'comparison_results': comparison_results,
            'comparison_report': comparison_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _run_system_experiment(self, system, data: np.ndarray, test_periods: int, system_name: str) -> Dict:
        """Run experiment with a specific system"""
        
        # Generate observations and rewards
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        # Run experiment
        results = []
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            observation = observations[i] if i < len(observations) else None
            reward = rewards[i] if i < len(rewards) else None
            
            if hasattr(system, 'process_financial_data'):
                result = system.process_financial_data(window_data, observation, reward)
                results.append(result)
        
        # Get system status
        status = system.get_system_status()
        
        return {
            'system_name': system_name,
            'results': results,
            'status': status,
            'total_updates': len(results),
            'performance_history': system.performance_history if hasattr(system, 'performance_history') else []
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
    
    def _compare_systems(self, original_results: Dict, optimized_results: Dict) -> Dict:
        """Compare original and optimized systems"""
        
        comparison = {
            'system_performance': {},
            'regret_bounds': {},
            'parameter_stability': {},
            'optimization_metrics': {},
            'improvements': {}
        }
        
        # System Performance Comparison
        original_perf = original_results['status']['system_overview']
        optimized_perf = optimized_results['status']['system_overview']
        
        comparison['system_performance'] = {
            'original': {
                'total_updates': original_perf['total_updates'],
                'average_performance': original_perf['average_performance'],
                'confidence_level': original_perf['confidence_level']
            },
            'optimized': {
                'total_updates': optimized_perf['total_updates'],
                'average_performance': optimized_perf['average_performance'],
                'confidence_level': optimized_perf['confidence_level']
            }
        }
        
        # Regret Bounds Comparison
        original_regret = original_results['status']['regret_bounds']
        optimized_regret = optimized_results['status']['regret_bounds']
        
        comparison['regret_bounds'] = {
            'original': original_regret,
            'optimized': optimized_regret,
            'improvements': {
                'regime_detection': self._calculate_improvement(original_regret['regime_detection'], optimized_regret['regime_detection']),
                'parameter_adaptation': self._calculate_improvement(original_regret['parameter_adaptation'], optimized_regret['parameter_adaptation']),
                'kalman_optimization': self._calculate_improvement(original_regret['kalman_optimization'], optimized_regret['kalman_optimization']),
                'total_regret': self._calculate_improvement(original_regret['total_regret'], optimized_regret['total_regret'])
            }
        }
        
        # Parameter Stability Comparison
        original_param = original_results['status']['parameter_adaptation']['optimizer_performance']
        optimized_param = optimized_results['status']['parameter_adaptation']['optimizer_performance']
        
        comparison['parameter_stability'] = {
            'original': {
                'parameter_stability': original_param['parameter_stability'],
                'learning_rate_adaptation': original_param['learning_rate_adaptation'],
                'regret_bound': original_param['regret_bound']
            },
            'optimized': {
                'parameter_stability': optimized_param['parameter_stability'],
                'learning_rate_adaptation': optimized_param['learning_rate_adaptation'],
                'regret_bound': optimized_param['regret_bound']
            },
            'improvements': {
                'parameter_stability': self._calculate_improvement(original_param['parameter_stability'], optimized_param['parameter_stability']),
                'learning_rate_adaptation': self._calculate_improvement(original_param['learning_rate_adaptation'], optimized_param['learning_rate_adaptation']),
                'regret_bound': self._calculate_improvement(original_param['regret_bound'], optimized_param['regret_bound'])
            }
        }
        
        # Optimization Metrics (if available)
        if 'optimization_metrics' in optimized_results['status']:
            comparison['optimization_metrics'] = optimized_results['status']['optimization_metrics']
        
        # Overall Improvements
        comparison['improvements'] = {
            'system_performance': self._calculate_improvement(
                original_perf['average_performance'], 
                optimized_perf['average_performance']
            ),
            'total_regret': comparison['regret_bounds']['improvements']['total_regret'],
            'parameter_stability': comparison['parameter_stability']['improvements']['parameter_stability']
        }
        
        return comparison
    
    def _calculate_improvement(self, original_value: float, optimized_value: float) -> float:
        """Calculate improvement percentage"""
        if original_value == 0:
            return 0.0
        
        improvement = (original_value - optimized_value) / original_value * 100
        return improvement
    
    def _generate_comparison_report(self, comparison_results: Dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("EPIC 1.5: Optimized System Comparison Report")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # System Performance Comparison
        report.append("📊 SYSTEM PERFORMANCE COMPARISON")
        report.append("-" * 40)
        
        system_perf = comparison_results['system_performance']
        original = system_perf['original']
        optimized = system_perf['optimized']
        
        report.append(f"Original System:")
        report.append(f"  Total Updates: {original['total_updates']}")
        report.append(f"  Average Performance: {original['average_performance']:.4f}")
        report.append(f"  Confidence Level: {original['confidence_level']:.2%}")
        
        report.append(f"\nOptimized System:")
        report.append(f"  Total Updates: {optimized['total_updates']}")
        report.append(f"  Average Performance: {optimized['average_performance']:.4f}")
        report.append(f"  Confidence Level: {optimized['confidence_level']:.2%}")
        
        # Regret Bounds Comparison
        report.append("\n📈 REGRET BOUNDS COMPARISON")
        report.append("-" * 40)
        
        regret_comp = comparison_results['regret_bounds']
        original_regret = regret_comp['original']
        optimized_regret = regret_comp['optimized']
        improvements = regret_comp['improvements']
        
        report.append(f"Original Regret Bounds:")
        report.append(f"  Regime Detection: {original_regret['regime_detection']:.4f}")
        report.append(f"  Parameter Adaptation: {original_regret['parameter_adaptation']:.4f}")
        report.append(f"  Kalman Optimization: {original_regret['kalman_optimization']:.4f}")
        report.append(f"  Total Regret: {original_regret['total_regret']:.4f}")
        
        report.append(f"\nOptimized Regret Bounds:")
        report.append(f"  Regime Detection: {optimized_regret['regime_detection']:.4f}")
        report.append(f"  Parameter Adaptation: {optimized_regret['parameter_adaptation']:.4f}")
        report.append(f"  Kalman Optimization: {optimized_regret['kalman_optimization']:.4f}")
        report.append(f"  Total Regret: {optimized_regret['total_regret']:.4f}")
        
        report.append(f"\nRegret Improvements:")
        report.append(f"  Regime Detection: {improvements['regime_detection']:.1f}%")
        report.append(f"  Parameter Adaptation: {improvements['parameter_adaptation']:.1f}%")
        report.append(f"  Kalman Optimization: {improvements['kalman_optimization']:.1f}%")
        report.append(f"  Total Regret: {improvements['total_regret']:.1f}%")
        
        # Parameter Stability Comparison
        report.append("\n⚙️ PARAMETER STABILITY COMPARISON")
        report.append("-" * 40)
        
        param_comp = comparison_results['parameter_stability']
        original_param = param_comp['original']
        optimized_param = param_comp['optimized']
        param_improvements = param_comp['improvements']
        
        report.append(f"Original Parameter Stability:")
        report.append(f"  Parameter Stability: {original_param['parameter_stability']:.4f}")
        report.append(f"  Learning Rate Adaptation: {original_param['learning_rate_adaptation']:.4f}")
        report.append(f"  Regret Bound: {original_param['regret_bound']:.4f}")
        
        report.append(f"\nOptimized Parameter Stability:")
        report.append(f"  Parameter Stability: {optimized_param['parameter_stability']:.4f}")
        report.append(f"  Learning Rate Adaptation: {optimized_param['learning_rate_adaptation']:.4f}")
        report.append(f"  Regret Bound: {optimized_param['regret_bound']:.4f}")
        
        report.append(f"\nParameter Stability Improvements:")
        report.append(f"  Parameter Stability: {param_improvements['parameter_stability']:.1f}%")
        report.append(f"  Learning Rate Adaptation: {param_improvements['learning_rate_adaptation']:.1f}%")
        report.append(f"  Regret Bound: {param_improvements['regret_bound']:.1f}%")
        
        # Overall Improvements
        report.append("\n🏆 OVERALL IMPROVEMENTS")
        report.append("-" * 40)
        
        overall_improvements = comparison_results['improvements']
        report.append(f"System Performance: {overall_improvements['system_performance']:.1f}%")
        report.append(f"Total Regret: {overall_improvements['total_regret']:.1f}%")
        report.append(f"Parameter Stability: {overall_improvements['parameter_stability']:.1f}%")
        
        # Success Assessment
        report.append("\n🎯 SUCCESS ASSESSMENT")
        report.append("-" * 40)
        
        success_metrics = []
        
        # Check if improvements are positive
        if overall_improvements['system_performance'] > 0:
            success_metrics.append("✅ System performance improved")
        else:
            success_metrics.append("❌ System performance did not improve")
        
        if overall_improvements['total_regret'] > 0:
            success_metrics.append("✅ Total regret reduced")
        else:
            success_metrics.append("❌ Total regret did not improve")
        
        if overall_improvements['parameter_stability'] > 0:
            success_metrics.append("✅ Parameter stability improved")
        else:
            success_metrics.append("❌ Parameter stability did not improve")
        
        for metric in success_metrics:
            report.append(f"  {metric}")
        
        # Overall Assessment
        positive_improvements = sum(1 for metric in success_metrics if metric.startswith("✅"))
        total_metrics = len(success_metrics)
        success_rate = (positive_improvements / total_metrics) * 100 if total_metrics > 0 else 0
        
        report.append(f"\nSuccess Rate: {success_rate:.1f}% ({positive_improvements}/{total_metrics})")
        
        if success_rate >= 80:
            report.append("Status: 🎉 EXCELLENT - Significant improvements achieved!")
        elif success_rate >= 60:
            report.append("Status: ✅ GOOD - Notable improvements achieved!")
        elif success_rate >= 40:
            report.append("Status: ⚠️  PARTIAL - Some improvements achieved")
        else:
            report.append("Status: ❌ POOR - Few improvements achieved")
        
        report.append("\n" + "=" * 60)
        report.append("End of Optimized System Comparison Report")
        
        return "\n".join(report)
    
    def _create_comparison_visualizations(self, original_results: Dict, optimized_results: Dict, comparison_results: Dict):
        """Create comparison visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. System Performance Comparison
        plt.subplot(3, 3, 1)
        systems = ['Original', 'Optimized']
        performances = [
            original_results['status']['system_overview']['average_performance'],
            optimized_results['status']['system_overview']['average_performance']
        ]
        
        bars = plt.bar(systems, performances, color=['blue', 'green'], alpha=0.7)
        plt.title('System Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Average Performance')
        
        # Add value labels
        for bar, value in zip(bars, performances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Regret Bounds Comparison
        plt.subplot(3, 3, 2)
        components = ['Regime Detection', 'Parameter Adaptation', 'Kalman Optimization']
        original_regret = [
            original_results['status']['regret_bounds']['regime_detection'],
            original_results['status']['regret_bounds']['parameter_adaptation'],
            original_results['status']['regret_bounds']['kalman_optimization']
        ]
        optimized_regret = [
            optimized_results['status']['regret_bounds']['regime_detection'],
            optimized_results['status']['regret_bounds']['parameter_adaptation'],
            optimized_results['status']['regret_bounds']['kalman_optimization']
        ]
        
        x = np.arange(len(components))
        width = 0.35
        
        plt.bar(x - width/2, original_regret, width, label='Original', alpha=0.7, color='blue')
        plt.bar(x + width/2, optimized_regret, width, label='Optimized', alpha=0.7, color='green')
        
        plt.title('Regret Bounds Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Regret Bound')
        plt.xticks(x, components, rotation=45)
        plt.legend()
        
        # 3. Parameter Stability Comparison
        plt.subplot(3, 3, 3)
        original_param = original_results['status']['parameter_adaptation']['optimizer_performance']
        optimized_param = optimized_results['status']['parameter_adaptation']['optimizer_performance']
        
        metrics = ['Parameter Stability', 'Learning Rate Adaptation', 'Regret Bound']
        original_values = [
            original_param['parameter_stability'],
            original_param['learning_rate_adaptation'],
            original_param['regret_bound']
        ]
        optimized_values = [
            optimized_param['parameter_stability'],
            optimized_param['learning_rate_adaptation'],
            optimized_param['regret_bound']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original', alpha=0.7, color='blue')
        plt.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.7, color='green')
        
        plt.title('Parameter Stability Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        
        # 4. Improvement Percentages
        plt.subplot(3, 3, 4)
        improvements = comparison_results['improvements']
        improvement_metrics = ['System Performance', 'Total Regret', 'Parameter Stability']
        improvement_values = [
            improvements['system_performance'],
            improvements['total_regret'],
            improvements['parameter_stability']
        ]
        
        bars = plt.bar(improvement_metrics, improvement_values, color=['green', 'red', 'blue'], alpha=0.7)
        plt.title('Improvement Percentages', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Regime Detection Performance
        plt.subplot(3, 3, 5)
        original_regime = original_results['status']['regime_detection']['bandit_performance']
        optimized_regime = optimized_results['status']['regime_detection']['bandit_performance']
        
        metrics = ['Total Pulls', 'Exploration Rate']
        original_values = [original_regime['total_pulls'], original_regime['exploration_rate']]
        optimized_values = [optimized_regime['total_pulls'], optimized_regime['exploration_rate']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original', alpha=0.7, color='blue')
        plt.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.7, color='green')
        
        plt.title('Regime Detection Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        
        # 6. Kalman Optimization Performance
        plt.subplot(3, 3, 6)
        original_kalman = original_results['status']['kalman_optimization']['optimizer_performance']
        optimized_kalman = optimized_results['status']['kalman_optimization']['optimizer_performance']
        
        metrics = ['Total Updates', 'Hessian Condition', 'Parameter Stability']
        original_values = [
            original_kalman['total_updates'],
            original_kalman['hessian_condition'],
            original_kalman['parameter_stability']
        ]
        optimized_values = [
            optimized_kalman['total_updates'],
            optimized_kalman['hessian_condition'],
            optimized_kalman['parameter_stability']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.bar(x - width/2, original_values, width, label='Original', alpha=0.7, color='blue')
        plt.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.7, color='green')
        
        plt.title('Kalman Optimization Performance', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        
        # 7. Performance Over Time
        plt.subplot(3, 3, 7)
        if original_results['performance_history'] and optimized_results['performance_history']:
            original_perf = original_results['performance_history']
            optimized_perf = optimized_results['performance_history']
            
            plt.plot(range(len(original_perf)), original_perf, 'b-', alpha=0.7, label='Original')
            plt.plot(range(len(optimized_perf)), optimized_perf, 'g-', alpha=0.7, label='Optimized')
            
            plt.title('Performance Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 8. Success Rate Assessment
        plt.subplot(3, 3, 8)
        success_metrics = ['System Performance', 'Total Regret', 'Parameter Stability']
        success_values = [
            1 if improvements['system_performance'] > 0 else 0,
            1 if improvements['total_regret'] > 0 else 0,
            1 if improvements['parameter_stability'] > 0 else 0
        ]
        
        bars = plt.bar(success_metrics, success_values, color=['green', 'red', 'blue'], alpha=0.7)
        plt.title('Success Rate Assessment', fontsize=14, fontweight='bold')
        plt.ylabel('Success (1=Yes, 0=No)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, success_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 9. Overall Assessment
        plt.subplot(3, 3, 9)
        positive_improvements = sum(success_values)
        total_metrics = len(success_values)
        success_rate = (positive_improvements / total_metrics) * 100
        
        assessment = ['Success Rate', 'Positive Improvements', 'Total Metrics']
        assessment_values = [success_rate, positive_improvements, total_metrics]
        
        bars = plt.bar(assessment, assessment_values, color=['purple', 'green', 'blue'], alpha=0.7)
        plt.title('Overall Assessment', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, assessment_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_optimized_system_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison visualization saved as: {filename}")
        
        plt.show()

def load_test_data() -> np.ndarray:
    """Load test data for optimized system comparison experiment"""
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
    """Run optimized system comparison experiment"""
    
    print("🔍 Starting EPIC 1.5 Optimized System Comparison Experiment")
    print("=" * 70)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run comparison experiment
    experiment = OptimizedSystemComparisonExperiment()
    
    try:
        results = experiment.run_comparison_experiment(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_optimized_system_comparison_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Optimized system comparison experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("OPTIMIZED SYSTEM COMPARISON REPORT")
        print("=" * 70)
        print(results['comparison_report'])
        
    except Exception as e:
        print(f"❌ Error running comparison experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
