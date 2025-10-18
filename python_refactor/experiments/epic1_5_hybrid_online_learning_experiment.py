#!/usr/bin/env python3
"""
EPIC 1.5: Hybrid Online Learning Experiment

This experiment tests the complete hybrid online learning system with:
- Contextual Bandits for regime detection (O(√(dT log T)) regret)
- Adaptive Mirror Descent for parameter adaptation (O(√(G²T)) regret)
- Online Newton Step for Kalman optimization (O(d log T) regret)

Expected to show significant performance improvements with theoretical guarantees.
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

class HybridOnlineLearningExperiment:
    """Experiment for testing hybrid online learning system"""
    
    def __init__(self):
        self.results = {}
        self.experiment_history = []
        
    def run_hybrid_experiment(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run comprehensive hybrid online learning experiment"""
        
        print("🚀 Starting EPIC 1.5 Hybrid Online Learning Experiment")
        print("=" * 70)
        
        # Initialize hybrid system
        print("\n🔧 Initializing Hybrid Online Learning System...")
        hybrid_system = HybridOnlineLearningSystem(
            learning_rates={
                'regime_detection': 0.01,
                'parameter_adaptation': 0.01,
                'kalman_optimization': 0.01
            },
            confidence=0.95,
            window_size=10
        )
        
        # Generate observations and rewards
        observations = data.copy()
        rewards = self._generate_rewards(data)
        
        # Run experiment
        print("\n📊 Running Hybrid System Experiment...")
        experiment = HybridSystemExperiment(hybrid_system)
        results = experiment.run_experiment(
            data, observations, rewards, "hybrid_online_learning_experiment"
        )
        
        # Calculate performance metrics
        print("\n📈 Calculating Performance Metrics...")
        performance_metrics = self._calculate_performance_metrics(results, hybrid_system)
        
        # Generate comparison report
        print("\n📋 Generating Comparison Report...")
        comparison_report = self._generate_comparison_report(performance_metrics, results)
        
        # Create visualizations
        print("\n🎨 Creating Visualizations...")
        self._create_visualizations(results, performance_metrics)
        
        return {
            'experiment_results': results,
            'performance_metrics': performance_metrics,
            'comparison_report': comparison_report,
            'hybrid_system': hybrid_system,
            'timestamp': datetime.now().isoformat()
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
    
    def _calculate_performance_metrics(self, results: Dict, hybrid_system: HybridOnlineLearningSystem) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Get system status
        system_status = hybrid_system.get_system_status()
        
        # Performance metrics
        performance_metrics = {
            'system_overview': system_status['system_overview'],
            'regime_detection': system_status['regime_detection'],
            'parameter_adaptation': system_status['parameter_adaptation'],
            'kalman_optimization': system_status['kalman_optimization'],
            'regret_bounds': system_status['regret_bounds'],
            'experiment_metrics': results.get('performance_metrics', {}),
            'regret_analysis': results.get('regret_analysis', {})
        }
        
        # Calculate additional metrics
        if results.get('system_updates'):
            # Regime detection accuracy
            regimes = [update['regime'] for update in results['system_updates']]
            regime_changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
            regime_stability = 1.0 - (regime_changes / max(1, len(regimes) - 1))
            
            # Regret progression
            regret_progression = [update['regret_bounds'] for update in results['system_updates']]
            
            performance_metrics['regime_stability'] = regime_stability
            performance_metrics['regret_progression'] = regret_progression
        
        return performance_metrics
    
    def _generate_comparison_report(self, performance_metrics: Dict, results: Dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("EPIC 1.5: Hybrid Online Learning Experiment Report")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # System Overview
        report.append("🔧 SYSTEM OVERVIEW")
        report.append("-" * 30)
        
        system_overview = performance_metrics['system_overview']
        report.append(f"Total Updates: {system_overview['total_updates']}")
        report.append(f"Average Performance: {system_overview['average_performance']:.4f}")
        report.append(f"Confidence Level: {system_overview['confidence_level']:.2%}")
        report.append(f"Window Size: {system_overview['window_size']}")
        
        # Regime Detection Performance
        report.append("\n🎯 REGIME DETECTION PERFORMANCE")
        report.append("-" * 30)
        
        regime_detection = performance_metrics['regime_detection']
        report.append(f"Current Regime: {regime_detection['regime_detection']['current_regime']}")
        report.append(f"Regime Confidence: {regime_detection['regime_detection']['confidence']:.2%}")
        report.append(f"Regret Bound: {regime_detection['regime_detection']['regret_bound']:.4f}")
        
        # Bandit Performance
        bandit_performance = regime_detection['bandit_performance']
        report.append(f"Total Pulls: {bandit_performance['total_pulls']}")
        report.append(f"Regime Distribution: {bandit_performance['regime_distribution']}")
        report.append(f"Exploration Rate: {bandit_performance['exploration_rate']:.2%}")
        
        # Parameter Adaptation Performance
        report.append("\n⚙️ PARAMETER ADAPTATION PERFORMANCE")
        report.append("-" * 30)
        
        param_adaptation = performance_metrics['parameter_adaptation']
        optimizer_perf = param_adaptation['optimizer_performance']
        report.append(f"Total Updates: {optimizer_perf['total_updates']}")
        report.append(f"Average Gradient Norm: {optimizer_perf['average_gradient_norm']:.4f}")
        report.append(f"Regret Bound: {optimizer_perf['regret_bound']:.4f}")
        report.append(f"Parameter Stability: {optimizer_perf['parameter_stability']:.4f}")
        report.append(f"Learning Rate Adaptation: {optimizer_perf['learning_rate_adaptation']:.4f}")
        
        # Kalman Optimization Performance
        report.append("\n🔍 KALMAN OPTIMIZATION PERFORMANCE")
        report.append("-" * 30)
        
        kalman_optimization = performance_metrics['kalman_optimization']
        kalman_perf = kalman_optimization['optimizer_performance']
        report.append(f"Total Optimizations: {kalman_perf['total_updates']}")
        report.append(f"Regret Bound: {kalman_perf['regret_bound']:.4f}")
        report.append(f"Hessian Condition: {kalman_perf['hessian_condition']:.4f}")
        report.append(f"Parameter Stability: {kalman_perf['parameter_stability']:.4f}")
        
        # Regret Bounds Analysis
        report.append("\n📊 REGRET BOUNDS ANALYSIS")
        report.append("-" * 30)
        
        regret_bounds = performance_metrics['regret_bounds']
        report.append(f"Regime Detection Regret: {regret_bounds['regime_detection']:.4f}")
        report.append(f"Parameter Adaptation Regret: {regret_bounds['parameter_adaptation']:.4f}")
        report.append(f"Kalman Optimization Regret: {regret_bounds['kalman_optimization']:.4f}")
        report.append(f"Total Regret: {regret_bounds['total_regret']:.4f}")
        
        # Theoretical Guarantees
        report.append("\n🏆 THEORETICAL GUARANTEES")
        report.append("-" * 30)
        
        T = system_overview['total_updates']
        if T > 0:
            # Regime detection: O(√(dT log T))
            d_regime = 4  # Context dimension
            regime_theoretical = np.sqrt(d_regime * T * np.log(T + 1))
            report.append(f"Regime Detection Theoretical: {regime_theoretical:.4f}")
            report.append(f"Regime Detection Actual: {regret_bounds['regime_detection']:.4f}")
            report.append(f"Regime Detection Ratio: {regret_bounds['regime_detection'] / regime_theoretical:.4f}")
            
            # Parameter adaptation: O(√(G²T))
            G_squared = optimizer_perf['average_gradient_norm'] ** 2
            param_theoretical = np.sqrt(G_squared * T)
            report.append(f"Parameter Adaptation Theoretical: {param_theoretical:.4f}")
            report.append(f"Parameter Adaptation Actual: {regret_bounds['parameter_adaptation']:.4f}")
            report.append(f"Parameter Adaptation Ratio: {regret_bounds['parameter_adaptation'] / param_theoretical:.4f}")
            
            # Kalman optimization: O(d log T)
            d_kalman = 16  # Parameter dimension
            kalman_theoretical = d_kalman * np.log(T + 1)
            report.append(f"Kalman Optimization Theoretical: {kalman_theoretical:.4f}")
            report.append(f"Kalman Optimization Actual: {regret_bounds['kalman_optimization']:.4f}")
            report.append(f"Kalman Optimization Ratio: {regret_bounds['kalman_optimization'] / kalman_theoretical:.4f}")
        
        # Success Metrics
        report.append("\n🎯 SUCCESS METRICS")
        report.append("-" * 30)
        
        success_metrics = []
        
        # Check if regret bounds are within theoretical limits
        if T > 0:
            if regret_bounds['regime_detection'] <= regime_theoretical:
                success_metrics.append("✅ Regime detection regret within theoretical bound")
            else:
                success_metrics.append("❌ Regime detection regret exceeds theoretical bound")
            
            if regret_bounds['parameter_adaptation'] <= param_theoretical:
                success_metrics.append("✅ Parameter adaptation regret within theoretical bound")
            else:
                success_metrics.append("❌ Parameter adaptation regret exceeds theoretical bound")
            
            if regret_bounds['kalman_optimization'] <= kalman_theoretical:
                success_metrics.append("✅ Kalman optimization regret within theoretical bound")
            else:
                success_metrics.append("❌ Kalman optimization regret exceeds theoretical bound")
        
        # Check system performance
        if system_overview['average_performance'] > 0:
            success_metrics.append("✅ System shows positive average performance")
        else:
            success_metrics.append("❌ System shows negative average performance")
        
        # Check regime stability
        if 'regime_stability' in performance_metrics:
            if performance_metrics['regime_stability'] > 0.5:
                success_metrics.append("✅ Regime detection shows good stability")
            else:
                success_metrics.append("❌ Regime detection shows poor stability")
        
        for metric in success_metrics:
            report.append(f"  {metric}")
        
        # Overall Assessment
        report.append("\n🏆 OVERALL ASSESSMENT")
        report.append("-" * 30)
        
        total_success = sum(1 for metric in success_metrics if metric.startswith("✅"))
        total_metrics = len(success_metrics)
        success_rate = (total_success / total_metrics) * 100 if total_metrics > 0 else 0
        
        report.append(f"Success Rate: {success_rate:.1f}% ({total_success}/{total_metrics})")
        
        if success_rate >= 80:
            report.append("Status: 🎉 EXCELLENT - Most theoretical guarantees achieved!")
        elif success_rate >= 60:
            report.append("Status: ✅ GOOD - Significant theoretical guarantees achieved!")
        elif success_rate >= 40:
            report.append("Status: ⚠️  PARTIAL - Some theoretical guarantees achieved")
        else:
            report.append("Status: ❌ POOR - Few theoretical guarantees achieved")
        
        report.append("\n" + "=" * 60)
        report.append("End of Hybrid Online Learning Experiment Report")
        
        return "\n".join(report)
    
    def _create_visualizations(self, results: Dict, performance_metrics: Dict):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Regret Bounds Comparison
        plt.subplot(3, 3, 1)
        regret_bounds = performance_metrics['regret_bounds']
        components = ['Regime Detection', 'Parameter Adaptation', 'Kalman Optimization']
        regret_values = [
            regret_bounds['regime_detection'],
            regret_bounds['parameter_adaptation'],
            regret_bounds['kalman_optimization']
        ]
        
        bars = plt.bar(components, regret_values, color=['blue', 'green', 'red'], alpha=0.7)
        plt.title('Regret Bounds by Component', fontsize=14, fontweight='bold')
        plt.ylabel('Regret Bound')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, regret_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Regime Detection Performance
        plt.subplot(3, 3, 2)
        regime_detection = performance_metrics['regime_detection']
        bandit_perf = regime_detection['bandit_performance']
        
        regimes = list(bandit_perf['regime_distribution'].keys())
        distribution = list(bandit_perf['regime_distribution'].values())
        
        bars = plt.bar(regimes, distribution, color=['green', 'red', 'blue'], alpha=0.7)
        plt.title('Regime Distribution', fontsize=14, fontweight='bold')
        plt.ylabel('Distribution')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, distribution):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Parameter Adaptation Performance
        plt.subplot(3, 3, 3)
        param_adaptation = performance_metrics['parameter_adaptation']
        optimizer_perf = param_adaptation['optimizer_performance']
        
        metrics = ['Total Updates', 'Gradient Norm', 'Regret Bound', 'Parameter Stability']
        values = [
            optimizer_perf['total_updates'],
            optimizer_perf['average_gradient_norm'],
            optimizer_perf['regret_bound'],
            optimizer_perf['parameter_stability']
        ]
        
        bars = plt.bar(metrics, values, color='green', alpha=0.7)
        plt.title('Parameter Adaptation Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Kalman Optimization Performance
        plt.subplot(3, 3, 4)
        kalman_optimization = performance_metrics['kalman_optimization']
        kalman_perf = kalman_optimization['optimizer_performance']
        
        metrics = ['Total Updates', 'Regret Bound', 'Hessian Condition', 'Parameter Stability']
        values = [
            kalman_perf['total_updates'],
            kalman_perf['regret_bound'],
            kalman_perf['hessian_condition'],
            kalman_perf['parameter_stability']
        ]
        
        bars = plt.bar(metrics, values, color='red', alpha=0.7)
        plt.title('Kalman Optimization Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Theoretical vs Actual Regret
        plt.subplot(3, 3, 5)
        T = performance_metrics['system_overview']['total_updates']
        if T > 0:
            # Theoretical bounds
            d_regime = 4
            regime_theoretical = np.sqrt(d_regime * T * np.log(T + 1))
            G_squared = optimizer_perf['average_gradient_norm'] ** 2
            param_theoretical = np.sqrt(G_squared * T)
            d_kalman = 16
            kalman_theoretical = d_kalman * np.log(T + 1)
            
            theoretical = [regime_theoretical, param_theoretical, kalman_theoretical]
            actual = [regret_bounds['regime_detection'], regret_bounds['parameter_adaptation'], regret_bounds['kalman_optimization']]
            
            x = np.arange(len(components))
            width = 0.35
            
            plt.bar(x - width/2, theoretical, width, label='Theoretical', alpha=0.7)
            plt.bar(x + width/2, actual, width, label='Actual', alpha=0.7)
            
            plt.title('Theoretical vs Actual Regret', fontsize=14, fontweight='bold')
            plt.ylabel('Regret')
            plt.xticks(x, components, rotation=45)
            plt.legend()
        
        # 6. System Performance Over Time
        plt.subplot(3, 3, 6)
        if results.get('system_updates'):
            timesteps = [update['timestep'] for update in results['system_updates']]
            regime_confidences = [update['regime_confidence'] for update in results['system_updates']]
            
            plt.plot(timesteps, regime_confidences, 'b-', alpha=0.7, label='Regime Confidence')
            plt.title('System Performance Over Time', fontsize=14, fontweight='bold')
            plt.xlabel('Time Steps')
            plt.ylabel('Regime Confidence')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 7. Regret Progression
        plt.subplot(3, 3, 7)
        if 'regret_progression' in performance_metrics:
            regret_progression = performance_metrics['regret_progression']
            if regret_progression:
                timesteps = list(range(len(regret_progression)))
                regime_regret = [r['regime_detection'] for r in regret_progression]
                param_regret = [r['parameter_adaptation'] for r in regret_progression]
                kalman_regret = [r['kalman_optimization'] for r in regret_progression]
                
                plt.plot(timesteps, regime_regret, 'b-', alpha=0.7, label='Regime Detection')
                plt.plot(timesteps, param_regret, 'g-', alpha=0.7, label='Parameter Adaptation')
                plt.plot(timesteps, kalman_regret, 'r-', alpha=0.7, label='Kalman Optimization')
                
                plt.title('Regret Progression Over Time', fontsize=14, fontweight='bold')
                plt.xlabel('Time Steps')
                plt.ylabel('Regret')
                plt.legend()
                plt.grid(True, alpha=0.3)
        
        # 8. System Overview
        plt.subplot(3, 3, 8)
        system_overview = performance_metrics['system_overview']
        metrics = ['Total Updates', 'Average Performance', 'Confidence Level', 'Window Size']
        values = [
            system_overview['total_updates'],
            system_overview['average_performance'],
            system_overview['confidence_level'],
            system_overview['window_size']
        ]
        
        bars = plt.bar(metrics, values, color='purple', alpha=0.7)
        plt.title('System Overview', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 9. Success Rate
        plt.subplot(3, 3, 9)
        # Calculate success rate based on regret bounds
        if T > 0:
            regime_success = 1 if regret_bounds['regime_detection'] <= regime_theoretical else 0
            param_success = 1 if regret_bounds['parameter_adaptation'] <= param_theoretical else 0
            kalman_success = 1 if regret_bounds['kalman_optimization'] <= kalman_theoretical else 0
            
            success_rates = [regime_success, param_success, kalman_success]
            components = ['Regime Detection', 'Parameter Adaptation', 'Kalman Optimization']
            
            bars = plt.bar(components, success_rates, color=['blue', 'green', 'red'], alpha=0.7)
            plt.title('Theoretical Guarantee Success Rate', fontsize=14, fontweight='bold')
            plt.ylabel('Success (1=Yes, 0=No)')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, success_rates):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_hybrid_online_learning_experiment_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {filename}")
        
        plt.show()

def load_test_data() -> np.ndarray:
    """Load test data for hybrid online learning experiment"""
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
    """Run hybrid online learning experiment"""
    
    print("🚀 Starting EPIC 1.5 Hybrid Online Learning Experiment")
    print("=" * 70)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run hybrid experiment
    experiment = HybridOnlineLearningExperiment()
    
    try:
        results = experiment.run_hybrid_experiment(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_hybrid_online_learning_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Hybrid online learning experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("HYBRID ONLINE LEARNING EXPERIMENT REPORT")
        print("=" * 70)
        print(results['comparison_report'])
        
    except Exception as e:
        print(f"❌ Error running hybrid experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
