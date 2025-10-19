#!/usr/bin/env python3
"""
EPIC 1.5: Parameter Stability Enhancement Experiment

This experiment tests the enhanced parameter stability system with:
1. Adaptive parameter bounds
2. Parameter smoothing mechanisms
3. Parameter validation
4. Parameter history analysis
5. Dynamic parameter adjustment

Goal: Improve parameter stability (17.0% contribution) for greater consistency
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

from algorithms.adaptive_parameter_stability import AdaptiveParameterStability

class ParameterStabilityEnhancementExperiment:
    """Experiment to test enhanced parameter stability capabilities"""
    
    def __init__(self):
        self.results = {}
        
    def run_parameter_stability_enhancement_experiment(self, test_periods: int = 200) -> Dict:
        """Run parameter stability enhancement experiment"""
        
        print("🔧 Starting PARAMETER STABILITY ENHANCEMENT Experiment")
        print("=" * 70)
        print("🎯 Focus: Improve parameter stability (17.0% contribution) for greater consistency")
        print("")
        
        # 1. Initialize parameter stability system
        print("🔧 Initializing adaptive parameter stability system...")
        stability_system = AdaptiveParameterStability(
            window_size=20,
            stability_threshold=0.85
        )
        
        # 2. Test baseline parameter stability
        print("\n📊 Testing baseline parameter stability...")
        baseline_results = self._test_baseline_parameter_stability(test_periods)
        
        # 3. Test enhanced parameter stability
        print("\n🧪 Testing enhanced parameter stability...")
        enhanced_results = self._test_enhanced_parameter_stability(stability_system, test_periods)
        
        # 4. Analyze improvements
        print("\n🔍 Analyzing parameter stability improvements...")
        improvement_analysis = self._analyze_parameter_stability_improvements(
            baseline_results, enhanced_results
        )
        
        # 5. Generate enhancement report
        print("\n📋 Generating parameter stability enhancement report...")
        report = self._generate_parameter_stability_report(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        # 6. Create enhancement visualizations
        print("\n🎨 Creating parameter stability visualizations...")
        self._create_parameter_stability_visualizations(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        return {
            'baseline_results': baseline_results,
            'enhanced_results': enhanced_results,
            'improvement_analysis': improvement_analysis,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _test_baseline_parameter_stability(self, test_periods: int) -> Dict:
        """Test baseline parameter stability (no stability mechanisms)"""
        
        results = []
        parameter_values = {
            'learning_rates': [],
            'confidence_levels': [],
            'adaptation_rates': [],
            'forgetting_factors': []
        }
        stability_scores = []
        
        # Simulate parameter updates without stability mechanisms
        for i in range(test_periods):
            # Generate noisy parameter updates
            base_learning_rate = 0.01
            base_confidence = 0.95
            base_adaptation_rate = 0.01
            base_forgetting_factor = 0.95
            
            # Add noise to simulate parameter instability
            noise_factor = 0.1
            learning_rate = base_learning_rate * (1 + np.random.normal(0, noise_factor))
            confidence = base_confidence * (1 + np.random.normal(0, noise_factor * 0.5))
            adaptation_rate = base_adaptation_rate * (1 + np.random.normal(0, noise_factor))
            forgetting_factor = base_forgetting_factor * (1 + np.random.normal(0, noise_factor * 0.5))
            
            # Apply basic bounds
            learning_rate = max(0.001, min(0.1, learning_rate))
            confidence = max(0.5, min(0.99, confidence))
            adaptation_rate = max(0.001, min(0.05, adaptation_rate))
            forgetting_factor = max(0.8, min(0.99, forgetting_factor))
            
            # Store values
            parameter_values['learning_rates'].append(learning_rate)
            parameter_values['confidence_levels'].append(confidence)
            parameter_values['adaptation_rates'].append(adaptation_rate)
            parameter_values['forgetting_factors'].append(forgetting_factor)
            
            # Calculate stability score (simplified)
            if i > 0:
                recent_values = [parameter_values['learning_rates'][-1], 
                               parameter_values['confidence_levels'][-1],
                               parameter_values['adaptation_rates'][-1],
                               parameter_values['forgetting_factors'][-1]]
                prev_values = [parameter_values['learning_rates'][-2], 
                             parameter_values['confidence_levels'][-2],
                             parameter_values['adaptation_rates'][-2],
                             parameter_values['forgetting_factors'][-2]]
                
                # Calculate volatility
                volatility = np.mean([abs(new - old) / (old + 1e-8) for new, old in zip(recent_values, prev_values)])
                stability_score = max(0, 1.0 - volatility)
                stability_scores.append(stability_score)
            else:
                stability_scores.append(0.5)
            
            results.append({
                'period': i,
                'learning_rate': learning_rate,
                'confidence': confidence,
                'adaptation_rate': adaptation_rate,
                'forgetting_factor': forgetting_factor,
                'stability_score': stability_scores[-1]
            })
        
        # Calculate metrics
        mean_stability = np.mean(stability_scores)
        stability_std = np.std(stability_scores)
        
        return {
            'system_type': 'BaselineParameterStability',
            'results': results,
            'parameter_values': parameter_values,
            'stability_scores': stability_scores,
            'mean_stability': mean_stability,
            'stability_std': stability_std
        }
    
    def _test_enhanced_parameter_stability(self, stability_system: AdaptiveParameterStability, 
                                        test_periods: int) -> Dict:
        """Test enhanced parameter stability with stability mechanisms"""
        
        results = []
        parameter_values = {
            'learning_rates': [],
            'confidence_levels': [],
            'adaptation_rates': [],
            'forgetting_factors': []
        }
        stability_scores = []
        
        # Simulate parameter updates with stability mechanisms
        for i in range(test_periods):
            # Generate noisy parameter updates
            base_learning_rate = 0.01
            base_confidence = 0.95
            base_adaptation_rate = 0.01
            base_forgetting_factor = 0.95
            
            # Add noise to simulate parameter instability
            noise_factor = 0.1
            raw_learning_rate = base_learning_rate * (1 + np.random.normal(0, noise_factor))
            raw_confidence = base_confidence * (1 + np.random.normal(0, noise_factor * 0.5))
            raw_adaptation_rate = base_adaptation_rate * (1 + np.random.normal(0, noise_factor))
            raw_forgetting_factor = base_forgetting_factor * (1 + np.random.normal(0, noise_factor * 0.5))
            
            # Apply stability mechanisms
            new_parameters = {
                'learning_rates': raw_learning_rate,
                'confidence_levels': raw_confidence,
                'adaptation_rates': raw_adaptation_rate,
                'forgetting_factors': raw_forgetting_factor
            }
            
            # Simulate performance metric
            performance_metric = 0.8 + np.random.normal(0, 0.1)
            
            # Update parameters with stability mechanisms
            stabilized_parameters = stability_system.update_parameters(new_parameters, performance_metric)
            
            # Store values
            parameter_values['learning_rates'].append(stabilized_parameters['learning_rates'])
            parameter_values['confidence_levels'].append(stabilized_parameters['confidence_levels'])
            parameter_values['adaptation_rates'].append(stabilized_parameters['adaptation_rates'])
            parameter_values['forgetting_factors'].append(stabilized_parameters['forgetting_factors'])
            
            # Get stability score from system
            stability_metrics = stability_system.get_parameter_stability_metrics()
            if 'overall' in stability_metrics:
                stability_score = stability_metrics['overall']['current_stability']
            else:
                stability_score = 0.5
            
            stability_scores.append(stability_score)
            
            results.append({
                'period': i,
                'learning_rate': stabilized_parameters['learning_rates'],
                'confidence': stabilized_parameters['confidence_levels'],
                'adaptation_rate': stabilized_parameters['adaptation_rates'],
                'forgetting_factor': stabilized_parameters['forgetting_factors'],
                'stability_score': stability_score,
                'stability_metrics': stability_metrics
            })
        
        # Calculate metrics
        mean_stability = np.mean(stability_scores)
        stability_std = np.std(stability_scores)
        
        return {
            'system_type': 'EnhancedParameterStability',
            'results': results,
            'parameter_values': parameter_values,
            'stability_scores': stability_scores,
            'mean_stability': mean_stability,
            'stability_std': stability_std
        }
    
    def _analyze_parameter_stability_improvements(self, baseline_results: Dict, enhanced_results: Dict) -> Dict:
        """Analyze improvements in parameter stability"""
        
        analysis = {}
        
        # Overall stability improvement
        baseline_stability = baseline_results['mean_stability']
        enhanced_stability = enhanced_results['mean_stability']
        stability_improvement = ((enhanced_stability - baseline_stability) / baseline_stability * 100) if baseline_stability > 0 else 0
        
        # Stability consistency improvement
        baseline_std = baseline_results['stability_std']
        enhanced_std = enhanced_results['stability_std']
        consistency_improvement = ((baseline_std - enhanced_std) / baseline_std * 100) if baseline_std > 0 else 0
        
        # Parameter-specific improvements
        parameter_improvements = {}
        for param_name in ['learning_rates', 'confidence_levels', 'adaptation_rates', 'forgetting_factors']:
            baseline_values = baseline_results['parameter_values'][param_name]
            enhanced_values = enhanced_results['parameter_values'][param_name]
            
            # Calculate volatility
            baseline_volatility = np.std(baseline_values)
            enhanced_volatility = np.std(enhanced_values)
            
            volatility_improvement = ((baseline_volatility - enhanced_volatility) / baseline_volatility * 100) if baseline_volatility > 0 else 0
            
            parameter_improvements[param_name] = {
                'baseline_volatility': baseline_volatility,
                'enhanced_volatility': enhanced_volatility,
                'volatility_improvement': volatility_improvement
            }
        
        analysis['overall'] = {
            'stability_improvement': stability_improvement,
            'consistency_improvement': consistency_improvement,
            'baseline_stability': baseline_stability,
            'enhanced_stability': enhanced_stability,
            'baseline_std': baseline_std,
            'enhanced_std': enhanced_std
        }
        
        analysis['parameter_specific'] = parameter_improvements
        
        # Overall assessment
        improvements = [stability_improvement, consistency_improvement]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        total_improvements = len(improvements)
        
        analysis['assessment'] = {
            'success_rate': (positive_improvements / total_improvements) * 100,
            'average_improvement': np.mean(improvements),
            'positive_improvements': positive_improvements,
            'total_improvements': total_improvements
        }
        
        return analysis
    
    def _generate_parameter_stability_report(self, baseline_results: Dict, enhanced_results: Dict, 
                                           improvement_analysis: Dict) -> str:
        """Generate parameter stability enhancement report"""
        
        report = []
        report.append("EPIC 1.5: PARAMETER STABILITY ENHANCEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        report.append("🎯 FOCUS: Improve parameter stability (17.0% contribution) for greater consistency")
        report.append("")
        
        # Overall Performance Comparison
        report.append("📊 OVERALL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        overall = improvement_analysis['overall']
        report.append(f"Baseline Stability: {overall['baseline_stability']:.4f}")
        report.append(f"Enhanced Stability: {overall['enhanced_stability']:.4f}")
        report.append(f"Stability Improvement: {overall['stability_improvement']:.1f}%")
        report.append("")
        report.append(f"Baseline Consistency: {overall['baseline_std']:.4f}")
        report.append(f"Enhanced Consistency: {overall['enhanced_std']:.4f}")
        report.append(f"Consistency Improvement: {overall['consistency_improvement']:.1f}%")
        report.append("")
        
        # Parameter-Specific Improvements
        report.append("🔧 PARAMETER-SPECIFIC IMPROVEMENTS")
        report.append("-" * 40)
        for param_name, metrics in improvement_analysis['parameter_specific'].items():
            report.append(f"{param_name.upper()}:")
            report.append(f"  Baseline Volatility: {metrics['baseline_volatility']:.4f}")
            report.append(f"  Enhanced Volatility: {metrics['enhanced_volatility']:.4f}")
            report.append(f"  Volatility Improvement: {metrics['volatility_improvement']:.1f}%")
            report.append("")
        
        # Assessment
        report.append("🏆 ASSESSMENT")
        report.append("-" * 40)
        assessment = improvement_analysis['assessment']
        report.append(f"Success Rate: {assessment['success_rate']:.1f}%")
        report.append(f"Average Improvement: {assessment['average_improvement']:.1f}%")
        report.append(f"Positive Improvements: {assessment['positive_improvements']}/{assessment['total_improvements']}")
        report.append("")
        
        # Conclusion
        report.append("🎯 CONCLUSION")
        report.append("-" * 40)
        if assessment['success_rate'] >= 75:
            report.append("Status: 🎉 EXCELLENT - Enhanced parameter stability shows significant improvements!")
        elif assessment['success_rate'] >= 50:
            report.append("Status: ✅ GOOD - Enhanced parameter stability shows notable improvements!")
        elif assessment['success_rate'] >= 25:
            report.append("Status: ⚠️ PARTIAL - Enhanced parameter stability shows some improvements")
        else:
            report.append("Status: ❌ POOR - Enhanced parameter stability does not show improvements")
        
        report.append("")
        report.append("Key improvements:")
        report.append("1. 🔧 Adaptive parameter bounds to prevent extreme values")
        report.append("2. 📊 Parameter smoothing to reduce oscillations")
        report.append("3. ✅ Parameter validation to ensure consistency")
        report.append("4. 📈 Parameter history analysis for trend detection")
        report.append("5. ⚙️ Dynamic parameter adjustment based on performance")
        
        report.append("\n" + "=" * 60)
        report.append("End of Parameter Stability Enhancement Report")
        
        return "\n".join(report)
    
    def _create_parameter_stability_visualizations(self, baseline_results: Dict, enhanced_results: Dict, 
                                                 improvement_analysis: Dict):
        """Create parameter stability enhancement visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Stability Score Comparison
        plt.subplot(3, 3, 1)
        systems = ['Baseline', 'Enhanced']
        stabilities = [improvement_analysis['overall']['baseline_stability'], 
                      improvement_analysis['overall']['enhanced_stability']]
        
        bars = plt.bar(systems, stabilities, color=['blue', 'green'], alpha=0.7)
        plt.title('Parameter Stability Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Stability Score')
        
        # Add value labels
        for bar, value in zip(bars, stabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Consistency Comparison
        plt.subplot(3, 3, 2)
        consistencies = [improvement_analysis['overall']['baseline_std'], 
                        improvement_analysis['overall']['enhanced_std']]
        
        bars = plt.bar(systems, consistencies, color=['blue', 'green'], alpha=0.7)
        plt.title('Parameter Consistency Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Consistency (Lower is Better)')
        
        # Add value labels
        for bar, value in zip(bars, consistencies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Improvement Percentages
        plt.subplot(3, 3, 3)
        improvement_metrics = ['Stability', 'Consistency']
        improvement_values = [
            improvement_analysis['overall']['stability_improvement'],
            improvement_analysis['overall']['consistency_improvement']
        ]
        
        bars = plt.bar(improvement_metrics, improvement_values, color=['green', 'blue'], alpha=0.7)
        plt.title('Improvement Percentages\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Parameter Volatility Improvements
        plt.subplot(3, 3, 4)
        param_names = list(improvement_analysis['parameter_specific'].keys())
        volatility_improvements = [improvement_analysis['parameter_specific'][param]['volatility_improvement'] 
                                 for param in param_names]
        
        bars = plt.bar(param_names, volatility_improvements, color=['green', 'blue', 'orange', 'purple'], alpha=0.7)
        plt.title('Parameter Volatility Improvements\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Volatility Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, volatility_improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Stability Over Time - Baseline
        plt.subplot(3, 3, 5)
        periods = range(len(baseline_results['stability_scores']))
        plt.plot(periods, baseline_results['stability_scores'], label='Baseline', alpha=0.7, color='blue')
        plt.title('Baseline Stability Over Time\n(Parameter Stability)', fontsize=14, fontweight='bold')
        plt.ylabel('Stability Score')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Stability Over Time - Enhanced
        plt.subplot(3, 3, 6)
        periods = range(len(enhanced_results['stability_scores']))
        plt.plot(periods, enhanced_results['stability_scores'], label='Enhanced', alpha=0.7, color='green')
        plt.title('Enhanced Stability Over Time\n(Parameter Stability)', fontsize=14, fontweight='bold')
        plt.ylabel('Stability Score')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Learning Rate Over Time - Baseline
        plt.subplot(3, 3, 7)
        periods = range(len(baseline_results['parameter_values']['learning_rates']))
        plt.plot(periods, baseline_results['parameter_values']['learning_rates'], label='Baseline', alpha=0.7, color='blue')
        plt.title('Baseline Learning Rate Over Time\n(Parameter Stability)', fontsize=14, fontweight='bold')
        plt.ylabel('Learning Rate')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Learning Rate Over Time - Enhanced
        plt.subplot(3, 3, 8)
        periods = range(len(enhanced_results['parameter_values']['learning_rates']))
        plt.plot(periods, enhanced_results['parameter_values']['learning_rates'], label='Enhanced', alpha=0.7, color='green')
        plt.title('Enhanced Learning Rate Over Time\n(Parameter Stability)', fontsize=14, fontweight='bold')
        plt.ylabel('Learning Rate')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Overall Assessment
        plt.subplot(3, 3, 9)
        assessment_categories = ['Excellent', 'Good', 'Partial', 'Poor']
        assessment_values = [0, 0, 0, 0]
        
        success_rate = improvement_analysis['assessment']['success_rate']
        if success_rate >= 75:
            assessment_values[0] = 1
        elif success_rate >= 50:
            assessment_values[1] = 1
        elif success_rate >= 25:
            assessment_values[2] = 1
        else:
            assessment_values[3] = 1
        
        bars = plt.bar(assessment_categories, assessment_values, color=['green', 'lightgreen', 'orange', 'red'], alpha=0.7)
        plt.title('Overall Assessment\n(Parameter Stability)', fontsize=14, fontweight='bold')
        plt.ylabel('Assessment')
        
        # Add value labels
        for bar, value in zip(bars, assessment_values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_parameter_stability_enhancement_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Parameter stability enhancement visualization saved as: {filename}")
        
        plt.show()

def main():
    """Run parameter stability enhancement experiment"""
    
    print("🔧 Starting EPIC 1.5 Parameter Stability Enhancement Experiment")
    print("=" * 70)
    print("🎯 Focus: Improve parameter stability (17.0% contribution) for greater consistency")
    print("")
    
    # Run parameter stability enhancement experiment
    experiment = ParameterStabilityEnhancementExperiment()
    
    try:
        results = experiment.run_parameter_stability_enhancement_experiment(test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_parameter_stability_enhancement_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Parameter stability enhancement experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("PARAMETER STABILITY ENHANCEMENT REPORT")
        print("=" * 70)
        print(results['report'])
        
    except Exception as e:
        print(f"❌ Error running parameter stability enhancement experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
