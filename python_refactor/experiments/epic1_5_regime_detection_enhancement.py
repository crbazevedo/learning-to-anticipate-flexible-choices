#!/usr/bin/env python3
"""
EPIC 1.5: Regime Detection Enhancement Experiment

This experiment tests the enhanced regime detection system with:
1. Advanced ensemble methods
2. Temporal consistency checks
3. Advanced feature engineering
4. Regime transition probabilities
5. Adaptive confidence thresholds

Goal: Increase regime confidence from 84.54% to 90%+
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

from algorithms.advanced_regime_detection_v2 import AdvancedRegimeDetectorV2
from algorithms.enhanced_hybrid_system import EnhancedHybridOnlineLearningSystem

class RegimeDetectionEnhancementExperiment:
    """Experiment to test enhanced regime detection capabilities"""
    
    def __init__(self):
        self.results = {}
        
    def run_regime_detection_enhancement_experiment(self, test_periods: int = 200) -> Dict:
        """Run regime detection enhancement experiment"""
        
        print("🎯 Starting REGIME DETECTION ENHANCEMENT Experiment")
        print("=" * 70)
        print("🎯 Focus: Increase regime confidence from 84.54% to 90%+")
        print("")
        
        # 1. Generate realistic financial data with clear regimes
        print("📊 Generating realistic financial data with clear regimes...")
        data, regime_labels = self._generate_regime_rich_data()
        
        # 2. Initialize enhanced regime detector
        print("\n🔧 Initializing enhanced regime detector...")
        enhanced_detector = AdvancedRegimeDetectorV2(
            window_size=20,
            confidence_threshold=0.7
        )
        
        # 3. Train models on historical data
        print("\n📚 Training ensemble models...")
        training_data = data[:100]  # Use first 100 periods for training
        training_labels = regime_labels[:100]
        enhanced_detector.train_models(training_data, training_labels)
        
        # 4. Test enhanced regime detection
        print("\n🧪 Testing enhanced regime detection...")
        enhanced_results = self._test_enhanced_regime_detection(
            enhanced_detector, data, regime_labels, test_periods
        )
        
        # 5. Compare with baseline
        print("\n📊 Comparing with baseline...")
        baseline_results = self._test_baseline_regime_detection(data, regime_labels, test_periods)
        
        # 6. Analyze improvements
        print("\n🔍 Analyzing improvements...")
        improvement_analysis = self._analyze_regime_detection_improvements(
            baseline_results, enhanced_results
        )
        
        # 7. Generate enhancement report
        print("\n📋 Generating enhancement report...")
        report = self._generate_regime_detection_report(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        # 8. Create enhancement visualizations
        print("\n🎨 Creating enhancement visualizations...")
        self._create_regime_detection_visualizations(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        return {
            'baseline_results': baseline_results,
            'enhanced_results': enhanced_results,
            'improvement_analysis': improvement_analysis,
            'report': report,
            'data_used': data,
            'regime_labels': regime_labels,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_regime_rich_data(self) -> Tuple[np.ndarray, List[str]]:
        """Generate financial data with clear regime patterns"""
        
        np.random.seed(42)
        n_periods = 300
        data = []
        regime_labels = []
        
        # Bull market (periods 0-80) - High ROI, Low Risk
        for i in range(80):
            trend = 0.03 + 0.0005 * i  # Increasing trend
            volatility = 0.008 + 0.0001 * i  # Low volatility
            roi = trend + np.random.normal(0, volatility)
            risk = 0.06 + 0.0001 * i + np.random.normal(0, 0.010)  # Low risk
            data.append([roi, risk])
            regime_labels.append('bull_market')
        
        # Bear market (periods 80-160) - Low ROI, High Risk
        for i in range(80):
            trend = -0.02 - 0.0003 * i  # Increasing decline
            volatility = 0.025 + 0.0002 * i  # High volatility
            roi = trend + np.random.normal(0, volatility)
            risk = 0.25 + 0.0002 * i + np.random.normal(0, 0.030)  # High risk
            data.append([roi, risk])
            regime_labels.append('bear_market')
        
        # Sideways market (periods 160-240) - Moderate ROI, Moderate Risk
        for i in range(80):
            trend = 0.008 + 0.0001 * np.sin(i * 0.15)  # Oscillating trend
            volatility = 0.012 + 0.0001 * i  # Moderate volatility
            roi = trend + np.random.normal(0, volatility)
            risk = 0.12 + 0.0001 * i + np.random.normal(0, 0.015)  # Moderate risk
            data.append([roi, risk])
            regime_labels.append('sideways_market')
        
        # Recovery market (periods 240-300) - Improving ROI, Decreasing Risk
        for i in range(60):
            trend = 0.015 + 0.0003 * i  # Gradual recovery
            volatility = 0.015 - 0.0001 * i  # Decreasing volatility
            roi = trend + np.random.normal(0, volatility)
            risk = 0.10 - 0.0001 * i + np.random.normal(0, 0.012)  # Decreasing risk
            data.append([roi, risk])
            regime_labels.append('recovery_market')
        
        return np.array(data), regime_labels
    
    def _test_enhanced_regime_detection(self, detector: AdvancedRegimeDetectorV2, 
                                      data: np.ndarray, regime_labels: List[str], 
                                      test_periods: int) -> Dict:
        """Test enhanced regime detection"""
        
        results = []
        confidences = []
        accuracies = []
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            true_regime = regime_labels[i] if i < len(regime_labels) else 'sideways_market'
            
            # Detect regime
            detection_result = detector.detect_regime_advanced(window_data)
            
            # Calculate accuracy
            predicted_regime = detection_result['regime']
            confidence = detection_result['confidence']
            accuracy = 1.0 if predicted_regime == true_regime else 0.0
            
            results.append({
                'period': i,
                'true_regime': true_regime,
                'predicted_regime': predicted_regime,
                'confidence': confidence,
                'accuracy': accuracy,
                'probabilities': detection_result['probabilities'],
                'temporal_consistency': detection_result.get('temporal_consistency', False)
            })
            
            confidences.append(confidence)
            accuracies.append(accuracy)
        
        # Calculate metrics
        overall_accuracy = np.mean(accuracies)
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Regime-specific metrics
        regime_metrics = {}
        for regime in ['bull_market', 'bear_market', 'sideways_market', 'recovery_market']:
            regime_indices = [i for i, r in enumerate(regime_labels[:len(results)]) if r == regime]
            if regime_indices:
                regime_accuracies = [accuracies[i] for i in regime_indices]
                regime_confidences = [confidences[i] for i in regime_indices]
                
                regime_metrics[regime] = {
                    'accuracy': np.mean(regime_accuracies),
                    'confidence': np.mean(regime_confidences),
                    'count': len(regime_indices)
                }
        
        return {
            'detector_type': 'EnhancedRegimeDetectorV2',
            'results': results,
            'overall_accuracy': overall_accuracy,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'regime_metrics': regime_metrics,
            'confidences': confidences,
            'accuracies': accuracies
        }
    
    def _test_baseline_regime_detection(self, data: np.ndarray, regime_labels: List[str], 
                                      test_periods: int) -> Dict:
        """Test baseline regime detection for comparison"""
        
        results = []
        confidences = []
        accuracies = []
        
        for i in range(min(test_periods, len(data))):
            window_data = data[:i+1]
            true_regime = regime_labels[i] if i < len(regime_labels) else 'sideways_market'
            
            # Simple baseline regime detection
            roi, risk = data[i]
            
            # Simple rule-based detection
            if roi > 0.02 and risk < 0.12:
                predicted_regime = 'bull_market'
                confidence = 0.7
            elif roi < -0.01 and risk > 0.18:
                predicted_regime = 'bear_market'
                confidence = 0.7
            elif roi > 0.01 and risk < 0.15:
                predicted_regime = 'recovery_market'
                confidence = 0.6
            else:
                predicted_regime = 'sideways_market'
                confidence = 0.5
            
            # Add some randomness
            confidence += np.random.normal(0, 0.1)
            confidence = max(0.1, min(0.9, confidence))
            
            accuracy = 1.0 if predicted_regime == true_regime else 0.0
            
            results.append({
                'period': i,
                'true_regime': true_regime,
                'predicted_regime': predicted_regime,
                'confidence': confidence,
                'accuracy': accuracy
            })
            
            confidences.append(confidence)
            accuracies.append(accuracy)
        
        # Calculate metrics
        overall_accuracy = np.mean(accuracies)
        mean_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        return {
            'detector_type': 'BaselineRegimeDetector',
            'results': results,
            'overall_accuracy': overall_accuracy,
            'mean_confidence': mean_confidence,
            'confidence_std': confidence_std,
            'confidences': confidences,
            'accuracies': accuracies
        }
    
    def _analyze_regime_detection_improvements(self, baseline_results: Dict, enhanced_results: Dict) -> Dict:
        """Analyze improvements in regime detection"""
        
        analysis = {}
        
        # Overall accuracy improvement
        baseline_accuracy = baseline_results['overall_accuracy']
        enhanced_accuracy = enhanced_results['overall_accuracy']
        accuracy_improvement = ((enhanced_accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        
        # Confidence improvement
        baseline_confidence = baseline_results['mean_confidence']
        enhanced_confidence = enhanced_results['mean_confidence']
        confidence_improvement = ((enhanced_confidence - baseline_confidence) / baseline_confidence * 100) if baseline_confidence > 0 else 0
        
        # Confidence stability improvement
        baseline_confidence_std = baseline_results['confidence_std']
        enhanced_confidence_std = enhanced_results['confidence_std']
        stability_improvement = ((baseline_confidence_std - enhanced_confidence_std) / baseline_confidence_std * 100) if baseline_confidence_std > 0 else 0
        
        analysis['overall'] = {
            'accuracy_improvement': accuracy_improvement,
            'confidence_improvement': confidence_improvement,
            'stability_improvement': stability_improvement,
            'baseline_accuracy': baseline_accuracy,
            'enhanced_accuracy': enhanced_accuracy,
            'baseline_confidence': baseline_confidence,
            'enhanced_confidence': enhanced_confidence
        }
        
        # Regime-specific improvements
        regime_improvements = {}
        for regime in ['bull_market', 'bear_market', 'sideways_market', 'recovery_market']:
            if regime in enhanced_results['regime_metrics']:
                enhanced_metrics = enhanced_results['regime_metrics'][regime]
                
                # Calculate baseline metrics for this regime
                regime_indices = [i for i, r in enumerate([r['true_regime'] for r in baseline_results['results']]) if r == regime]
                if regime_indices:
                    baseline_accuracies = [baseline_results['accuracies'][i] for i in regime_indices]
                    baseline_confidences = [baseline_results['confidences'][i] for i in regime_indices]
                    
                    baseline_accuracy = np.mean(baseline_accuracies)
                    baseline_confidence = np.mean(baseline_confidences)
                    
                    accuracy_improvement = ((enhanced_metrics['accuracy'] - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
                    confidence_improvement = ((enhanced_metrics['confidence'] - baseline_confidence) / baseline_confidence * 100) if baseline_confidence > 0 else 0
                    
                    regime_improvements[regime] = {
                        'accuracy_improvement': accuracy_improvement,
                        'confidence_improvement': confidence_improvement,
                        'baseline_accuracy': baseline_accuracy,
                        'enhanced_accuracy': enhanced_metrics['accuracy'],
                        'baseline_confidence': baseline_confidence,
                        'enhanced_confidence': enhanced_metrics['confidence']
                    }
        
        analysis['regime_specific'] = regime_improvements
        
        # Overall assessment
        improvements = [accuracy_improvement, confidence_improvement, stability_improvement]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        total_improvements = len(improvements)
        
        analysis['assessment'] = {
            'success_rate': (positive_improvements / total_improvements) * 100,
            'average_improvement': np.mean(improvements),
            'positive_improvements': positive_improvements,
            'total_improvements': total_improvements
        }
        
        return analysis
    
    def _generate_regime_detection_report(self, baseline_results: Dict, enhanced_results: Dict, 
                                       improvement_analysis: Dict) -> str:
        """Generate regime detection enhancement report"""
        
        report = []
        report.append("EPIC 1.5: REGIME DETECTION ENHANCEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        report.append("🎯 FOCUS: Increase regime confidence from 84.54% to 90%+")
        report.append("")
        
        # Overall Performance Comparison
        report.append("📊 OVERALL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        overall = improvement_analysis['overall']
        report.append(f"Baseline Accuracy: {overall['baseline_accuracy']:.4f}")
        report.append(f"Enhanced Accuracy: {overall['enhanced_accuracy']:.4f}")
        report.append(f"Accuracy Improvement: {overall['accuracy_improvement']:.1f}%")
        report.append("")
        report.append(f"Baseline Confidence: {overall['baseline_confidence']:.4f}")
        report.append(f"Enhanced Confidence: {overall['enhanced_confidence']:.4f}")
        report.append(f"Confidence Improvement: {overall['confidence_improvement']:.1f}%")
        report.append("")
        
        # Regime-Specific Improvements
        report.append("🎯 REGIME-SPECIFIC IMPROVEMENTS")
        report.append("-" * 40)
        for regime, metrics in improvement_analysis['regime_specific'].items():
            report.append(f"{regime.upper()}:")
            report.append(f"  Accuracy: {metrics['baseline_accuracy']:.4f} → {metrics['enhanced_accuracy']:.4f} ({metrics['accuracy_improvement']:.1f}%)")
            report.append(f"  Confidence: {metrics['baseline_confidence']:.4f} → {metrics['enhanced_confidence']:.4f} ({metrics['confidence_improvement']:.1f}%)")
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
            report.append("Status: 🎉 EXCELLENT - Enhanced regime detection shows significant improvements!")
        elif assessment['success_rate'] >= 50:
            report.append("Status: ✅ GOOD - Enhanced regime detection shows notable improvements!")
        elif assessment['success_rate'] >= 25:
            report.append("Status: ⚠️ PARTIAL - Enhanced regime detection shows some improvements")
        else:
            report.append("Status: ❌ POOR - Enhanced regime detection does not show improvements")
        
        report.append("")
        report.append("Key improvements:")
        report.append("1. 🎯 Advanced ensemble methods for better accuracy")
        report.append("2. 🔧 Temporal consistency checks for regime stability")
        report.append("3. 📊 Advanced feature engineering for better detection")
        report.append("4. ⚙️ Regime transition probabilities for context")
        report.append("5. 📈 Adaptive confidence thresholds for optimization")
        
        report.append("\n" + "=" * 60)
        report.append("End of Regime Detection Enhancement Report")
        
        return "\n".join(report)
    
    def _create_regime_detection_visualizations(self, baseline_results: Dict, enhanced_results: Dict, 
                                              improvement_analysis: Dict):
        """Create regime detection enhancement visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Accuracy Comparison
        plt.subplot(3, 3, 1)
        systems = ['Baseline', 'Enhanced']
        accuracies = [improvement_analysis['overall']['baseline_accuracy'], 
                     improvement_analysis['overall']['enhanced_accuracy']]
        
        bars = plt.bar(systems, accuracies, color=['blue', 'green'], alpha=0.7)
        plt.title('Regime Detection Accuracy Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        
        # Add value labels
        for bar, value in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Confidence Comparison
        plt.subplot(3, 3, 2)
        confidences = [improvement_analysis['overall']['baseline_confidence'], 
                      improvement_analysis['overall']['enhanced_confidence']]
        
        bars = plt.bar(systems, confidences, color=['blue', 'green'], alpha=0.7)
        plt.title('Regime Detection Confidence Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence')
        
        # Add value labels
        for bar, value in zip(bars, confidences):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Improvement Percentages
        plt.subplot(3, 3, 3)
        improvement_metrics = ['Accuracy', 'Confidence', 'Stability']
        improvement_values = [
            improvement_analysis['overall']['accuracy_improvement'],
            improvement_analysis['overall']['confidence_improvement'],
            improvement_analysis['overall']['stability_improvement']
        ]
        
        bars = plt.bar(improvement_metrics, improvement_values, color=['green', 'blue', 'orange'], alpha=0.7)
        plt.title('Improvement Percentages\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Regime-Specific Accuracy Improvements
        plt.subplot(3, 3, 4)
        regimes = list(improvement_analysis['regime_specific'].keys())
        regime_accuracy_improvements = [improvement_analysis['regime_specific'][regime]['accuracy_improvement'] 
                                       for regime in regimes]
        
        bars = plt.bar(regimes, regime_accuracy_improvements, color=['green', 'red', 'blue', 'orange'], alpha=0.7)
        plt.title('Regime-Specific Accuracy Improvements\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, regime_accuracy_improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 5. Regime-Specific Confidence Improvements
        plt.subplot(3, 3, 5)
        regime_confidence_improvements = [improvement_analysis['regime_specific'][regime]['confidence_improvement'] 
                                       for regime in regimes]
        
        bars = plt.bar(regimes, regime_confidence_improvements, color=['green', 'red', 'blue', 'orange'], alpha=0.7)
        plt.title('Regime-Specific Confidence Improvements\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, regime_confidence_improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Confidence Over Time - Baseline
        plt.subplot(3, 3, 6)
        periods = range(len(baseline_results['confidences']))
        plt.plot(periods, baseline_results['confidences'], label='Baseline', alpha=0.7, color='blue')
        plt.title('Baseline Confidence Over Time\n(Regime Detection)', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Confidence Over Time - Enhanced
        plt.subplot(3, 3, 7)
        periods = range(len(enhanced_results['confidences']))
        plt.plot(periods, enhanced_results['confidences'], label='Enhanced', alpha=0.7, color='green')
        plt.title('Enhanced Confidence Over Time\n(Regime Detection)', fontsize=14, fontweight='bold')
        plt.ylabel('Confidence')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Accuracy Over Time - Baseline
        plt.subplot(3, 3, 8)
        periods = range(len(baseline_results['accuracies']))
        plt.plot(periods, baseline_results['accuracies'], label='Baseline', alpha=0.7, color='blue')
        plt.title('Baseline Accuracy Over Time\n(Regime Detection)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Accuracy Over Time - Enhanced
        plt.subplot(3, 3, 9)
        periods = range(len(enhanced_results['accuracies']))
        plt.plot(periods, enhanced_results['accuracies'], label='Enhanced', alpha=0.7, color='green')
        plt.title('Enhanced Accuracy Over Time\n(Regime Detection)', fontsize=14, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_regime_detection_enhancement_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Regime detection enhancement visualization saved as: {filename}")
        
        plt.show()

def main():
    """Run regime detection enhancement experiment"""
    
    print("🎯 Starting EPIC 1.5 Regime Detection Enhancement Experiment")
    print("=" * 70)
    print("🎯 Focus: Increase regime confidence from 84.54% to 90%+")
    print("")
    
    # Run regime detection enhancement experiment
    experiment = RegimeDetectionEnhancementExperiment()
    
    try:
        results = experiment.run_regime_detection_enhancement_experiment(test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_regime_detection_enhancement_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Regime detection enhancement experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("REGIME DETECTION ENHANCEMENT REPORT")
        print("=" * 70)
        print(results['report'])
        
    except Exception as e:
        print(f"❌ Error running regime detection enhancement experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
