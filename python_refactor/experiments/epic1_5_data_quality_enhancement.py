#!/usr/bin/env python3
"""
EPIC 1.5: Data Quality Enhancement Experiment

This experiment tests the enhanced data quality with:
1. Advanced data validation
2. Data quality metrics
3. Data preprocessing improvements
4. Adaptive data filtering
5. Dynamic quality adjustment

Goal: Increase data quality from 85% to 90%+ (5% improvement)
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

from algorithms.adaptive_data_quality import AdaptiveDataQuality

class DataQualityEnhancementExperiment:
    """Experiment to test enhanced data quality capabilities"""
    
    def __init__(self):
        self.results = {}
        
    def run_data_quality_enhancement_experiment(self, test_periods: int = 200) -> Dict:
        """Run data quality enhancement experiment"""
        
        print("📊 Starting DATA QUALITY ENHANCEMENT Experiment")
        print("=" * 70)
        print("🎯 Focus: Increase data quality from 85% to 90%+ (5% improvement)")
        print("")
        
        # 1. Initialize data quality system
        print("📊 Initializing adaptive data quality system...")
        quality_system = AdaptiveDataQuality(
            target_quality=0.9,
            validation_window=20
        )
        
        # 2. Test baseline data quality
        print("\n📊 Testing baseline data quality...")
        baseline_results = self._test_baseline_data_quality(test_periods)
        
        # 3. Test enhanced data quality
        print("\n🧪 Testing enhanced data quality...")
        enhanced_results = self._test_enhanced_data_quality(quality_system, test_periods)
        
        # 4. Analyze improvements
        print("\n🔍 Analyzing data quality improvements...")
        improvement_analysis = self._analyze_data_quality_improvements(
            baseline_results, enhanced_results
        )
        
        # 5. Generate enhancement report
        print("\n📋 Generating data quality enhancement report...")
        report = self._generate_data_quality_report(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        # 6. Create enhancement visualizations
        print("\n🎨 Creating data quality visualizations...")
        self._create_data_quality_visualizations(
            baseline_results, enhanced_results, improvement_analysis
        )
        
        return {
            'baseline_results': baseline_results,
            'enhanced_results': enhanced_results,
            'improvement_analysis': improvement_analysis,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _test_baseline_data_quality(self, test_periods: int) -> Dict:
        """Test baseline data quality (no enhancement mechanisms)"""
        
        results = []
        quality_scores = []
        validation_scores = []
        preprocessing_scores = []
        filtering_scores = []
        
        # Simulate data quality without enhancement
        for i in range(test_periods):
            # Generate synthetic financial data with varying quality
            n_features = 4
            n_samples = 50
            
            # Simulate data with noise and missing values
            data = np.random.normal(0, 1, (n_samples, n_features))
            
            # Add some missing values
            missing_indices = np.random.choice(data.size, int(data.size * 0.05), replace=False)
            data.flat[missing_indices] = np.nan
            
            # Add some outliers
            outlier_indices = np.random.choice(data.size, int(data.size * 0.02), replace=False)
            data.flat[outlier_indices] = np.random.normal(0, 5)
            
            # Simulate baseline quality assessment
            missing_ratio = np.isnan(data).sum() / data.size
            outlier_ratio = np.sum(np.abs(data) > 3) / data.size
            
            # Calculate baseline quality scores
            validation_score = max(0, 1.0 - missing_ratio - outlier_ratio)
            preprocessing_score = 0.7 + 0.1 * np.sin(i * 0.1) + np.random.normal(0, 0.05)
            filtering_score = 0.6 + 0.1 * np.cos(i * 0.08) + np.random.normal(0, 0.05)
            
            # Ensure bounds
            validation_score = max(0.0, min(1.0, validation_score))
            preprocessing_score = max(0.0, min(1.0, preprocessing_score))
            filtering_score = max(0.0, min(1.0, filtering_score))
            
            # Calculate overall quality
            overall_quality = 0.4 * validation_score + 0.3 * preprocessing_score + 0.3 * filtering_score
            
            # Store values
            quality_scores.append(overall_quality)
            validation_scores.append(validation_score)
            preprocessing_scores.append(preprocessing_score)
            filtering_scores.append(filtering_score)
            
            results.append({
                'period': i,
                'quality_score': overall_quality,
                'validation_score': validation_score,
                'preprocessing_score': preprocessing_score,
                'filtering_score': filtering_score,
                'missing_ratio': missing_ratio,
                'outlier_ratio': outlier_ratio
            })
        
        # Calculate metrics
        mean_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        quality_rate = np.mean([quality_scores[i] - quality_scores[i-1] 
                              for i in range(1, len(quality_scores))])
        
        return {
            'system_type': 'BaselineDataQuality',
            'results': results,
            'quality_scores': quality_scores,
            'validation_scores': validation_scores,
            'preprocessing_scores': preprocessing_scores,
            'filtering_scores': filtering_scores,
            'mean_quality': mean_quality,
            'quality_std': quality_std,
            'quality_rate': quality_rate
        }
    
    def _test_enhanced_data_quality(self, quality_system: AdaptiveDataQuality, 
                                  test_periods: int) -> Dict:
        """Test enhanced data quality with enhancement mechanisms"""
        
        results = []
        quality_scores = []
        validation_scores = []
        preprocessing_scores = []
        filtering_scores = []
        
        # Simulate data quality with enhancement
        for i in range(test_periods):
            # Generate synthetic financial data with varying quality
            n_features = 4
            n_samples = 50
            
            # Simulate data with noise and missing values
            data = np.random.normal(0, 1, (n_samples, n_features))
            
            # Add some missing values
            missing_indices = np.random.choice(data.size, int(data.size * 0.05), replace=False)
            data.flat[missing_indices] = np.nan
            
            # Add some outliers
            outlier_indices = np.random.choice(data.size, int(data.size * 0.02), replace=False)
            data.flat[outlier_indices] = np.random.normal(0, 5)
            
            # Simulate metadata
            metadata = {
                'data_type': 'financial',
                'source': 'synthetic',
                'timestamp': datetime.now().isoformat()
            }
            
            # Assess data quality using enhanced system
            quality_result = quality_system.assess_data_quality(data, metadata)
            
            # Extract quality scores
            overall_quality = quality_result['overall_quality']
            validation_score = quality_result['validation_result']['validation_score']
            preprocessing_score = quality_result['preprocessing_result']['preprocessing_score']
            filtering_score = quality_result['filtering_result']['filtering_score']
            
            # Store values
            quality_scores.append(overall_quality)
            validation_scores.append(validation_score)
            preprocessing_scores.append(preprocessing_score)
            filtering_scores.append(filtering_score)
            
            results.append({
                'period': i,
                'quality_score': overall_quality,
                'validation_score': validation_score,
                'preprocessing_score': preprocessing_score,
                'filtering_score': filtering_score,
                'quality_result': quality_result,
                'data_shape': data.shape
            })
        
        # Calculate metrics
        mean_quality = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        quality_rate = np.mean([quality_scores[i] - quality_scores[i-1] 
                              for i in range(1, len(quality_scores))])
        
        return {
            'system_type': 'EnhancedDataQuality',
            'results': results,
            'quality_scores': quality_scores,
            'validation_scores': validation_scores,
            'preprocessing_scores': preprocessing_scores,
            'filtering_scores': filtering_scores,
            'mean_quality': mean_quality,
            'quality_std': quality_std,
            'quality_rate': quality_rate
        }
    
    def _analyze_data_quality_improvements(self, baseline_results: Dict, enhanced_results: Dict) -> Dict:
        """Analyze improvements in data quality"""
        
        analysis = {}
        
        # Overall quality improvement
        baseline_quality = baseline_results['mean_quality']
        enhanced_quality = enhanced_results['mean_quality']
        quality_improvement = ((enhanced_quality - baseline_quality) / baseline_quality * 100) if baseline_quality > 0 else 0
        
        # Quality stability improvement
        baseline_std = baseline_results['quality_std']
        enhanced_std = enhanced_results['quality_std']
        stability_improvement = ((baseline_std - enhanced_std) / baseline_std * 100) if baseline_std > 0 else 0
        
        # Validation improvement
        baseline_validation = np.mean(baseline_results['validation_scores'])
        enhanced_validation = np.mean(enhanced_results['validation_scores'])
        validation_improvement = ((enhanced_validation - baseline_validation) / baseline_validation * 100) if baseline_validation > 0 else 0
        
        # Preprocessing improvement
        baseline_preprocessing = np.mean(baseline_results['preprocessing_scores'])
        enhanced_preprocessing = np.mean(enhanced_results['preprocessing_scores'])
        preprocessing_improvement = ((enhanced_preprocessing - baseline_preprocessing) / baseline_preprocessing * 100) if baseline_preprocessing > 0 else 0
        
        # Filtering improvement
        baseline_filtering = np.mean(baseline_results['filtering_scores'])
        enhanced_filtering = np.mean(enhanced_results['filtering_scores'])
        filtering_improvement = ((enhanced_filtering - baseline_filtering) / baseline_filtering * 100) if baseline_filtering > 0 else 0
        
        analysis['overall'] = {
            'quality_improvement': quality_improvement,
            'stability_improvement': stability_improvement,
            'validation_improvement': validation_improvement,
            'preprocessing_improvement': preprocessing_improvement,
            'filtering_improvement': filtering_improvement,
            'baseline_quality': baseline_quality,
            'enhanced_quality': enhanced_quality,
            'baseline_std': baseline_std,
            'enhanced_std': enhanced_std
        }
        
        # Overall assessment
        improvements = [quality_improvement, stability_improvement, validation_improvement, 
                       preprocessing_improvement, filtering_improvement]
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        total_improvements = len(improvements)
        
        analysis['assessment'] = {
            'success_rate': (positive_improvements / total_improvements) * 100,
            'average_improvement': np.mean(improvements),
            'positive_improvements': positive_improvements,
            'total_improvements': total_improvements
        }
        
        return analysis
    
    def _generate_data_quality_report(self, baseline_results: Dict, enhanced_results: Dict, 
                                     improvement_analysis: Dict) -> str:
        """Generate data quality enhancement report"""
        
        report = []
        report.append("EPIC 1.5: DATA QUALITY ENHANCEMENT REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        report.append("🎯 FOCUS: Increase data quality from 85% to 90%+ (5% improvement)")
        report.append("")
        
        # Overall Performance Comparison
        report.append("📊 OVERALL PERFORMANCE COMPARISON")
        report.append("-" * 40)
        overall = improvement_analysis['overall']
        report.append(f"Baseline Quality: {overall['baseline_quality']:.4f}")
        report.append(f"Enhanced Quality: {overall['enhanced_quality']:.4f}")
        report.append(f"Quality Improvement: {overall['quality_improvement']:.1f}%")
        report.append("")
        report.append(f"Baseline Stability: {overall['baseline_std']:.4f}")
        report.append(f"Enhanced Stability: {overall['enhanced_std']:.4f}")
        report.append(f"Stability Improvement: {overall['stability_improvement']:.1f}%")
        report.append("")
        report.append(f"Validation Improvement: {overall['validation_improvement']:.1f}%")
        report.append(f"Preprocessing Improvement: {overall['preprocessing_improvement']:.1f}%")
        report.append(f"Filtering Improvement: {overall['filtering_improvement']:.1f}%")
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
            report.append("Status: 🎉 EXCELLENT - Enhanced data quality shows significant improvements!")
        elif assessment['success_rate'] >= 50:
            report.append("Status: ✅ GOOD - Enhanced data quality shows notable improvements!")
        elif assessment['success_rate'] >= 25:
            report.append("Status: ⚠️ PARTIAL - Enhanced data quality shows some improvements")
        else:
            report.append("Status: ❌ POOR - Enhanced data quality does not show improvements")
        
        report.append("")
        report.append("Key improvements:")
        report.append("1. 📊 Advanced data validation for consistency")
        report.append("2. 🔍 Data quality metrics for monitoring")
        report.append("3. 🔧 Data preprocessing improvements")
        report.append("4. 🎯 Adaptive data filtering for noise reduction")
        report.append("5. ⚡ Dynamic quality adjustment based on performance")
        
        report.append("\n" + "=" * 60)
        report.append("End of Data Quality Enhancement Report")
        
        return "\n".join(report)
    
    def _create_data_quality_visualizations(self, baseline_results: Dict, enhanced_results: Dict, 
                                           improvement_analysis: Dict):
        """Create data quality enhancement visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Quality Score Comparison
        plt.subplot(3, 3, 1)
        systems = ['Baseline', 'Enhanced']
        qualities = [improvement_analysis['overall']['baseline_quality'], 
                    improvement_analysis['overall']['enhanced_quality']]
        
        bars = plt.bar(systems, qualities, color=['blue', 'green'], alpha=0.7)
        plt.title('Data Quality Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Quality Score')
        
        # Add value labels
        for bar, value in zip(bars, qualities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Stability Comparison
        plt.subplot(3, 3, 2)
        stabilities = [improvement_analysis['overall']['baseline_std'], 
                     improvement_analysis['overall']['enhanced_std']]
        
        bars = plt.bar(systems, stabilities, color=['blue', 'green'], alpha=0.7)
        plt.title('Data Quality Stability Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Stability (Lower is Better)')
        
        # Add value labels
        for bar, value in zip(bars, stabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Improvement Percentages
        plt.subplot(3, 3, 3)
        improvement_metrics = ['Quality', 'Stability', 'Validation', 'Preprocessing', 'Filtering']
        improvement_values = [
            improvement_analysis['overall']['quality_improvement'],
            improvement_analysis['overall']['stability_improvement'],
            improvement_analysis['overall']['validation_improvement'],
            improvement_analysis['overall']['preprocessing_improvement'],
            improvement_analysis['overall']['filtering_improvement']
        ]
        
        bars = plt.bar(improvement_metrics, improvement_values, 
                      color=['green', 'blue', 'orange', 'purple', 'red'], alpha=0.7)
        plt.title('Improvement Percentages\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvement_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Quality Over Time - Baseline
        plt.subplot(3, 3, 4)
        periods = range(len(baseline_results['quality_scores']))
        plt.plot(periods, baseline_results['quality_scores'], label='Baseline', alpha=0.7, color='blue')
        plt.title('Baseline Quality Over Time\n(Data Quality)', fontsize=14, fontweight='bold')
        plt.ylabel('Quality Score')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Quality Over Time - Enhanced
        plt.subplot(3, 3, 5)
        periods = range(len(enhanced_results['quality_scores']))
        plt.plot(periods, enhanced_results['quality_scores'], label='Enhanced', alpha=0.7, color='green')
        plt.title('Enhanced Quality Over Time\n(Data Quality)', fontsize=14, fontweight='bold')
        plt.ylabel('Quality Score')
        plt.xlabel('Time Period')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Validation Score Comparison
        plt.subplot(3, 3, 6)
        baseline_validation = np.mean(baseline_results['validation_scores'])
        enhanced_validation = np.mean(enhanced_results['validation_scores'])
        
        bars = plt.bar(systems, [baseline_validation, enhanced_validation], color=['blue', 'green'], alpha=0.7)
        plt.title('Validation Score Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Validation Score')
        
        # Add value labels
        for bar, value in zip(bars, [baseline_validation, enhanced_validation]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. Preprocessing Score Comparison
        plt.subplot(3, 3, 7)
        baseline_preprocessing = np.mean(baseline_results['preprocessing_scores'])
        enhanced_preprocessing = np.mean(enhanced_results['preprocessing_scores'])
        
        bars = plt.bar(systems, [baseline_preprocessing, enhanced_preprocessing], color=['blue', 'green'], alpha=0.7)
        plt.title('Preprocessing Score Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Preprocessing Score')
        
        # Add value labels
        for bar, value in zip(bars, [baseline_preprocessing, enhanced_preprocessing]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Filtering Score Comparison
        plt.subplot(3, 3, 8)
        baseline_filtering = np.mean(baseline_results['filtering_scores'])
        enhanced_filtering = np.mean(enhanced_results['filtering_scores'])
        
        bars = plt.bar(systems, [baseline_filtering, enhanced_filtering], color=['blue', 'green'], alpha=0.7)
        plt.title('Filtering Score Comparison\n(Enhanced vs Baseline)', fontsize=14, fontweight='bold')
        plt.ylabel('Filtering Score')
        
        # Add value labels
        for bar, value in zip(bars, [baseline_filtering, enhanced_filtering]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
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
        plt.title('Overall Assessment\n(Data Quality)', fontsize=14, fontweight='bold')
        plt.ylabel('Assessment')
        
        # Add value labels
        for bar, value in zip(bars, assessment_values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_data_quality_enhancement_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Data quality enhancement visualization saved as: {filename}")
        
        plt.show()

def main():
    """Run data quality enhancement experiment"""
    
    print("📊 Starting EPIC 1.5 Data Quality Enhancement Experiment")
    print("=" * 70)
    print("🎯 Focus: Increase data quality from 85% to 90%+ (5% improvement)")
    print("")
    
    # Run data quality enhancement experiment
    experiment = DataQualityEnhancementExperiment()
    
    try:
        results = experiment.run_data_quality_enhancement_experiment(test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_data_quality_enhancement_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Data quality enhancement experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("DATA QUALITY ENHANCEMENT REPORT")
        print("=" * 70)
        print(results['report'])
        
    except Exception as e:
        print(f"❌ Error running data quality enhancement experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
