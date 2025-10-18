#!/usr/bin/env python3
"""
EPIC 1.5: Performance Comparison Experiment

This experiment validates the expected RMSE improvements between:
- Basic Kalman Filter
- Enhanced Kalman Filter  
- Regime-Integrated Kalman Filter

Expected improvements:
- Enhanced vs Basic: 5-15% better RMSE
- Regime-Integrated vs Basic: 10-25% better RMSE
- Regime-Integrated vs Enhanced: 3-8% better RMSE
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

from algorithms.enhanced_kalman_filter import EnhancedKalmanFilter
from algorithms.regime_integrated_kalman import RegimeIntegratedKalmanFilter
from algorithms.kalman_filter import kalman_filter, KalmanParams, create_kalman_params, get_portfolio_state
from algorithms.regime_detection_bnn import MarketRegimeDetectionBNN

class BasicKalmanFilter:
    """Wrapper class for basic Kalman filter functions"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.params = create_kalman_params()
        self.state_log = []
        self.prediction_log = []
    
    def predict(self, horizon: int = 1) -> np.ndarray:
        """Make prediction using basic Kalman filter"""
        try:
            # Get current state
            roi, risk = get_portfolio_state(self.params)
            prediction = np.array([roi, risk])
            self.prediction_log.append(prediction)
            return prediction
        except Exception as e:
            print(f"Basic Kalman prediction error: {e}")
            return np.array([0.0, 0.0])
    
    def update(self, observation: np.ndarray) -> None:
        """Update Kalman filter with new observation"""
        try:
            kalman_filter(self.params, observation)
            # Log state after update
            roi, risk = get_portfolio_state(self.params)
            self.state_log.append(np.array([roi, risk, 0.0, 0.0]))
        except Exception as e:
            print(f"Basic Kalman update error: {e}")
    
    def get_current_state(self) -> np.ndarray:
        """Get current state"""
        try:
            roi, risk = get_portfolio_state(self.params)
            return np.array([roi, risk, 0.0, 0.0])
        except:
            return np.array([0.0, 0.0, 0.0, 0.0])

class PerformanceComparisonExperiment:
    """Comprehensive performance comparison experiment"""
    
    def __init__(self):
        self.results = {}
        self.predictions = {}
        self.errors = {}
        self.uncertainties = {}
        
    def run_performance_comparison(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run comprehensive performance comparison"""
        
        print("🚀 Starting EPIC 1.5 Performance Comparison Experiment")
        print("=" * 70)
        
        # Initialize models
        models = self._initialize_models()
        
        # Run prediction experiment
        print("\n📊 Running Prediction Experiment...")
        prediction_results = self._run_prediction_experiment(models, data, test_periods)
        
        # Calculate performance metrics
        print("\n📈 Calculating Performance Metrics...")
        performance_metrics = self._calculate_performance_metrics(prediction_results, data, test_periods)
        
        # Generate comparison report
        print("\n📋 Generating Comparison Report...")
        comparison_report = self._generate_comparison_report(performance_metrics)
        
        # Create visualizations
        print("\n🎨 Creating Visualizations...")
        self._create_visualizations(prediction_results, performance_metrics)
        
        return {
            'prediction_results': prediction_results,
            'performance_metrics': performance_metrics,
            'comparison_report': comparison_report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _initialize_models(self) -> Dict:
        """Initialize all models for comparison"""
        
        models = {}
        
        # Basic Kalman Filter
        print("Initializing Basic Kalman Filter...")
        models['basic'] = {
            'model': BasicKalmanFilter(window_size=10),
            'name': 'Basic Kalman Filter',
            'color': 'blue'
        }
        
        # Enhanced Kalman Filter
        print("Initializing Enhanced Kalman Filter...")
        models['enhanced'] = {
            'model': EnhancedKalmanFilter(state_dim=4, observation_dim=2),
            'name': 'Enhanced Kalman Filter',
            'color': 'green'
        }
        
        # Regime-Integrated Kalman Filter
        print("Initializing Regime-Integrated Kalman Filter...")
        regime_detector = MarketRegimeDetectionBNN()
        models['regime_integrated'] = {
            'model': RegimeIntegratedKalmanFilter(regime_detector),
            'name': 'Regime-Integrated Kalman Filter',
            'color': 'red'
        }
        
        return models
    
    def _run_prediction_experiment(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Run prediction experiment for all models"""
        
        prediction_results = {}
        
        for model_name, model_info in models.items():
            print(f"  🔄 Running {model_info['name']}...")
            
            model = model_info['model']
            predictions = []
            uncertainties = []
            confidences = []
            states = []
            times = []
            
            # Run prediction loop
            for i in range(min(test_periods, len(data))):
                start_time = time.time()
                
                # Make prediction
                try:
                    prediction = model.predict(horizon=1)
                    predictions.append(prediction)
                    
                    # Get uncertainty if available
                    if hasattr(model, 'get_uncertainty'):
                        uncertainty = model.get_uncertainty()
                        uncertainties.append(uncertainty)
                    else:
                        uncertainties.append(np.array([0.0, 0.0]))
                    
                    # Get confidence if available
                    if hasattr(model, 'get_confidence'):
                        confidence = model.get_confidence()
                        confidences.append(confidence)
                    else:
                        confidences.append(0.0)
                    
                    # Get current state
                    if hasattr(model, 'get_current_state'):
                        state = model.get_current_state()
                        states.append(state)
                    else:
                        states.append(np.array([0.0, 0.0, 0.0, 0.0]))
                    
                except Exception as e:
                    print(f"    ⚠️  Prediction error in {model_name}: {e}")
                    predictions.append(np.array([0.0, 0.0]))
                    uncertainties.append(np.array([0.0, 0.0]))
                    confidences.append(0.0)
                    states.append(np.array([0.0, 0.0, 0.0, 0.0]))
                
                # Update model
                try:
                    model.update(data[i])
                except Exception as e:
                    print(f"    ⚠️  Update error in {model_name}: {e}")
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            prediction_results[model_name] = {
                'predictions': np.array(predictions),
                'uncertainties': np.array(uncertainties),
                'confidences': np.array(confidences),
                'states': np.array(states),
                'times': np.array(times),
                'model_info': model_info
            }
        
        return prediction_results
    
    def _calculate_performance_metrics(self, prediction_results: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        performance_metrics = {}
        
        for model_name, results in prediction_results.items():
            predictions = results['predictions']
            uncertainties = results['uncertainties']
            confidences = results['confidences']
            states = results['states']
            times = results['times']
            
            # Calculate prediction errors
            actual_values = data[:len(predictions)]
            errors = predictions - actual_values
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean(errors**2))
            rmse_roi = np.sqrt(np.mean(errors[:, 0]**2))
            rmse_risk = np.sqrt(np.mean(errors[:, 1]**2))
            
            # Calculate MAE
            mae = np.mean(np.abs(errors))
            mae_roi = np.mean(np.abs(errors[:, 0]))
            mae_risk = np.mean(np.abs(errors[:, 1]))
            
            # Calculate MAPE
            mape_roi = np.mean(np.abs(errors[:, 0] / (actual_values[:, 0] + 1e-8))) * 100
            mape_risk = np.mean(np.abs(errors[:, 1] / (actual_values[:, 1] + 1e-8))) * 100
            
            # Calculate uncertainty metrics
            mean_uncertainty = np.mean(uncertainties)
            uncertainty_std = np.std(uncertainties)
            mean_confidence = np.mean(confidences)
            
            # Calculate state metrics
            state_variance = np.var(states, axis=0)
            state_dimensions = states.shape[1]
            
            # Calculate timing metrics
            mean_time = np.mean(times)
            total_time = np.sum(times)
            
            # Calculate prediction stability
            prediction_std = np.std(predictions, axis=0)
            prediction_stability = 1.0 / (1.0 + np.mean(prediction_std))
            
            performance_metrics[model_name] = {
                'rmse': rmse,
                'rmse_roi': rmse_roi,
                'rmse_risk': rmse_risk,
                'mae': mae,
                'mae_roi': mae_roi,
                'mae_risk': mae_risk,
                'mape_roi': mape_roi,
                'mape_risk': mape_risk,
                'mean_uncertainty': mean_uncertainty,
                'uncertainty_std': uncertainty_std,
                'mean_confidence': mean_confidence,
                'state_variance': state_variance,
                'state_dimensions': state_dimensions,
                'mean_time': mean_time,
                'total_time': total_time,
                'prediction_stability': prediction_stability,
                'predictions': predictions,
                'errors': errors,
                'uncertainties': uncertainties,
                'confidences': confidences,
                'states': states
            }
        
        # Calculate relative performance
        basic_rmse = performance_metrics['basic']['rmse']
        
        for model_name, metrics in performance_metrics.items():
            if model_name != 'basic':
                rmse_improvement = ((basic_rmse - metrics['rmse']) / basic_rmse) * 100
                metrics['rmse_improvement_vs_basic'] = rmse_improvement
        
        # Calculate enhanced vs regime-integrated
        enhanced_rmse = performance_metrics['enhanced']['rmse']
        regime_rmse = performance_metrics['regime_integrated']['rmse']
        regime_improvement = ((enhanced_rmse - regime_rmse) / enhanced_rmse) * 100
        performance_metrics['regime_integrated']['rmse_improvement_vs_enhanced'] = regime_improvement
        
        return performance_metrics
    
    def _generate_comparison_report(self, performance_metrics: Dict) -> str:
        """Generate comprehensive comparison report"""
        
        report = []
        report.append("EPIC 1.5: Performance Comparison Report")
        report.append("=" * 50)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # Performance Summary
        report.append("📊 PERFORMANCE SUMMARY")
        report.append("-" * 30)
        
        for model_name, metrics in performance_metrics.items():
            model_info = {
                'basic': 'Basic Kalman Filter',
                'enhanced': 'Enhanced Kalman Filter',
                'regime_integrated': 'Regime-Integrated Kalman Filter'
            }[model_name]
            
            report.append(f"\n{model_info}:")
            report.append(f"  RMSE: {metrics['rmse']:.6f}")
            report.append(f"  RMSE (ROI): {metrics['rmse_roi']:.6f}")
            report.append(f"  RMSE (Risk): {metrics['rmse_risk']:.6f}")
            report.append(f"  MAE: {metrics['mae']:.6f}")
            report.append(f"  MAPE (ROI): {metrics['mape_roi']:.2f}%")
            report.append(f"  MAPE (Risk): {metrics['mape_risk']:.2f}%")
            
            if 'rmse_improvement_vs_basic' in metrics:
                report.append(f"  RMSE Improvement vs Basic: {metrics['rmse_improvement_vs_basic']:.2f}%")
            
            if 'rmse_improvement_vs_enhanced' in metrics:
                report.append(f"  RMSE Improvement vs Enhanced: {metrics['rmse_improvement_vs_enhanced']:.2f}%")
        
        # Relative Performance Analysis
        report.append("\n📈 RELATIVE PERFORMANCE ANALYSIS")
        report.append("-" * 30)
        
        basic_rmse = performance_metrics['basic']['rmse']
        enhanced_rmse = performance_metrics['enhanced']['rmse']
        regime_rmse = performance_metrics['regime_integrated']['rmse']
        
        enhanced_improvement = ((basic_rmse - enhanced_rmse) / basic_rmse) * 100
        regime_improvement = ((basic_rmse - regime_rmse) / basic_rmse) * 100
        regime_vs_enhanced = ((enhanced_rmse - regime_rmse) / enhanced_rmse) * 100
        
        report.append(f"\nEnhanced vs Basic:")
        report.append(f"  RMSE Improvement: {enhanced_improvement:.2f}%")
        report.append(f"  Expected: 5-15%")
        report.append(f"  Status: {'✅ ACHIEVED' if enhanced_improvement >= 5 else '❌ NOT ACHIEVED'}")
        
        report.append(f"\nRegime-Integrated vs Basic:")
        report.append(f"  RMSE Improvement: {regime_improvement:.2f}%")
        report.append(f"  Expected: 10-25%")
        report.append(f"  Status: {'✅ ACHIEVED' if regime_improvement >= 10 else '❌ NOT ACHIEVED'}")
        
        report.append(f"\nRegime-Integrated vs Enhanced:")
        report.append(f"  RMSE Improvement: {regime_vs_enhanced:.2f}%")
        report.append(f"  Expected: 3-8%")
        report.append(f"  Status: {'✅ ACHIEVED' if regime_vs_enhanced >= 3 else '❌ NOT ACHIEVED'}")
        
        # Enhanced Features Analysis
        report.append("\n🔧 ENHANCED FEATURES ANALYSIS")
        report.append("-" * 30)
        
        for model_name, metrics in performance_metrics.items():
            model_info = {
                'basic': 'Basic Kalman Filter',
                'enhanced': 'Enhanced Kalman Filter',
                'regime_integrated': 'Regime-Integrated Kalman Filter'
            }[model_name]
            
            report.append(f"\n{model_info}:")
            report.append(f"  State Dimensions: {metrics['state_dimensions']}")
            report.append(f"  Mean Uncertainty: {metrics['mean_uncertainty']:.6f}")
            report.append(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
            report.append(f"  Prediction Stability: {metrics['prediction_stability']:.4f}")
            report.append(f"  Mean Time per Prediction: {metrics['mean_time']:.6f}s")
        
        # Success Metrics
        report.append("\n🎯 SUCCESS METRICS")
        report.append("-" * 30)
        
        success_metrics = []
        
        # Check if enhanced model shows improvement
        if enhanced_improvement >= 5:
            success_metrics.append("✅ Enhanced model shows 5%+ RMSE improvement")
        else:
            success_metrics.append("❌ Enhanced model does not show 5%+ RMSE improvement")
        
        # Check if regime-integrated model shows improvement
        if regime_improvement >= 10:
            success_metrics.append("✅ Regime-integrated model shows 10%+ RMSE improvement")
        else:
            success_metrics.append("❌ Regime-integrated model does not show 10%+ RMSE improvement")
        
        # Check if regime-integrated beats enhanced
        if regime_vs_enhanced >= 3:
            success_metrics.append("✅ Regime-integrated model beats enhanced model")
        else:
            success_metrics.append("❌ Regime-integrated model does not beat enhanced model")
        
        # Check uncertainty quantification
        enhanced_uncertainty = performance_metrics['enhanced']['mean_uncertainty']
        regime_uncertainty = performance_metrics['regime_integrated']['mean_uncertainty']
        basic_uncertainty = performance_metrics['basic']['mean_uncertainty']
        
        if enhanced_uncertainty > basic_uncertainty:
            success_metrics.append("✅ Enhanced model provides uncertainty quantification")
        else:
            success_metrics.append("❌ Enhanced model uncertainty quantification not working")
        
        if regime_uncertainty > basic_uncertainty:
            success_metrics.append("✅ Regime-integrated model provides uncertainty quantification")
        else:
            success_metrics.append("❌ Regime-integrated model uncertainty quantification not working")
        
        for metric in success_metrics:
            report.append(f"  {metric}")
        
        # Overall Assessment
        report.append("\n🏆 OVERALL ASSESSMENT")
        report.append("-" * 30)
        
        total_success = sum(1 for metric in success_metrics if metric.startswith("✅"))
        total_metrics = len(success_metrics)
        success_rate = (total_success / total_metrics) * 100
        
        report.append(f"Success Rate: {success_rate:.1f}% ({total_success}/{total_metrics})")
        
        if success_rate >= 80:
            report.append("Status: 🎉 EXCELLENT - Most expected improvements achieved!")
        elif success_rate >= 60:
            report.append("Status: ✅ GOOD - Significant improvements achieved!")
        elif success_rate >= 40:
            report.append("Status: ⚠️  PARTIAL - Some improvements achieved")
        else:
            report.append("Status: ❌ POOR - Few improvements achieved")
        
        report.append("\n" + "=" * 50)
        report.append("End of Performance Comparison Report")
        
        return "\n".join(report)
    
    def _create_visualizations(self, prediction_results: Dict, performance_metrics: Dict):
        """Create comprehensive visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. RMSE Comparison
        plt.subplot(2, 3, 1)
        model_names = list(performance_metrics.keys())
        rmse_values = [performance_metrics[name]['rmse'] for name in model_names]
        colors = ['blue', 'green', 'red']
        
        bars = plt.bar(model_names, rmse_values, color=colors, alpha=0.7)
        plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Prediction Accuracy Over Time
        plt.subplot(2, 3, 2)
        for model_name, results in prediction_results.items():
            predictions = results['predictions']
            errors = np.sqrt(np.sum((predictions - predictions.mean(axis=0))**2, axis=1))
            plt.plot(errors, label=results['model_info']['name'], 
                    color=results['model_info']['color'], alpha=0.7)
        
        plt.title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Prediction Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Uncertainty Quantification
        plt.subplot(2, 3, 3)
        uncertainty_data = []
        model_labels = []
        
        for model_name, metrics in performance_metrics.items():
            uncertainty_data.append(metrics['mean_uncertainty'])
            model_labels.append(model_name.replace('_', ' ').title())
        
        bars = plt.bar(model_labels, uncertainty_data, color=colors, alpha=0.7)
        plt.title('Uncertainty Quantification', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Uncertainty')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, uncertainty_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. State Space Comparison
        plt.subplot(2, 3, 4)
        state_dims = [performance_metrics[name]['state_dimensions'] for name in model_names]
        
        bars = plt.bar(model_labels, state_dims, color=colors, alpha=0.7)
        plt.title('State Space Dimensions', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Dimensions')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, state_dims):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Performance Improvement
        plt.subplot(2, 3, 5)
        basic_rmse = performance_metrics['basic']['rmse']
        improvements = []
        
        for model_name in model_names:
            if model_name != 'basic':
                improvement = ((basic_rmse - performance_metrics[model_name]['rmse']) / basic_rmse) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)
        
        bars = plt.bar(model_labels, improvements, color=colors, alpha=0.7)
        plt.title('RMSE Improvement vs Basic Model', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Prediction Stability
        plt.subplot(2, 3, 6)
        stability_data = [performance_metrics[name]['prediction_stability'] for name in model_names]
        
        bars = plt.bar(model_labels, stability_data, color=colors, alpha=0.7)
        plt.title('Prediction Stability', fontsize=14, fontweight='bold')
        plt.ylabel('Stability Score')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, stability_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"epic1_5_performance_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved as: {filename}")
        
        plt.show()

def load_test_data() -> np.ndarray:
    """Load test data for performance comparison"""
    # Generate synthetic financial data for testing
    np.random.seed(42)  # For reproducibility
    
    # Generate 200 periods of ROI and risk data
    n_periods = 200
    
    # Generate ROI data with some trend and noise
    roi_trend = np.linspace(0.01, 0.05, n_periods)
    roi_noise = np.random.normal(0, 0.02, n_periods)
    roi_data = roi_trend + roi_noise
    
    # Generate risk data with some volatility
    risk_trend = np.linspace(0.1, 0.15, n_periods)
    risk_noise = np.random.normal(0, 0.01, n_periods)
    risk_data = risk_trend + risk_noise
    
    # Combine into observation matrix
    data = np.column_stack([roi_data, risk_data])
    
    return data

def main():
    """Run performance comparison experiment"""
    
    print("🚀 Starting EPIC 1.5 Performance Comparison Experiment")
    print("=" * 70)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run performance comparison
    experiment = PerformanceComparisonExperiment()
    
    try:
        results = experiment.run_performance_comparison(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_performance_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Performance comparison completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("PERFORMANCE COMPARISON REPORT")
        print("=" * 70)
        print(results['comparison_report'])
        
    except Exception as e:
        print(f"❌ Error running performance comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
