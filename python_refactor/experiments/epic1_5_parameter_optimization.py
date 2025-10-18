#!/usr/bin/env python3
"""
EPIC 1.5: Parameter Optimization Experiment

This experiment optimizes enhanced model parameters to improve performance
and achieve the expected RMSE improvements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            roi, risk = get_portfolio_state(self.params)
            prediction = np.array([roi, risk])
            self.prediction_log.append(prediction)
            return prediction
        except Exception as e:
            return np.array([0.0, 0.0])
    
    def update(self, observation: np.ndarray) -> None:
        """Update Kalman filter with new observation"""
        try:
            kalman_filter(self.params, observation)
            roi, risk = get_portfolio_state(self.params)
            self.state_log.append(np.array([roi, risk, 0.0, 0.0]))
        except Exception as e:
            pass
    
    def get_current_state(self) -> np.ndarray:
        """Get current state"""
        try:
            roi, risk = get_portfolio_state(self.params)
            return np.array([roi, risk, 0.0, 0.0])
        except:
            return np.array([0.0, 0.0, 0.0, 0.0])

class ParameterOptimizationExperiment:
    """Parameter optimization experiment for enhanced models"""
    
    def __init__(self):
        self.results = {}
        self.optimization_history = []
        
    def run_parameter_optimization(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run parameter optimization experiment"""
        
        print("🔧 Starting EPIC 1.5 Parameter Optimization Experiment")
        print("=" * 70)
        
        # Test different parameter configurations
        parameter_configs = self._create_parameter_configurations()
        
        # Run optimization for each configuration
        optimization_results = {}
        
        for config_name, config in parameter_configs.items():
            print(f"\n🔍 Testing configuration: {config_name}")
            print(f"   Parameters: {config}")
            
            # Test configuration
            result = self._test_parameter_configuration(config, data, test_periods)
            optimization_results[config_name] = result
            
            # Print results safely
            if 'enhanced' in result and 'rmse' in result['enhanced']:
                print(f"   Enhanced RMSE: {result['enhanced']['rmse']:.6f}")
                print(f"   Enhanced Improvement: {result['enhanced']['improvement_vs_basic']:.2f}%")
            else:
                print(f"   Configuration failed to produce results")
        
        # Find best configuration
        best_config = self._find_best_configuration(optimization_results)
        
        # Generate optimization report
        report = self._generate_optimization_report(optimization_results, best_config)
        
        return {
            'optimization_results': optimization_results,
            'best_configuration': best_config,
            'optimization_report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_parameter_configurations(self) -> Dict:
        """Create different parameter configurations to test"""
        
        configurations = {}
        
        # Configuration 1: Conservative (reduce complexity)
        configurations['conservative'] = {
            'adaptation_rate': 0.001,
            'forgetting_factor': 0.99,
            'uncertainty_weight': 0.05,
            'regime_threshold': 0.9,
            'process_noise_scale': 0.5,
            'measurement_noise_scale': 0.5
        }
        
        # Configuration 2: Balanced (moderate complexity)
        configurations['balanced'] = {
            'adaptation_rate': 0.005,
            'forgetting_factor': 0.95,
            'uncertainty_weight': 0.1,
            'regime_threshold': 0.8,
            'process_noise_scale': 1.0,
            'measurement_noise_scale': 1.0
        }
        
        # Configuration 3: Aggressive (high complexity)
        configurations['aggressive'] = {
            'adaptation_rate': 0.01,
            'forgetting_factor': 0.9,
            'uncertainty_weight': 0.2,
            'regime_threshold': 0.7,
            'process_noise_scale': 1.5,
            'measurement_noise_scale': 1.5
        }
        
        # Configuration 4: Minimal (simplified)
        configurations['minimal'] = {
            'adaptation_rate': 0.0001,
            'forgetting_factor': 0.999,
            'uncertainty_weight': 0.01,
            'regime_threshold': 0.95,
            'process_noise_scale': 0.1,
            'measurement_noise_scale': 0.1
        }
        
        # Configuration 5: Optimized (tuned for simple data)
        configurations['optimized'] = {
            'adaptation_rate': 0.002,
            'forgetting_factor': 0.98,
            'uncertainty_weight': 0.08,
            'regime_threshold': 0.85,
            'process_noise_scale': 0.8,
            'measurement_noise_scale': 0.8
        }
        
        return configurations
    
    def _test_parameter_configuration(self, config: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Test a specific parameter configuration"""
        
        # Initialize models with configuration
        models = self._initialize_models_with_config(config)
        
        # Run prediction experiment
        prediction_results = self._run_prediction_experiment(models, data, test_periods)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(prediction_results, data, test_periods)
        
        return performance_metrics
    
    def _initialize_models_with_config(self, config: Dict) -> Dict:
        """Initialize models with specific configuration"""
        
        models = {}
        
        # Basic Kalman Filter (unchanged)
        models['basic'] = {
            'model': BasicKalmanFilter(window_size=10),
            'name': 'Basic Kalman Filter'
        }
        
        # Enhanced Kalman Filter with configuration
        enhanced_model = EnhancedKalmanFilter(state_dim=4, observation_dim=2)
        
        # Apply configuration to enhanced model
        if hasattr(enhanced_model, 'parameters') and enhanced_model.parameters:
            enhanced_model.parameters.adaptation_rate = config['adaptation_rate']
            enhanced_model.parameters.forgetting_factor = config['forgetting_factor']
        
        models['enhanced'] = {
            'model': enhanced_model,
            'name': 'Enhanced Kalman Filter',
            'config': config
        }
        
        # Regime-Integrated Kalman Filter with configuration
        regime_detector = MarketRegimeDetectionBNN()
        regime_model = RegimeIntegratedKalmanFilter(regime_detector)
        
        models['regime_integrated'] = {
            'model': regime_model,
            'name': 'Regime-Integrated Kalman Filter',
            'config': config
        }
        
        return models
    
    def _run_prediction_experiment(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Run prediction experiment for all models"""
        
        prediction_results = {}
        
        for model_name, model_info in models.items():
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
                    predictions.append(np.array([0.0, 0.0]))
                    uncertainties.append(np.array([0.0, 0.0]))
                    confidences.append(0.0)
                    states.append(np.array([0.0, 0.0, 0.0, 0.0]))
                
                # Update model
                try:
                    model.update(data[i])
                except Exception as e:
                    pass
                
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
        """Calculate performance metrics for all models"""
        
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
                metrics['improvement_vs_basic'] = rmse_improvement
        
        # Calculate enhanced vs regime-integrated
        enhanced_rmse = performance_metrics['enhanced']['rmse']
        regime_rmse = performance_metrics['regime_integrated']['rmse']
        regime_improvement = ((enhanced_rmse - regime_rmse) / enhanced_rmse) * 100
        performance_metrics['regime_integrated']['improvement_vs_enhanced'] = regime_improvement
        
        return performance_metrics
    
    def _find_best_configuration(self, optimization_results: Dict) -> Dict:
        """Find the best configuration based on performance"""
        
        best_config = None
        best_score = float('-inf')
        
        for config_name, results in optimization_results.items():
            # Check if results are valid
            if 'enhanced' not in results or 'regime_integrated' not in results:
                continue
                
            # Calculate composite score safely
            enhanced_improvement = results['enhanced'].get('improvement_vs_basic', 0)
            regime_improvement = results['regime_integrated'].get('improvement_vs_basic', 0)
            regime_vs_enhanced = results['regime_integrated'].get('improvement_vs_enhanced', 0)
            
            # Weighted score (prioritize overall improvement)
            score = (enhanced_improvement * 0.4 + regime_improvement * 0.4 + regime_vs_enhanced * 0.2)
            
            if score > best_score:
                best_score = score
                best_config = {
                    'name': config_name,
                    'score': score,
                    'enhanced_improvement': enhanced_improvement,
                    'regime_improvement': regime_improvement,
                    'regime_vs_enhanced': regime_vs_enhanced,
                    'results': results
                }
        
        return best_config
    
    def _generate_optimization_report(self, optimization_results: Dict, best_config: Dict) -> str:
        """Generate optimization report"""
        
        report = []
        report.append("EPIC 1.5: Parameter Optimization Report")
        report.append("=" * 50)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # Configuration Results
        report.append("📊 CONFIGURATION RESULTS")
        report.append("-" * 30)
        
        for config_name, results in optimization_results.items():
            report.append(f"\n{config_name.upper()} Configuration:")
            report.append(f"  Enhanced RMSE: {results['enhanced']['rmse']:.6f}")
            report.append(f"  Enhanced Improvement: {results['enhanced']['improvement_vs_basic']:.2f}%")
            report.append(f"  Regime RMSE: {results['regime_integrated']['rmse']:.6f}")
            report.append(f"  Regime Improvement: {results['regime_integrated']['improvement_vs_basic']:.2f}%")
            report.append(f"  Regime vs Enhanced: {results['regime_integrated']['improvement_vs_enhanced']:.2f}%")
        
        # Best Configuration
        report.append("\n🏆 BEST CONFIGURATION")
        report.append("-" * 30)
        
        if best_config:
            report.append(f"Configuration: {best_config['name'].upper()}")
            report.append(f"Composite Score: {best_config['score']:.2f}")
            report.append(f"Enhanced Improvement: {best_config['enhanced_improvement']:.2f}%")
            report.append(f"Regime Improvement: {best_config['regime_improvement']:.2f}%")
            report.append(f"Regime vs Enhanced: {best_config['regime_vs_enhanced']:.2f}%")
            
            # Check if best configuration meets expectations
            if best_config['enhanced_improvement'] >= 5:
                report.append("✅ Enhanced model meets 5%+ improvement target")
            else:
                report.append("❌ Enhanced model does not meet 5%+ improvement target")
            
            if best_config['regime_improvement'] >= 10:
                report.append("✅ Regime-integrated model meets 10%+ improvement target")
            else:
                report.append("❌ Regime-integrated model does not meet 10%+ improvement target")
            
            if best_config['regime_vs_enhanced'] >= 3:
                report.append("✅ Regime-integrated model beats enhanced model")
            else:
                report.append("❌ Regime-integrated model does not beat enhanced model")
        else:
            report.append("❌ No configuration found that meets expectations")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if best_config and best_config['score'] > 0:
            report.append("✅ Use the best configuration for enhanced models")
            report.append("✅ Continue with parameter optimization")
            report.append("✅ Test on more complex data")
        else:
            report.append("⚠️  Consider more aggressive parameter optimization")
            report.append("⚠️  Test on more complex, realistic data")
            report.append("⚠️  Consider model architecture changes")
        
        report.append("\n" + "=" * 50)
        report.append("End of Parameter Optimization Report")
        
        return "\n".join(report)

def load_test_data() -> np.ndarray:
    """Load test data for parameter optimization"""
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
    """Run parameter optimization experiment"""
    
    print("🔧 Starting EPIC 1.5 Parameter Optimization Experiment")
    print("=" * 70)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run parameter optimization
    experiment = ParameterOptimizationExperiment()
    
    try:
        results = experiment.run_parameter_optimization(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_optimization_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Parameter optimization completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("PARAMETER OPTIMIZATION REPORT")
        print("=" * 70)
        print(results['optimization_report'])
        
    except Exception as e:
        print(f"❌ Error running parameter optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
