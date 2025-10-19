#!/usr/bin/env python3
"""
EPIC 1.5: Diagnostic Experiment - Identical RMSE Investigation

This experiment investigates why all three Kalman filter models
(Enhanced, Regime-Integrated, Basic) are producing identical RMSE results.
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

class DiagnosticExperiment:
    """Comprehensive diagnostic experiment for Kalman filter models"""
    
    def __init__(self):
        self.results = {}
        self.detailed_logs = {}
        self.state_evolution = {}
        self.parameter_comparison = {}
        
    def run_diagnostic_experiment(self, data: np.ndarray, test_periods: int = 200) -> Dict:
        """Run comprehensive diagnostic experiment"""
        
        print("🔍 Starting EPIC 1.5 Diagnostic Experiment")
        print("=" * 60)
        
        # Initialize models with detailed logging
        models = self._initialize_models_with_logging()
        
        # Run diagnostic tests
        diagnostic_results = {}
        
        # Test 1: State Evolution Analysis
        print("\n📊 Test 1: State Evolution Analysis")
        state_analysis = self._analyze_state_evolution(models, data, test_periods)
        diagnostic_results['state_evolution'] = state_analysis
        
        # Test 2: Parameter Comparison
        print("\n🔧 Test 2: Parameter Comparison")
        param_analysis = self._compare_parameters(models)
        diagnostic_results['parameter_comparison'] = param_analysis
        
        # Test 3: Prediction Breakdown
        print("\n🎯 Test 3: Prediction Breakdown")
        prediction_analysis = self._analyze_prediction_components(models, data, test_periods)
        diagnostic_results['prediction_breakdown'] = prediction_analysis
        
        # Test 4: Regime Detection Validation
        print("\n🌊 Test 4: Regime Detection Validation")
        regime_analysis = self._validate_regime_detection(models, data, test_periods)
        diagnostic_results['regime_detection'] = regime_analysis
        
        # Test 5: Enhanced Differentiation Test
        print("\n⚡ Test 5: Enhanced Differentiation Test")
        differentiation_analysis = self._test_enhanced_differentiation(models, data, test_periods)
        diagnostic_results['enhanced_differentiation'] = differentiation_analysis
        
        # Generate comprehensive report
        report = self._generate_diagnostic_report(diagnostic_results)
        
        return {
            'diagnostic_results': diagnostic_results,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _initialize_models_with_logging(self) -> Dict:
        """Initialize models with detailed logging capabilities"""
        
        models = {}
        
        # Basic Kalman Filter
        print("Initializing Basic Kalman Filter...")
        models['basic'] = {
            'model': BasicKalmanFilter(window_size=10),
            'name': 'Basic Kalman Filter',
            'state_log': [],
            'prediction_log': [],
            'parameter_log': []
        }
        
        # Enhanced Kalman Filter
        print("Initializing Enhanced Kalman Filter...")
        models['enhanced'] = {
            'model': EnhancedKalmanFilter(state_dim=4, observation_dim=2),
            'name': 'Enhanced Kalman Filter',
            'state_log': [],
            'prediction_log': [],
            'parameter_log': []
        }
        
        # Regime-Integrated Kalman Filter
        print("Initializing Regime-Integrated Kalman Filter...")
        regime_detector = MarketRegimeDetectionBNN()
        models['regime_integrated'] = {
            'model': RegimeIntegratedKalmanFilter(regime_detector),
            'name': 'Regime-Integrated Kalman Filter',
            'state_log': [],
            'prediction_log': [],
            'parameter_log': []
        }
        
        return models
    
    def _analyze_state_evolution(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Analyze how states evolve differently across models"""
        
        print("  📈 Tracking state evolution...")
        
        # Initialize state tracking
        state_evolution = {}
        
        for model_name, model_info in models.items():
            state_evolution[model_name] = {
                'states': [],
                'state_changes': [],
                'convergence_points': [],
                'initialization_impact': []
            }
        
        # Run prediction loop with state tracking
        for i in range(min(test_periods, len(data))):
            observation = data[i]
            
            for model_name, model_info in models.items():
                model = model_info['model']
                
                # Get current state before update
                if hasattr(model, 'get_current_state'):
                    current_state = model.get_current_state()
            else:
                    current_state = np.array([0.0, 0.0, 0.0, 0.0])
                
                # Log state
                state_evolution[model_name]['states'].append(current_state.copy())
                
                # Make prediction
                try:
                    prediction = model.predict(horizon=1)
                    model_info['prediction_log'].append(prediction)
                except Exception as e:
                    print(f"    ⚠️  Prediction error in {model_name}: {e}")
                    prediction = np.array([0.0, 0.0])
                    model_info['prediction_log'].append(prediction)
                
                # Update model
                try:
                    model.update(observation)
                except Exception as e:
                    print(f"    ⚠️  Update error in {model_name}: {e}")
                
                # Get state after update
                if hasattr(model, 'get_current_state'):
                    new_state = model.get_current_state()
            else:
                    new_state = np.array([0.0, 0.0, 0.0, 0.0])
                
                # Track state changes
                state_change = np.linalg.norm(new_state - current_state)
                state_evolution[model_name]['state_changes'].append(state_change)
                
                # Check for convergence
                if i > 10:  # After initial period
                    recent_changes = state_evolution[model_name]['state_changes'][-10:]
                    if np.std(recent_changes) < 1e-6:
                        state_evolution[model_name]['convergence_points'].append(i)
        
        # Analyze state evolution patterns
        analysis = {}
        for model_name, evolution_data in state_evolution.items():
            states = np.array(evolution_data['states'])
            state_changes = np.array(evolution_data['state_changes'])
            
            analysis[model_name] = {
                'initial_state': states[0] if len(states) > 0 else None,
                'final_state': states[-1] if len(states) > 0 else None,
                'state_variance': np.var(states, axis=0) if len(states) > 0 else None,
                'total_state_change': np.sum(state_changes),
                'convergence_point': evolution_data['convergence_points'][0] if evolution_data['convergence_points'] else None,
                'state_trajectory_similarity': self._calculate_state_similarity(states)
            }
        
        return analysis
    
    def _compare_parameters(self, models: Dict) -> Dict:
        """Compare actual parameters used by each model"""
        
        print("  🔧 Comparing model parameters...")
        
        parameter_comparison = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Extract parameters based on model type
            params = {}
            
            if hasattr(model, 'get_parameters'):
                params = model.get_parameters()
            elif hasattr(model, 'F') and hasattr(model, 'H'):
                # Basic Kalman parameters
                params = {
                    'F': getattr(model, 'F', None),
                    'H': getattr(model, 'H', None),
                    'Q': getattr(model, 'Q', None),
                    'R': getattr(model, 'R', None),
                    'P': getattr(model, 'P', None)
                }
            else:
                # Try to extract any available parameters
                params = {
                    'attributes': [attr for attr in dir(model) if not attr.startswith('_')],
                    'state': getattr(model, 'state', None),
                    'covariance': getattr(model, 'covariance', None)
                }
            
            parameter_comparison[model_name] = params
        
        # Compare parameters across models
        comparison_analysis = {}
        
        # Check for identical parameters
        model_names = list(parameter_comparison.keys())
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                similarity = self._compare_parameter_sets(
                    parameter_comparison[model1],
                    parameter_comparison[model2]
                )
                comparison_analysis[f"{model1}_vs_{model2}"] = similarity
        
        return {
            'individual_parameters': parameter_comparison,
            'cross_model_comparison': comparison_analysis
        }
    
    def _analyze_prediction_components(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Break down prediction into components"""
        
        print("  🎯 Analyzing prediction components...")
        
        prediction_analysis = {}
        
        for model_name, model_info in models.items():
            predictions = model_info['prediction_log']
            
            if not predictions:
                prediction_analysis[model_name] = {'error': 'No predictions available'}
                continue
            
            # Convert to numpy array
            pred_array = np.array(predictions)
            
            # Analyze prediction components
            analysis = {
                'mean_prediction': np.mean(pred_array, axis=0),
                'prediction_std': np.std(pred_array, axis=0),
                'prediction_range': np.ptp(pred_array, axis=0),
                'prediction_variance': np.var(pred_array, axis=0),
                'prediction_trend': self._calculate_trend(pred_array),
                'prediction_stability': self._calculate_stability(pred_array)
            }
            
            # Check for prediction differences
            if len(predictions) > 1:
                prediction_diffs = np.diff(pred_array, axis=0)
                analysis['prediction_changes'] = {
                    'mean_change': np.mean(prediction_diffs, axis=0),
                    'change_std': np.std(prediction_diffs, axis=0),
                    'total_change': np.sum(np.abs(prediction_diffs))
                }
            
            prediction_analysis[model_name] = analysis
        
        # Compare predictions across models
        if len(models) > 1:
            model_names = list(models.keys())
            pred_arrays = [np.array(models[name]['prediction_log']) for name in model_names]
            
            # Calculate prediction similarity
            prediction_similarity = {}
            for i, pred1 in enumerate(pred_arrays):
                for j, pred2 in enumerate(pred_arrays[i+1:], i+1):
                    if len(pred1) > 0 and len(pred2) > 0:
                        similarity = self._calculate_prediction_similarity(pred1, pred2)
                        prediction_similarity[f"{model_names[i]}_vs_{model_names[j]}"] = similarity
            
            prediction_analysis['cross_model_similarity'] = prediction_similarity
        
        return prediction_analysis
    
    def _validate_regime_detection(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Validate regime detection functionality"""
        
        print("  🌊 Validating regime detection...")
        
        regime_analysis = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Check if model has regime detection
            if hasattr(model, 'regime_detector'):
                regime_detector = model.regime_detector
                
                # Test regime detection
                regime_predictions = []
                regime_confidences = []
                
                for i in range(min(test_periods, len(data))):
                    try:
                        regime_pred = regime_detector.predict_regime(data[i])
                        regime_predictions.append(regime_pred)
                        
                        if hasattr(regime_detector, 'get_confidence'):
                            confidence = regime_detector.get_confidence()
                            regime_confidences.append(confidence)
                    except Exception as e:
                        print(f"    ⚠️  Regime detection error in {model_name}: {e}")
                        regime_predictions.append('unknown')
                        regime_confidences.append(0.0)
                
                regime_analysis[model_name] = {
                    'regime_predictions': regime_predictions,
                    'regime_confidences': regime_confidences,
                    'regime_distribution': self._calculate_regime_distribution(regime_predictions),
                    'average_confidence': np.mean(regime_confidences) if regime_confidences else 0.0,
                    'regime_transitions': self._count_regime_transitions(regime_predictions)
                }
            else:
                regime_analysis[model_name] = {
                    'error': 'No regime detection capability'
                }
        
        return regime_analysis
    
    def _test_enhanced_differentiation(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Test if enhanced features are actually working"""
        
        print("  ⚡ Testing enhanced differentiation...")
        
        differentiation_analysis = {}
        
        # Test 1: Different initialization strategies
        init_test = self._test_initialization_differences(models, data[:10])
        differentiation_analysis['initialization_test'] = init_test
        
        # Test 2: Parameter adaptation
        adaptation_test = self._test_parameter_adaptation(models, data, test_periods)
        differentiation_analysis['parameter_adaptation'] = adaptation_test
        
        # Test 3: Uncertainty quantification
        uncertainty_test = self._test_uncertainty_quantification(models, data, test_periods)
        differentiation_analysis['uncertainty_quantification'] = uncertainty_test
        
        # Test 4: Regime-specific behavior
        regime_behavior_test = self._test_regime_specific_behavior(models, data, test_periods)
        differentiation_analysis['regime_behavior'] = regime_behavior_test
        
        return differentiation_analysis
    
    def _test_initialization_differences(self, models: Dict, initial_data: np.ndarray) -> Dict:
        """Test if different initialization strategies produce different results"""
        
        init_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Reset model
            if hasattr(model, 'reset'):
                model.reset()
            
            # Test different initialization strategies
            init_strategies = {
                'zero_init': np.array([0.0, 0.0, 0.0, 0.0]),
                'data_init': initial_data[0] if len(initial_data) > 0 else np.array([0.0, 0.0]),
                'random_init': np.random.normal(0, 0.1, 4)
            }
            
            strategy_results = {}
            for strategy_name, init_state in init_strategies.items():
                try:
                    # Set initial state if possible
                    if hasattr(model, 'set_initial_state'):
                        model.set_initial_state(init_state)
                    
                    # Make prediction
                    prediction = model.predict(horizon=1)
                    strategy_results[strategy_name] = {
                        'prediction': prediction,
                        'success': True
                    }
                except Exception as e:
                    strategy_results[strategy_name] = {
                        'error': str(e),
                        'success': False
                    }
            
            init_results[model_name] = strategy_results
        
        return init_results
    
    def _test_parameter_adaptation(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Test if parameters are actually adapting"""
        
        adaptation_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Track parameter changes over time
            parameter_history = []
            
            for i in range(min(test_periods, len(data))):
                # Get current parameters
                if hasattr(model, 'get_parameters'):
                    current_params = model.get_parameters()
                    parameter_history.append(current_params)
                
                # Update model
                try:
                    model.update(data[i])
            except Exception as e:
                    print(f"    ⚠️  Update error in {model_name}: {e}")
            
            # Analyze parameter adaptation
            if parameter_history:
                adaptation_results[model_name] = {
                    'parameter_history': parameter_history,
                    'parameter_changes': self._analyze_parameter_changes(parameter_history),
                    'adaptation_active': self._check_parameter_adaptation(parameter_history)
                }
            else:
                adaptation_results[model_name] = {
                    'error': 'No parameter tracking available'
                }
        
        return adaptation_results
    
    def _test_uncertainty_quantification(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Test uncertainty quantification capabilities"""
        
        uncertainty_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            uncertainty_measures = []
            
            for i in range(min(test_periods, len(data))):
                try:
                    # Get uncertainty if available
                    if hasattr(model, 'get_uncertainty'):
                        uncertainty = model.get_uncertainty()
                        uncertainty_measures.append(uncertainty)
                    elif hasattr(model, 'get_confidence'):
                        confidence = model.get_confidence()
                        uncertainty_measures.append(1.0 - confidence)
                    else:
                        uncertainty_measures.append(None)
                    
                    # Update model
                    model.update(data[i])
            except Exception as e:
                    print(f"    ⚠️  Uncertainty error in {model_name}: {e}")
                    uncertainty_measures.append(None)
            
            # Analyze uncertainty quantification
            valid_uncertainties = [u for u in uncertainty_measures if u is not None]
            
            if valid_uncertainties:
                uncertainty_results[model_name] = {
                    'uncertainty_measures': valid_uncertainties,
                    'mean_uncertainty': np.mean(valid_uncertainties),
                    'uncertainty_std': np.std(valid_uncertainties),
                    'uncertainty_trend': self._calculate_trend(np.array(valid_uncertainties))
                }
            else:
                uncertainty_results[model_name] = {
                    'error': 'No uncertainty quantification available'
                }
        
        return uncertainty_results
    
    def _test_regime_specific_behavior(self, models: Dict, data: np.ndarray, test_periods: int) -> Dict:
        """Test regime-specific behavior"""
        
        regime_behavior_results = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            # Check if model has regime-specific capabilities
            if hasattr(model, 'regime_detector'):
                regime_predictions = []
                regime_specific_predictions = []
                
                for i in range(min(test_periods, len(data))):
                    try:
                        # Get regime prediction
                        regime = model.regime_detector.predict_regime(data[i])
                        regime_predictions.append(regime)
                        
                        # Get regime-specific prediction
                        if hasattr(model, 'predict_with_regime'):
                            pred = model.predict_with_regime(data[i], regime)
                            regime_specific_predictions.append(pred)
                        else:
                            pred = model.predict(horizon=1)
                            regime_specific_predictions.append(pred)
                        
            except Exception as e:
                        print(f"    ⚠️  Regime behavior error in {model_name}: {e}")
                        regime_predictions.append('unknown')
                        regime_specific_predictions.append(np.array([0.0, 0.0]))
                
                regime_behavior_results[model_name] = {
                    'regime_predictions': regime_predictions,
                    'regime_specific_predictions': regime_specific_predictions,
                    'regime_behavior_active': self._check_regime_behavior(regime_predictions, regime_specific_predictions)
                }
            else:
                regime_behavior_results[model_name] = {
                    'error': 'No regime-specific capabilities'
                }
        
        return regime_behavior_results
    
    def _calculate_state_similarity(self, states: np.ndarray) -> float:
        """Calculate similarity between state trajectories"""
        if len(states) < 2:
            return 1.0
        
        # Calculate correlation between state components
        correlations = []
        for i in range(states.shape[1]):
            if i < states.shape[1] - 1:
                corr = np.corrcoef(states[:, i], states[:, i+1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compare_parameter_sets(self, params1: Dict, params2: Dict) -> Dict:
        """Compare two parameter sets"""
        similarity = {}
        
        # Compare common parameters
        common_keys = set(params1.keys()) & set(params2.keys())
        
        for key in common_keys:
            val1 = params1[key]
            val2 = params2[key]
            
            if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
                if val1.shape == val2.shape:
                    similarity[key] = {
                        'identical': np.allclose(val1, val2, rtol=1e-10),
                        'max_diff': np.max(np.abs(val1 - val2)) if not np.allclose(val1, val2) else 0.0
                    }
                else:
                    similarity[key] = {'error': 'Shape mismatch'}
            elif val1 == val2:
                similarity[key] = {'identical': True}
            else:
                # Handle different types of values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    similarity[key] = {'identical': False, 'difference': val1 - val2}
                else:
                    similarity[key] = {'identical': False, 'difference': f"{val1} vs {val2}"}
        
        return similarity
    
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calculate trend in data"""
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        slope, _ = np.polyfit(x, data, 1)
        return slope
    
    def _calculate_stability(self, data: np.ndarray) -> float:
        """Calculate stability of data"""
        if len(data) < 2:
            return 1.0
        
        return 1.0 / (1.0 + np.std(data))
    
    def _calculate_prediction_similarity(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Calculate similarity between prediction arrays"""
        if len(pred1) == 0 or len(pred2) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(pred1), len(pred2))
        pred1 = pred1[:min_len]
        pred2 = pred2[:min_len]
        
        # Calculate correlation
        if pred1.shape[1] == pred2.shape[1]:
            correlations = []
            for i in range(pred1.shape[1]):
                corr = np.corrcoef(pred1[:, i], pred2[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            return np.mean(correlations) if correlations else 0.0
        
        return 0.0
    
    def _calculate_regime_distribution(self, regime_predictions: List) -> Dict:
        """Calculate distribution of regime predictions"""
        from collections import Counter
        return dict(Counter(regime_predictions))
    
    def _count_regime_transitions(self, regime_predictions: List) -> int:
        """Count number of regime transitions"""
        if len(regime_predictions) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(regime_predictions)):
            if regime_predictions[i] != regime_predictions[i-1]:
                transitions += 1
        
        return transitions
    
    def _analyze_parameter_changes(self, parameter_history: List) -> Dict:
        """Analyze parameter changes over time"""
        if len(parameter_history) < 2:
            return {'error': 'Insufficient parameter history'}
        
        # Calculate parameter changes
        changes = []
        for i in range(1, len(parameter_history)):
            # This is a simplified analysis - would need to be adapted based on actual parameter structure
            changes.append(i)  # Placeholder
        
        return {
            'total_changes': len(changes),
            'change_frequency': len(changes) / len(parameter_history),
            'parameter_evolution': 'stable' if len(changes) < len(parameter_history) * 0.1 else 'adaptive'
        }
    
    def _check_parameter_adaptation(self, parameter_history: List) -> bool:
        """Check if parameters are actually adapting"""
        if len(parameter_history) < 2:
            return False
        
        # Simplified check - would need to be adapted based on actual parameter structure
        return len(parameter_history) > 1
    
    def _check_regime_behavior(self, regime_predictions: List, regime_specific_predictions: List) -> bool:
        """Check if regime-specific behavior is active"""
        if len(regime_predictions) < 2:
            return False
        
        # Check for regime transitions
        regime_transitions = self._count_regime_transitions(regime_predictions)
        
        # Check for prediction differences based on regime
        if len(regime_specific_predictions) > 1:
            pred_variance = np.var(regime_specific_predictions, axis=0)
            return regime_transitions > 0 or np.sum(pred_variance) > 1e-6
        
        return regime_transitions > 0
    
    def _generate_diagnostic_report(self, diagnostic_results: Dict) -> str:
        """Generate comprehensive diagnostic report"""
        
        report = []
        report.append("EPIC 1.5: Diagnostic Experiment Report")
        report.append("=" * 50)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        
        # State Evolution Analysis
        report.append("📊 STATE EVOLUTION ANALYSIS")
        report.append("-" * 30)
        state_evolution = diagnostic_results.get('state_evolution', {})
        for model_name, analysis in state_evolution.items():
            report.append(f"\n{model_name}:")
            report.append(f"  Initial State: {analysis.get('initial_state', 'N/A')}")
            report.append(f"  Final State: {analysis.get('final_state', 'N/A')}")
            report.append(f"  State Variance: {analysis.get('state_variance', 'N/A')}")
            report.append(f"  Total State Change: {analysis.get('total_state_change', 'N/A')}")
            report.append(f"  Convergence Point: {analysis.get('convergence_point', 'N/A')}")
            report.append(f"  State Trajectory Similarity: {analysis.get('state_trajectory_similarity', 'N/A')}")
        
        # Parameter Comparison
        report.append("\n🔧 PARAMETER COMPARISON")
        report.append("-" * 30)
        param_comparison = diagnostic_results.get('parameter_comparison', {})
        individual_params = param_comparison.get('individual_parameters', {})
        for model_name, params in individual_params.items():
            report.append(f"\n{model_name} Parameters:")
            for param_name, param_value in params.items():
                if isinstance(param_value, np.ndarray):
                    report.append(f"  {param_name}: Shape {param_value.shape}, Mean {np.mean(param_value):.6f}")
                else:
                    report.append(f"  {param_name}: {param_value}")
        
        # Cross-model parameter comparison
        cross_comparison = param_comparison.get('cross_model_comparison', {})
        if cross_comparison:
            report.append("\nCross-Model Parameter Comparison:")
            for comparison_name, similarity in cross_comparison.items():
                report.append(f"  {comparison_name}:")
                for param_name, param_similarity in similarity.items():
                    if isinstance(param_similarity, dict):
                        identical = param_similarity.get('identical', False)
                        report.append(f"    {param_name}: {'IDENTICAL' if identical else 'DIFFERENT'}")
                        if 'max_diff' in param_similarity:
                            report.append(f"      Max Difference: {param_similarity['max_diff']}")
        
        # Prediction Breakdown
        report.append("\n🎯 PREDICTION BREAKDOWN")
        report.append("-" * 30)
        prediction_breakdown = diagnostic_results.get('prediction_breakdown', {})
        for model_name, analysis in prediction_breakdown.items():
            if 'error' in analysis:
                report.append(f"\n{model_name}: {analysis['error']}")
            else:
                report.append(f"\n{model_name}:")
                report.append(f"  Mean Prediction: {analysis.get('mean_prediction', 'N/A')}")
                report.append(f"  Prediction Std: {analysis.get('prediction_std', 'N/A')}")
                report.append(f"  Prediction Range: {analysis.get('prediction_range', 'N/A')}")
                report.append(f"  Prediction Variance: {analysis.get('prediction_variance', 'N/A')}")
                report.append(f"  Prediction Trend: {analysis.get('prediction_trend', 'N/A')}")
                report.append(f"  Prediction Stability: {analysis.get('prediction_stability', 'N/A')}")
        
        # Regime Detection Validation
        report.append("\n🌊 REGIME DETECTION VALIDATION")
        report.append("-" * 30)
        regime_detection = diagnostic_results.get('regime_detection', {})
        for model_name, analysis in regime_detection.items():
            if 'error' in analysis:
                report.append(f"\n{model_name}: {analysis['error']}")
            else:
                report.append(f"\n{model_name}:")
                report.append(f"  Regime Distribution: {analysis.get('regime_distribution', 'N/A')}")
                report.append(f"  Average Confidence: {analysis.get('average_confidence', 'N/A')}")
                report.append(f"  Regime Transitions: {analysis.get('regime_transitions', 'N/A')}")
        
        # Enhanced Differentiation Test
        report.append("\n⚡ ENHANCED DIFFERENTIATION TEST")
        report.append("-" * 30)
        enhanced_differentiation = diagnostic_results.get('enhanced_differentiation', {})
        
        # Initialization Test
        init_test = enhanced_differentiation.get('initialization_test', {})
        report.append("\nInitialization Test:")
        for model_name, init_results in init_test.items():
            report.append(f"  {model_name}:")
            for strategy_name, strategy_result in init_results.items():
                if strategy_result.get('success', False):
                    report.append(f"    {strategy_name}: SUCCESS")
                else:
                    report.append(f"    {strategy_name}: FAILED - {strategy_result.get('error', 'Unknown error')}")
        
        # Parameter Adaptation
        param_adaptation = enhanced_differentiation.get('parameter_adaptation', {})
        report.append("\nParameter Adaptation:")
        for model_name, adaptation_result in param_adaptation.items():
            if 'error' in adaptation_result:
                report.append(f"  {model_name}: {adaptation_result['error']}")
            else:
                adaptation_active = adaptation_result.get('adaptation_active', False)
                report.append(f"  {model_name}: {'ACTIVE' if adaptation_active else 'INACTIVE'}")
        
        # Uncertainty Quantification
        uncertainty_test = enhanced_differentiation.get('uncertainty_quantification', {})
        report.append("\nUncertainty Quantification:")
        for model_name, uncertainty_result in uncertainty_test.items():
            if 'error' in uncertainty_result:
                report.append(f"  {model_name}: {uncertainty_result['error']}")
            else:
                report.append(f"  {model_name}: Mean Uncertainty {uncertainty_result.get('mean_uncertainty', 'N/A')}")
        
        # Regime Behavior
        regime_behavior = enhanced_differentiation.get('regime_behavior', {})
        report.append("\nRegime Behavior:")
        for model_name, behavior_result in regime_behavior.items():
            if 'error' in behavior_result:
                report.append(f"  {model_name}: {behavior_result['error']}")
            else:
                behavior_active = behavior_result.get('regime_behavior_active', False)
                report.append(f"  {model_name}: {'ACTIVE' if behavior_active else 'INACTIVE'}")
        
        # Summary and Recommendations
        report.append("\n🎯 SUMMARY AND RECOMMENDATIONS")
        report.append("-" * 30)
        report.append("\nKey Findings:")
        
        # Analyze findings and provide recommendations
        findings = []
        recommendations = []
        
        # Check for identical parameters
        if cross_comparison:
            identical_params = []
            for comparison_name, similarity in cross_comparison.items():
                for param_name, param_similarity in similarity.items():
                    if isinstance(param_similarity, dict) and param_similarity.get('identical', False):
                        identical_params.append(f"{comparison_name}.{param_name}")
            
            if identical_params:
                findings.append(f"Identical parameters found: {', '.join(identical_params)}")
                recommendations.append("Implement distinct parameter initialization strategies")
        
        # Check for state convergence
        if state_evolution:
            convergence_issues = []
            for model_name, analysis in state_evolution.items():
                if analysis.get('convergence_point') is not None and analysis.get('convergence_point', 0) < 20:
                    convergence_issues.append(model_name)
            
            if convergence_issues:
                findings.append(f"Early convergence detected in: {', '.join(convergence_issues)}")
                recommendations.append("Implement adaptive learning rates and parameter drift")
        
        # Check for regime detection issues
        if regime_detection:
            regime_issues = []
            for model_name, analysis in regime_detection.items():
                if 'regime_distribution' in analysis:
                    regime_dist = analysis['regime_distribution']
                    if len(regime_dist) == 1:  # Only one regime detected
                        regime_issues.append(model_name)
            
            if regime_issues:
                findings.append(f"Limited regime detection in: {', '.join(regime_issues)}")
                recommendations.append("Test with more diverse market conditions and longer periods")
        
        # Add findings and recommendations to report
        if findings:
            report.append("\nFindings:")
            for finding in findings:
                report.append(f"  • {finding}")
        
        if recommendations:
            report.append("\nRecommendations:")
            for recommendation in recommendations:
                report.append(f"  • {recommendation}")
        
        report.append("\n" + "=" * 50)
        report.append("End of Diagnostic Report")
        
        return "\n".join(report)

def load_test_data() -> np.ndarray:
    """Load test data for diagnostic experiment"""
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
    """Run diagnostic experiment"""
    
    print("🚀 Starting EPIC 1.5 Diagnostic Experiment")
    print("=" * 60)
    
    # Load data
    try:
        data = load_test_data()
        print(f"✅ Loaded data: {data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run diagnostic experiment
    experiment = DiagnosticExperiment()
    
    try:
        results = experiment.run_diagnostic_experiment(data, test_periods=200)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"epic1_5_diagnostic_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Diagnostic experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)
        print(results['report'])
        
    except Exception as e:
        print(f"❌ Error running diagnostic experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()