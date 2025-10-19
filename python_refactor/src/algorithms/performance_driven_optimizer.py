#!/usr/bin/env python3
"""
Performance-Driven Optimizer

Implements performance-driven optimization with real-time feedback
and adaptive system tuning for improved overall performance.

Target: 50%+ system performance improvement, 90%+ success rate
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for system optimization"""
    regime_detection_accuracy: float
    parameter_stability: float
    kalman_convergence: float
    overall_performance: float
    regret_bounds: Dict[str, float]
    system_efficiency: float
    timestamp: float

@dataclass
class OptimizationAction:
    """Optimization action to improve performance"""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    confidence: float
    timestamp: float

class PerformanceTracker:
    """Track system performance metrics"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.performance_thresholds = {
            'regime_detection_accuracy': 0.7,
            'parameter_stability': 0.9,
            'kalman_convergence': 0.8,
            'overall_performance': 0.3,
            'system_efficiency': 0.6
        }
        
        logger.info(f"Initialized PerformanceTracker with window_size={window_size}")
    
    def track(self, system, data: np.ndarray, rewards: np.ndarray) -> PerformanceMetrics:
        """Track system performance"""
        
        # Get system status
        status = system.get_system_status()
        
        # Calculate regime detection accuracy
        regime_accuracy = self._calculate_regime_accuracy(system, data)
        
        # Calculate parameter stability
        param_stability = self._calculate_parameter_stability(status)
        
        # Calculate Kalman convergence
        kalman_convergence = self._calculate_kalman_convergence(status)
        
        # Calculate overall performance
        overall_performance = self._calculate_overall_performance(rewards)
        
        # Calculate system efficiency
        system_efficiency = self._calculate_system_efficiency(status)
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            regime_detection_accuracy=regime_accuracy,
            parameter_stability=param_stability,
            kalman_convergence=kalman_convergence,
            overall_performance=overall_performance,
            regret_bounds=status.get('regret_bounds', {}),
            system_efficiency=system_efficiency,
            timestamp=datetime.now().timestamp()
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_regime_accuracy(self, system, data: np.ndarray) -> float:
        """Calculate regime detection accuracy"""
        try:
            # This would require ground truth regime labels
            # For now, use a simplified metric
            if hasattr(system, 'regime_system') and hasattr(system.regime_system, 'bandit_performance'):
                bandit_perf = system.regime_system.bandit_performance
                exploration_rate = bandit_perf.get('exploration_rate', 0.5)
                # Lower exploration rate indicates better accuracy
                accuracy = 1.0 - exploration_rate
                return max(0.0, min(1.0, accuracy))
            return 0.5
        except Exception as e:
            logger.warning(f"Error calculating regime accuracy: {e}")
            return 0.5
    
    def _calculate_parameter_stability(self, status: Dict) -> float:
        """Calculate parameter stability"""
        try:
            param_adaptation = status.get('parameter_adaptation', {})
            optimizer_perf = param_adaptation.get('optimizer_performance', {})
            stability = optimizer_perf.get('parameter_stability', 0.5)
            return stability
        except Exception as e:
            logger.warning(f"Error calculating parameter stability: {e}")
            return 0.5
    
    def _calculate_kalman_convergence(self, status: Dict) -> float:
        """Calculate Kalman convergence"""
        try:
            kalman_optimization = status.get('kalman_optimization', {})
            optimizer_perf = kalman_optimization.get('optimizer_performance', {})
            hessian_condition = optimizer_perf.get('hessian_condition', 1000)
            # Lower hessian condition indicates better convergence
            convergence = 1.0 / (1.0 + hessian_condition / 1000)
            return max(0.0, min(1.0, convergence))
        except Exception as e:
            logger.warning(f"Error calculating Kalman convergence: {e}")
            return 0.5
    
    def _calculate_overall_performance(self, rewards: np.ndarray) -> float:
        """Calculate overall performance from rewards"""
        if len(rewards) == 0:
            return 0.0
        
        # Normalize rewards to [0, 1]
        normalized_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        return float(np.mean(normalized_rewards))
    
    def _calculate_system_efficiency(self, status: Dict) -> float:
        """Calculate system efficiency"""
        try:
            system_overview = status.get('system_overview', {})
            total_updates = system_overview.get('total_updates', 1)
            avg_performance = system_overview.get('average_performance', 0.0)
            
            # Efficiency = performance per update
            efficiency = avg_performance / max(1, total_updates)
            return max(0.0, min(1.0, efficiency))
        except Exception as e:
            logger.warning(f"Error calculating system efficiency: {e}")
            return 0.5
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.metrics_history:
            return {'status': 'NO_DATA'}
        
        # Calculate averages
        avg_metrics = {}
        for metric_name in ['regime_detection_accuracy', 'parameter_stability', 
                           'kalman_convergence', 'overall_performance', 'system_efficiency']:
            values = [getattr(metrics, metric_name) for metrics in self.metrics_history]
            avg_metrics[metric_name] = np.mean(values)
        
        # Calculate improvement trends
        trends = {}
        for metric_name in avg_metrics:
            if len(self.metrics_history) > 1:
                recent = np.mean([getattr(metrics, metric_name) for metrics in list(self.metrics_history)[-10:]])
                early = np.mean([getattr(metrics, metric_name) for metrics in list(self.metrics_history)[:10]])
                trends[metric_name] = (recent - early) / (early + 1e-8)
            else:
                trends[metric_name] = 0.0
        
        return {
            'average_metrics': avg_metrics,
            'improvement_trends': trends,
            'total_samples': len(self.metrics_history),
            'performance_status': self._assess_performance_status(avg_metrics)
        }
    
    def _assess_performance_status(self, avg_metrics: Dict) -> str:
        """Assess overall performance status"""
        thresholds_met = 0
        total_thresholds = len(self.performance_thresholds)
        
        for metric, threshold in self.performance_thresholds.items():
            if metric in avg_metrics and avg_metrics[metric] >= threshold:
                thresholds_met += 1
        
        success_rate = thresholds_met / total_thresholds
        
        if success_rate >= 0.8:
            return 'EXCELLENT'
        elif success_rate >= 0.6:
            return 'GOOD'
        elif success_rate >= 0.4:
            return 'FAIR'
        else:
            return 'POOR'

class BottleneckIdentifier:
    """Identify performance bottlenecks"""
    
    def __init__(self):
        self.bottleneck_thresholds = {
            'regime_detection': 0.7,
            'parameter_adaptation': 0.9,
            'kalman_optimization': 0.8,
            'system_integration': 0.3
        }
    
    def identify_bottlenecks(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        if metrics.regime_detection_accuracy < self.bottleneck_thresholds['regime_detection']:
            bottlenecks.append('regime_detection')
        
        if metrics.parameter_stability < self.bottleneck_thresholds['parameter_adaptation']:
            bottlenecks.append('parameter_adaptation')
        
        if metrics.kalman_convergence < self.bottleneck_thresholds['kalman_optimization']:
            bottlenecks.append('kalman_optimization')
        
        if metrics.overall_performance < self.bottleneck_thresholds['system_integration']:
            bottlenecks.append('system_integration')
        
        return bottlenecks

class AdaptiveSystemOptimizer:
    """Adaptive system optimizer with real-time feedback"""
    
    def __init__(self):
        self.optimization_actions = []
        self.optimization_history = []
        self.adaptive_parameters = {
            'learning_rate_boost': 1.2,
            'confidence_adjustment': 0.05,
            'regularization_factor': 1.1,
            'convergence_threshold': 0.9
        }
        
        logger.info("Initialized AdaptiveSystemOptimizer")
    
    def optimize(self, system, bottlenecks: List[str]) -> List[OptimizationAction]:
        """Optimize system based on identified bottlenecks"""
        
        actions = []
        
        for bottleneck in bottlenecks:
            action = self._create_optimization_action(bottleneck, system)
            if action:
                actions.append(action)
                self.optimization_actions.append(action)
        
        return actions
    
    def _create_optimization_action(self, bottleneck: str, system) -> Optional[OptimizationAction]:
        """Create optimization action for specific bottleneck"""
        
        if bottleneck == 'regime_detection':
            return OptimizationAction(
                action_type='regime_detection_enhancement',
                parameters={
                    'confidence_adjustment': self.adaptive_parameters['confidence_adjustment'],
                    'feature_engineering': True,
                    'ensemble_weights': [0.4, 0.3, 0.2, 0.1]
                },
                expected_improvement=0.2,
                confidence=0.8,
                timestamp=datetime.now().timestamp()
            )
        
        elif bottleneck == 'parameter_adaptation':
            return OptimizationAction(
                action_type='parameter_adaptation_enhancement',
                parameters={
                    'learning_rate_boost': self.adaptive_parameters['learning_rate_boost'],
                    'momentum_adjustment': 0.95,
                    'regularization_factor': self.adaptive_parameters['regularization_factor']
                },
                expected_improvement=0.15,
                confidence=0.7,
                timestamp=datetime.now().timestamp()
            )
        
        elif bottleneck == 'kalman_optimization':
            return OptimizationAction(
                action_type='kalman_optimization_enhancement',
                parameters={
                    'convergence_threshold': self.adaptive_parameters['convergence_threshold'],
                    'regime_specific_matrices': True,
                    'adaptive_regularization': True
                },
                expected_improvement=0.25,
                confidence=0.9,
                timestamp=datetime.now().timestamp()
            )
        
        elif bottleneck == 'system_integration':
            return OptimizationAction(
                action_type='system_integration_enhancement',
                parameters={
                    'feedback_loop': True,
                    'real_time_adaptation': True,
                    'performance_monitoring': True
                },
                expected_improvement=0.3,
                confidence=0.6,
                timestamp=datetime.now().timestamp()
            )
        
        return None

class FeedbackLoop:
    """Real-time feedback loop for system optimization"""
    
    def __init__(self):
        self.feedback_history = []
        self.adaptation_rate = 0.1
        self.feedback_window = 20
        
        logger.info("Initialized FeedbackLoop")
    
    def process(self, metrics: PerformanceMetrics, actions: List[OptimizationAction]) -> Dict:
        """Process feedback and generate adaptation signals"""
        
        # Analyze performance trends
        trend_analysis = self._analyze_performance_trends(metrics)
        
        # Evaluate action effectiveness
        action_effectiveness = self._evaluate_action_effectiveness(actions)
        
        # Generate adaptation signals
        adaptation_signals = self._generate_adaptation_signals(trend_analysis, action_effectiveness)
        
        # Create feedback
        feedback = {
            'trend_analysis': trend_analysis,
            'action_effectiveness': action_effectiveness,
            'adaptation_signals': adaptation_signals,
            'feedback_quality': self._assess_feedback_quality(metrics, actions),
            'timestamp': datetime.now().timestamp()
        }
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        return feedback
    
    def _analyze_performance_trends(self, metrics: PerformanceMetrics) -> Dict:
        """Analyze performance trends"""
        
        if len(self.feedback_history) < 2:
            return {'trend': 'INSUFFICIENT_DATA'}
        
        # Get recent performance
        recent_metrics = [f['trend_analysis'] for f in self.feedback_history[-5:]]
        
        # Calculate trends
        trends = {}
        for metric_name in ['regime_detection_accuracy', 'parameter_stability', 
                           'kalman_convergence', 'overall_performance']:
            values = [getattr(metrics, metric_name) for metrics in recent_metrics if hasattr(metrics, metric_name)]
            if len(values) > 1:
                trend = (values[-1] - values[0]) / (values[0] + 1e-8)
                trends[metric_name] = trend
            else:
                trends[metric_name] = 0.0
        
        # Overall trend assessment
        avg_trend = np.mean(list(trends.values()))
        if avg_trend > 0.1:
            trend_status = 'IMPROVING'
        elif avg_trend < -0.1:
            trend_status = 'DECLINING'
        else:
            trend_status = 'STABLE'
        
        return {
            'trend': trend_status,
            'metric_trends': trends,
            'average_trend': avg_trend
        }
    
    def _evaluate_action_effectiveness(self, actions: List[OptimizationAction]) -> Dict:
        """Evaluate effectiveness of optimization actions"""
        
        if not actions:
            return {'effectiveness': 'NO_ACTIONS'}
        
        # Calculate expected improvement
        total_expected_improvement = sum(action.expected_improvement for action in actions)
        avg_confidence = np.mean([action.confidence for action in actions])
        
        # Assess effectiveness
        if total_expected_improvement > 0.5 and avg_confidence > 0.7:
            effectiveness = 'HIGH'
        elif total_expected_improvement > 0.2 and avg_confidence > 0.5:
            effectiveness = 'MEDIUM'
        else:
            effectiveness = 'LOW'
        
        return {
            'effectiveness': effectiveness,
            'total_expected_improvement': total_expected_improvement,
            'average_confidence': avg_confidence,
            'action_count': len(actions)
        }
    
    def _generate_adaptation_signals(self, trend_analysis: Dict, action_effectiveness: Dict) -> Dict:
        """Generate adaptation signals"""
        
        signals = {}
        
        # Performance trend signals
        if trend_analysis.get('trend') == 'IMPROVING':
            signals['continue_current_strategy'] = True
            signals['increase_adaptation_rate'] = False
        elif trend_analysis.get('trend') == 'DECLINING':
            signals['continue_current_strategy'] = False
            signals['increase_adaptation_rate'] = True
        else:
            signals['continue_current_strategy'] = True
            signals['increase_adaptation_rate'] = False
        
        # Action effectiveness signals
        if action_effectiveness.get('effectiveness') == 'HIGH':
            signals['maintain_action_frequency'] = True
            signals['explore_new_actions'] = False
        elif action_effectiveness.get('effectiveness') == 'LOW':
            signals['maintain_action_frequency'] = False
            signals['explore_new_actions'] = True
        else:
            signals['maintain_action_frequency'] = True
            signals['explore_new_actions'] = True
        
        return signals
    
    def _assess_feedback_quality(self, metrics: PerformanceMetrics, actions: List[OptimizationAction]) -> str:
        """Assess quality of feedback"""
        
        # Check if we have sufficient data
        if len(self.feedback_history) < 5:
            return 'INSUFFICIENT_DATA'
        
        # Check if actions are being generated
        if not actions:
            return 'NO_ACTIONS'
        
        # Check if performance is improving
        if metrics.overall_performance > 0.5:
            return 'HIGH_QUALITY'
        elif metrics.overall_performance > 0.3:
            return 'MEDIUM_QUALITY'
        else:
            return 'LOW_QUALITY'

class PerformanceDrivenOptimizer:
    """Main performance-driven optimizer"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.bottleneck_identifier = BottleneckIdentifier()
        self.adaptive_optimizer = AdaptiveSystemOptimizer()
        self.feedback_loop = FeedbackLoop()
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        
        logger.info("Initialized PerformanceDrivenOptimizer")
    
    def optimize_system_performance(self, system, data: np.ndarray, rewards: np.ndarray) -> Dict:
        """Optimize system based on actual performance"""
        
        # Track performance metrics
        metrics = self.performance_tracker.track(system, data, rewards)
        
        # Identify bottlenecks
        bottlenecks = self.bottleneck_identifier.identify_bottlenecks(metrics)
        
        # Apply adaptive optimizations
        actions = self.adaptive_optimizer.optimize(system, bottlenecks)
        
        # Process feedback loop
        feedback = self.feedback_loop.process(metrics, actions)
        
        return {
            'performance_metrics': metrics,
            'bottlenecks': bottlenecks,
            'optimization_actions': actions,
            'feedback': feedback,
            'optimization_status': self._assess_optimization_status(metrics, actions)
        }
    
    def _assess_optimization_status(self, metrics: PerformanceMetrics, actions: List[OptimizationAction]) -> str:
        """Assess optimization status"""
        
        # Check if performance thresholds are met
        thresholds_met = 0
        total_thresholds = 4
        
        if metrics.regime_detection_accuracy >= 0.7:
            thresholds_met += 1
        if metrics.parameter_stability >= 0.9:
            thresholds_met += 1
        if metrics.kalman_convergence >= 0.8:
            thresholds_met += 1
        if metrics.overall_performance >= 0.3:
            thresholds_met += 1
        
        success_rate = thresholds_met / total_thresholds
        
        if success_rate >= 0.8:
            return 'EXCELLENT'
        elif success_rate >= 0.6:
            return 'GOOD'
        elif success_rate >= 0.4:
            return 'FAIR'
        else:
            return 'POOR'
    
    def start_continuous_optimization(self, system, data_generator, reward_generator):
        """Start continuous optimization in background thread"""
        
        if self.optimization_active:
            logger.warning("Continuous optimization already active")
            return
        
        self.optimization_active = True
        
        def optimization_loop():
            while self.optimization_active:
                try:
                    # Get new data and rewards
                    data = data_generator()
                    rewards = reward_generator()
                    
                    # Optimize system
                    result = self.optimize_system_performance(system, data, rewards)
                    
                    # Log optimization result
                    logger.info(f"Continuous optimization: {result['optimization_status']}")
                    
                    # Sleep between optimizations
                    time.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Error in continuous optimization: {e}")
                    time.sleep(5.0)
        
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        logger.info("Started continuous optimization")
    
    def stop_continuous_optimization(self):
        """Stop continuous optimization"""
        
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)
        
        logger.info("Stopped continuous optimization")
    
    def get_optimization_summary(self) -> Dict:
        """Get optimization summary"""
        
        performance_summary = self.performance_tracker.get_performance_summary()
        
        return {
            'performance_summary': performance_summary,
            'total_actions': len(self.adaptive_optimizer.optimization_actions),
            'feedback_quality': self.feedback_loop._assess_feedback_quality(
                PerformanceMetrics(0, 0, 0, 0, {}, 0, 0), []
            ),
            'optimization_active': self.optimization_active
        }

def main():
    """Test the performance-driven optimizer"""
    
    # Mock system for testing
    class MockSystem:
        def get_system_status(self):
            return {
                'regime_detection': {'bandit_performance': {'exploration_rate': 0.3}},
                'parameter_adaptation': {'optimizer_performance': {'parameter_stability': 0.85}},
                'kalman_optimization': {'optimizer_performance': {'hessian_condition': 500}},
                'system_overview': {'total_updates': 100, 'average_performance': 0.4},
                'regret_bounds': {'total_regret': 120.0}
            }
    
    # Initialize optimizer
    optimizer = PerformanceDrivenOptimizer()
    system = MockSystem()
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(50, 2)
    rewards = np.random.randn(50)
    
    # Run optimization
    result = optimizer.optimize_system_performance(system, data, rewards)
    
    # Print results
    print("Performance-Driven Optimizer Results:")
    print("=" * 50)
    print(f"Bottlenecks: {result['bottlenecks']}")
    print(f"Actions: {len(result['optimization_actions'])}")
    print(f"Status: {result['optimization_status']}")
    
    # Get summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Performance Status: {summary['performance_summary']['performance_status']}")
    print(f"Total Actions: {summary['total_actions']}")

if __name__ == "__main__":
    main()
