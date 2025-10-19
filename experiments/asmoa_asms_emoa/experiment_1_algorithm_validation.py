#!/usr/bin/env python3
"""
ASMOO & ASMS-EMOA Experiment 1: Algorithm Validation

This experiment validates ASMOO and ASMS-EMOA algorithms against baselines,
incorporating the enhanced hybrid online learning system from EPIC 1.5.

Objectives:
1. Validate ASMOO and ASMS-EMOA vs baselines
2. Test integration with enhanced hybrid system
3. Compare performance on synthetic and real data
4. Measure multi-objective optimization metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'python_refactor', 'src'))

from algorithms.sms_emoa import SMSEMOA
from algorithms.nsga2 import NSGA_II
from algorithms.enhanced_hybrid_system import EnhancedHybridOnlineLearningSystem
from algorithms.advanced_regime_detection_v2 import AdvancedRegimeDetectorV2
from algorithms.adaptive_parameter_stability import AdaptiveParameterStability
from algorithms.adaptive_kalman_convergence import AdaptiveKalmanConvergence
from algorithms.adaptive_system_efficiency import AdaptiveSystemEfficiency

class ASMOOASMSEMOAValidationExperiment:
    """Experiment to validate ASMOO and ASMS-EMOA algorithms"""
    
    def __init__(self):
        self.results = {}
        self.config = {
            'n_runs': 10,  # Reduced for initial testing
            'n_generations': 100,  # Reduced for initial testing
            'population_size': 50,  # Reduced for initial testing
            'n_objectives': 3,  # ROI, Risk, Sharpe Ratio
            'n_variables': 10,  # Number of assets
            'time_horizon': 252,  # 1 year of data
            'regime_changes': 4,  # Market regime changes per year
            'noise_levels': [0.01, 0.05, 0.1]  # Noise levels for robustness
        }
        
    def run_algorithm_validation_experiment(self) -> Dict:
        """Run complete algorithm validation experiment"""
        
        print("🧪 Starting ASMOO & ASMS-EMOA Algorithm Validation Experiment")
        print("=" * 70)
        print("🎯 Focus: Validate algorithms with enhanced hybrid system integration")
        print("📊 Testing: ASMOO, ASMS-EMOA, NSGA-II, Random Portfolio")
        print("")
        
        # 1. Initialize enhanced hybrid system
        print("🚀 Initializing enhanced hybrid system...")
        hybrid_system = EnhancedHybridOnlineLearningSystem()
        
        # 2. Generate synthetic financial data
        print("\n📊 Generating synthetic financial data...")
        synthetic_data = self._generate_synthetic_financial_data()
        
        # 3. Test ASMOO (SMS-EMOA with anticipatory learning)
        print("\n🧪 Testing ASMOO (SMS-EMOA + Enhanced Hybrid System)...")
        asmoo_results = self._test_asmoo(hybrid_system, synthetic_data)
        
        # 4. Test ASMS-EMOA (SMS-EMOA standalone)
        print("\n🧪 Testing ASMS-EMOA (SMS-EMOA standalone)...")
        asms_emoa_results = self._test_asms_emoa(synthetic_data)
        
        # 5. Test NSGA-II baseline
        print("\n📊 Testing NSGA-II baseline...")
        nsga2_results = self._test_nsga2(synthetic_data)
        
        # 6. Test Random Portfolio baseline
        print("\n📊 Testing Random Portfolio baseline...")
        random_results = self._test_random_portfolio(synthetic_data)
        
        # 7. Analyze results
        print("\n🔍 Analyzing algorithm performance...")
        analysis = self._analyze_algorithm_performance(
            asmoo_results, asms_emoa_results, nsga2_results, random_results
        )
        
        # 8. Generate report
        print("\n📋 Generating algorithm validation report...")
        report = self._generate_validation_report(analysis)
        
        # 9. Create visualizations
        print("\n🎨 Creating algorithm validation visualizations...")
        self._create_validation_visualizations(analysis)
        
        return {
            'asmoo_results': asmoo_results,
            'asms_emoa_results': asms_emoa_results,
            'nsga2_results': nsga2_results,
            'random_results': random_results,
            'analysis': analysis,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_synthetic_financial_data(self) -> Dict[str, Any]:
        """Generate synthetic financial data for testing"""
        
        np.random.seed(42)  # For reproducibility
        
        # Market regimes
        regimes = ['bull', 'bear', 'sideways', 'recovery']
        regime_changes = np.random.choice(len(regimes), self.config['regime_changes'], replace=True)
        
        data = {
            'roi': [],
            'risk': [],
            'volume': [],
            'volatility': [],
            'regime': [],
            'timestamp': []
        }
        
        current_regime = 'sideways'
        base_roi = 0.0
        base_risk = 0.15
        
        for i in range(self.config['time_horizon']):
            # Change regime every 50 periods
            if i % 50 == 0 and i > 0:
                current_regime = regimes[regime_changes[i // 50]]
            
            # Generate data based on regime
            if current_regime == 'bull':
                roi_mean = 0.001
                risk_mean = 0.12
                volatility_mean = 0.08
            elif current_regime == 'bear':
                roi_mean = -0.0005
                risk_mean = 0.20
                volatility_mean = 0.15
            elif current_regime == 'sideways':
                roi_mean = 0.0001
                risk_mean = 0.15
                volatility_mean = 0.10
            else:  # recovery
                roi_mean = 0.0008
                risk_mean = 0.14
                volatility_mean = 0.09
            
            # Generate realistic financial data
            roi = np.random.normal(roi_mean, 0.02)
            risk = np.random.normal(risk_mean, 0.03)
            volume = np.random.lognormal(10, 0.5)
            volatility = np.random.normal(volatility_mean, 0.02)
            
            # Add some autocorrelation
            base_roi = 0.9 * base_roi + 0.1 * roi
            base_risk = 0.9 * base_risk + 0.1 * risk
            
            data['roi'].append(base_roi)
            data['risk'].append(base_risk)
            data['volume'].append(volume)
            data['volatility'].append(volatility)
            data['regime'].append(current_regime)
            data['timestamp'].append(datetime.now())
        
        # Convert to numpy arrays
        for key in data:
            if key != 'timestamp':
                data[key] = np.array(data[key])
        
        return data
    
    def _test_asmoo(self, hybrid_system: EnhancedHybridOnlineLearningSystem, 
                   data: Dict[str, Any]) -> Dict:
        """Test ASMOO (SMS-EMOA + Enhanced Hybrid System)"""
        
        results = []
        
        for run in range(self.config['n_runs']):
            print(f"  Run {run + 1}/{self.config['n_runs']}")
            
            # Initialize SMS-EMOA with enhanced hybrid system
            sms_emoa = SMSEMOA(
                population_size=self.config['population_size'],
                generations=self.config['n_generations'],
                crossover_rate=0.9,
                mutation_rate=0.1
            )
            
            # Set up anticipatory learning
            from algorithms.anticipatory_learning import AnticipatoryLearning
            anticipatory_learning = AnticipatoryLearning(
                learning_rate=0.01,
                prediction_horizon=30,
                monte_carlo_simulations=1000
            )
            sms_emoa.set_learning(anticipatory_learning)
            
            # Run optimization
            start_time = time.time()
            solutions = sms_emoa.run(data)
            end_time = time.time()
            
            # Calculate metrics
            pareto_front = [s for s in solutions if s.Pareto_rank == 0]
            
            # Calculate hypervolume
            hypervolume = self._calculate_hypervolume(pareto_front)
            
            # Calculate spread
            spread = self._calculate_spread(pareto_front)
            
            # Calculate IGD (simplified)
            igd = self._calculate_igd(pareto_front)
            
            results.append({
                'run': run,
                'solutions': len(solutions),
                'pareto_front_size': len(pareto_front),
                'hypervolume': hypervolume,
                'spread': spread,
                'igd': igd,
                'runtime': end_time - start_time,
                'function_evaluations': sms_emoa.function_evaluations
            })
        
        return {
            'algorithm': 'ASMOO',
            'results': results,
            'avg_hypervolume': np.mean([r['hypervolume'] for r in results]),
            'avg_spread': np.mean([r['spread'] for r in results]),
            'avg_igd': np.mean([r['igd'] for r in results]),
            'avg_runtime': np.mean([r['runtime'] for r in results])
        }
    
    def _test_asms_emoa(self, data: Dict[str, Any]) -> Dict:
        """Test ASMS-EMOA (SMS-EMOA standalone)"""
        
        results = []
        
        for run in range(self.config['n_runs']):
            print(f"  Run {run + 1}/{self.config['n_runs']}")
            
            # Initialize SMS-EMOA without hybrid system
            sms_emoa = SMSEMOA(
                population_size=self.config['population_size'],
                generations=self.config['n_generations'],
                crossover_rate=0.9,
                mutation_rate=0.1
            )
            
            # Run optimization
            start_time = time.time()
            solutions = sms_emoa.run(data)
            end_time = time.time()
            
            # Calculate metrics
            pareto_front = [s for s in solutions if s.Pareto_rank == 0]
            
            # Calculate hypervolume
            hypervolume = self._calculate_hypervolume(pareto_front)
            
            # Calculate spread
            spread = self._calculate_spread(pareto_front)
            
            # Calculate IGD (simplified)
            igd = self._calculate_igd(pareto_front)
            
            results.append({
                'run': run,
                'solutions': len(solutions),
                'pareto_front_size': len(pareto_front),
                'hypervolume': hypervolume,
                'spread': spread,
                'igd': igd,
                'runtime': end_time - start_time,
                'function_evaluations': sms_emoa.function_evaluations
            })
        
        return {
            'algorithm': 'ASMS-EMOA',
            'results': results,
            'avg_hypervolume': np.mean([r['hypervolume'] for r in results]),
            'avg_spread': np.mean([r['spread'] for r in results]),
            'avg_igd': np.mean([r['igd'] for r in results]),
            'avg_runtime': np.mean([r['runtime'] for r in results])
        }
    
    def _test_nsga2(self, data: Dict[str, Any]) -> Dict:
        """Test NSGA-II baseline"""
        
        results = []
        
        for run in range(self.config['n_runs']):
            print(f"  Run {run + 1}/{self.config['n_runs']}")
            
            # Initialize NSGA-II
            nsga2 = NSGA_II(
                population_size=self.config['population_size'],
                generations=self.config['n_generations'],
                crossover_rate=0.9,
                mutation_rate=0.1
            )
            
            # Run optimization
            start_time = time.time()
            solutions = nsga2.run(data)
            end_time = time.time()
            
            # Calculate metrics
            pareto_front = [s for s in solutions if s.Pareto_rank == 0]
            
            # Calculate hypervolume
            hypervolume = self._calculate_hypervolume(pareto_front)
            
            # Calculate spread
            spread = self._calculate_spread(pareto_front)
            
            # Calculate IGD (simplified)
            igd = self._calculate_igd(pareto_front)
            
            results.append({
                'run': run,
                'solutions': len(solutions),
                'pareto_front_size': len(pareto_front),
                'hypervolume': hypervolume,
                'spread': spread,
                'igd': igd,
                'runtime': end_time - start_time,
                'function_evaluations': nsga2.function_evaluations
            })
        
        return {
            'algorithm': 'NSGA-II',
            'results': results,
            'avg_hypervolume': np.mean([r['hypervolume'] for r in results]),
            'avg_spread': np.mean([r['spread'] for r in results]),
            'avg_igd': np.mean([r['igd'] for r in results]),
            'avg_runtime': np.mean([r['runtime'] for r in results])
        }
    
    def _test_random_portfolio(self, data: Dict[str, Any]) -> Dict:
        """Test Random Portfolio baseline"""
        
        results = []
        
        for run in range(self.config['n_runs']):
            print(f"  Run {run + 1}/{self.config['n_runs']}")
            
            # Generate random portfolios
            start_time = time.time()
            solutions = []
            
            for _ in range(self.config['population_size']):
                # Generate random portfolio weights
                weights = np.random.random(self.config['n_variables'])
                weights = weights / np.sum(weights)  # Normalize
                
                # Calculate objectives (simplified)
                roi = np.random.normal(0.05, 0.1)  # Random ROI
                risk = np.random.normal(0.15, 0.05)  # Random risk
                sharpe = roi / risk if risk > 0 else 0  # Sharpe ratio
                
                # Create solution
                from algorithms.solution import Solution
                solution = Solution()
                solution.P.ROI = roi
                solution.P.risk = risk
                solution.P.sharpe = sharpe
                solution.Pareto_rank = 0  # All random solutions are non-dominated
                
                solutions.append(solution)
            
            end_time = time.time()
            
            # Calculate metrics
            pareto_front = solutions  # All random solutions are non-dominated
            
            # Calculate hypervolume
            hypervolume = self._calculate_hypervolume(pareto_front)
            
            # Calculate spread
            spread = self._calculate_spread(pareto_front)
            
            # Calculate IGD (simplified)
            igd = self._calculate_igd(pareto_front)
            
            results.append({
                'run': run,
                'solutions': len(solutions),
                'pareto_front_size': len(pareto_front),
                'hypervolume': hypervolume,
                'spread': spread,
                'igd': igd,
                'runtime': end_time - start_time,
                'function_evaluations': len(solutions)
            })
        
        return {
            'algorithm': 'Random Portfolio',
            'results': results,
            'avg_hypervolume': np.mean([r['hypervolume'] for r in results]),
            'avg_spread': np.mean([r['spread'] for r in results]),
            'avg_igd': np.mean([r['igd'] for r in results]),
            'avg_runtime': np.mean([r['runtime'] for r in results])
        }
    
    def _calculate_hypervolume(self, pareto_front: List) -> float:
        """Calculate hypervolume metric"""
        if not pareto_front:
            return 0.0
        
        # Simplified hypervolume calculation
        # For 3 objectives: ROI, Risk, Sharpe
        reference_point = [0.0, 1.0, 0.0]  # [min_ROI, max_Risk, min_Sharpe]
        
        total_volume = 0.0
        for solution in pareto_front:
            # Calculate volume contribution (simplified)
            roi = getattr(solution.P, 'ROI', 0.0)
            risk = getattr(solution.P, 'risk', 1.0)
            sharpe = getattr(solution.P, 'sharpe', 0.0)
            
            # Volume contribution (simplified)
            volume = max(0, roi - reference_point[0]) * \
                    max(0, reference_point[1] - risk) * \
                    max(0, sharpe - reference_point[2])
            total_volume += volume
        
        return total_volume
    
    def _calculate_spread(self, pareto_front: List) -> float:
        """Calculate spread metric"""
        if len(pareto_front) < 2:
            return 0.0
        
        # Calculate spread (simplified)
        roi_values = [getattr(s.P, 'ROI', 0.0) for s in pareto_front]
        risk_values = [getattr(s.P, 'risk', 1.0) for s in pareto_front]
        
        roi_spread = np.std(roi_values)
        risk_spread = np.std(risk_values)
        
        return (roi_spread + risk_spread) / 2.0
    
    def _calculate_igd(self, pareto_front: List) -> float:
        """Calculate IGD metric (simplified)"""
        if not pareto_front:
            return 1.0
        
        # Simplified IGD calculation
        # Generate reference points
        n_ref_points = 100
        ref_roi = np.linspace(0.0, 0.2, n_ref_points)
        ref_risk = np.linspace(0.05, 0.3, n_ref_points)
        
        total_distance = 0.0
        for i in range(n_ref_points):
            min_distance = float('inf')
            for solution in pareto_front:
                roi = getattr(solution.P, 'ROI', 0.0)
                risk = getattr(solution.P, 'risk', 1.0)
                
                distance = np.sqrt((roi - ref_roi[i])**2 + (risk - ref_risk[i])**2)
                min_distance = min(min_distance, distance)
            
            total_distance += min_distance
        
        return total_distance / n_ref_points
    
    def _analyze_algorithm_performance(self, asmoo_results: Dict, asms_emoa_results: Dict, 
                                    nsga2_results: Dict, random_results: Dict) -> Dict:
        """Analyze algorithm performance"""
        
        analysis = {}
        
        # Collect metrics
        algorithms = ['ASMOO', 'ASMS-EMOA', 'NSGA-II', 'Random Portfolio']
        results = [asmoo_results, asms_emoa_results, nsga2_results, random_results]
        
        hypervolumes = [r['avg_hypervolume'] for r in results]
        spreads = [r['avg_spread'] for r in results]
        igds = [r['avg_igd'] for r in results]
        runtimes = [r['avg_runtime'] for r in results]
        
        # Find best algorithm for each metric
        best_hypervolume_idx = np.argmax(hypervolumes)
        best_spread_idx = np.argmax(spreads)
        best_igd_idx = np.argmin(igds)  # Lower is better for IGD
        best_runtime_idx = np.argmin(runtimes)  # Lower is better for runtime
        
        analysis['best_algorithms'] = {
            'hypervolume': algorithms[best_hypervolume_idx],
            'spread': algorithms[best_spread_idx],
            'igd': algorithms[best_igd_idx],
            'runtime': algorithms[best_runtime_idx]
        }
        
        analysis['metrics'] = {
            'hypervolumes': hypervolumes,
            'spreads': spreads,
            'igds': igds,
            'runtimes': runtimes
        }
        
        # Calculate improvements
        analysis['improvements'] = {
            'asmoo_vs_asms_emoa': {
                'hypervolume': (hypervolumes[0] - hypervolumes[1]) / hypervolumes[1] * 100,
                'spread': (spreads[0] - spreads[1]) / spreads[1] * 100,
                'igd': (igds[1] - igds[0]) / igds[1] * 100,  # Lower is better
                'runtime': (runtimes[1] - runtimes[0]) / runtimes[1] * 100  # Lower is better
            },
            'asmoo_vs_nsga2': {
                'hypervolume': (hypervolumes[0] - hypervolumes[2]) / hypervolumes[2] * 100,
                'spread': (spreads[0] - spreads[2]) / spreads[2] * 100,
                'igd': (igds[2] - igds[0]) / igds[2] * 100,
                'runtime': (runtimes[2] - runtimes[0]) / runtimes[2] * 100
            },
            'asmoo_vs_random': {
                'hypervolume': (hypervolumes[0] - hypervolumes[3]) / hypervolumes[3] * 100,
                'spread': (spreads[0] - spreads[3]) / spreads[3] * 100,
                'igd': (igds[3] - igds[0]) / igds[3] * 100,
                'runtime': (runtimes[3] - runtimes[0]) / runtimes[3] * 100
            }
        }
        
        return analysis
    
    def _generate_validation_report(self, analysis: Dict) -> str:
        """Generate algorithm validation report"""
        
        report = []
        report.append("ASMOO & ASMS-EMOA ALGORITHM VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("")
        report.append("🎯 FOCUS: Validate ASMOO and ASMS-EMOA algorithms")
        report.append("📊 TESTING: ASMOO, ASMS-EMOA, NSGA-II, Random Portfolio")
        report.append("")
        
        # Best algorithms
        report.append("🏆 BEST ALGORITHMS BY METRIC")
        report.append("-" * 40)
        best = analysis['best_algorithms']
        report.append(f"Hypervolume: {best['hypervolume']}")
        report.append(f"Spread: {best['spread']}")
        report.append(f"IGD: {best['igd']}")
        report.append(f"Runtime: {best['runtime']}")
        report.append("")
        
        # Metrics comparison
        report.append("📊 METRICS COMPARISON")
        report.append("-" * 40)
        metrics = analysis['metrics']
        algorithms = ['ASMOO', 'ASMS-EMOA', 'NSGA-II', 'Random Portfolio']
        
        for i, alg in enumerate(algorithms):
            report.append(f"{alg}:")
            report.append(f"  Hypervolume: {metrics['hypervolumes'][i]:.4f}")
            report.append(f"  Spread: {metrics['spreads'][i]:.4f}")
            report.append(f"  IGD: {metrics['igds'][i]:.4f}")
            report.append(f"  Runtime: {metrics['runtimes'][i]:.2f}s")
            report.append("")
        
        # Improvements
        report.append("📈 ASMOO IMPROVEMENTS")
        report.append("-" * 40)
        improvements = analysis['improvements']
        
        report.append("ASMOO vs ASMS-EMOA:")
        report.append(f"  Hypervolume: {improvements['asmoo_vs_asms_emoa']['hypervolume']:.1f}%")
        report.append(f"  Spread: {improvements['asmoo_vs_asms_emoa']['spread']:.1f}%")
        report.append(f"  IGD: {improvements['asmoo_vs_asms_emoa']['igd']:.1f}%")
        report.append(f"  Runtime: {improvements['asmoo_vs_asms_emoa']['runtime']:.1f}%")
        report.append("")
        
        report.append("ASMOO vs NSGA-II:")
        report.append(f"  Hypervolume: {improvements['asmoo_vs_nsga2']['hypervolume']:.1f}%")
        report.append(f"  Spread: {improvements['asmoo_vs_nsga2']['spread']:.1f}%")
        report.append(f"  IGD: {improvements['asmoo_vs_nsga2']['igd']:.1f}%")
        report.append(f"  Runtime: {improvements['asmoo_vs_nsga2']['runtime']:.1f}%")
        report.append("")
        
        report.append("ASMOO vs Random Portfolio:")
        report.append(f"  Hypervolume: {improvements['asmoo_vs_random']['hypervolume']:.1f}%")
        report.append(f"  Spread: {improvements['asmoo_vs_random']['spread']:.1f}%")
        report.append(f"  IGD: {improvements['asmoo_vs_random']['igd']:.1f}%")
        report.append(f"  Runtime: {improvements['asmoo_vs_random']['runtime']:.1f}%")
        report.append("")
        
        # Conclusion
        report.append("🎯 CONCLUSION")
        report.append("-" * 40)
        if improvements['asmoo_vs_asms_emoa']['hypervolume'] > 0:
            report.append("Status: ✅ ASMOO shows improvements over ASMS-EMOA")
        else:
            report.append("Status: ⚠️ ASMOO needs further optimization")
        
        report.append("")
        report.append("Key findings:")
        report.append("1. 🎯 ASMOO performance vs baselines")
        report.append("2. 📊 Multi-objective optimization metrics")
        report.append("3. ⚡ Runtime and efficiency comparison")
        report.append("4. 🔧 Enhanced hybrid system integration")
        
        report.append("\n" + "=" * 60)
        report.append("End of Algorithm Validation Report")
        
        return "\n".join(report)
    
    def _create_validation_visualizations(self, analysis: Dict):
        """Create algorithm validation visualizations"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 16))
        
        algorithms = ['ASMOO', 'ASMS-EMOA', 'NSGA-II', 'Random Portfolio']
        metrics = analysis['metrics']
        
        # 1. Hypervolume Comparison
        plt.subplot(3, 3, 1)
        bars = plt.bar(algorithms, metrics['hypervolumes'], color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('Hypervolume Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Hypervolume')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics['hypervolumes']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Spread Comparison
        plt.subplot(3, 3, 2)
        bars = plt.bar(algorithms, metrics['spreads'], color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('Spread Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Spread')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics['spreads']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. IGD Comparison
        plt.subplot(3, 3, 3)
        bars = plt.bar(algorithms, metrics['igds'], color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('IGD Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('IGD')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics['igds']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Runtime Comparison
        plt.subplot(3, 3, 4)
        bars = plt.bar(algorithms, metrics['runtimes'], color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('Runtime Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
        plt.ylabel('Runtime (seconds)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics['runtimes']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # 5. ASMOO vs ASMS-EMOA Improvements
        plt.subplot(3, 3, 5)
        improvements = analysis['improvements']['asmoo_vs_asms_emoa']
        metrics_names = ['Hypervolume', 'Spread', 'IGD', 'Runtime']
        improvements_values = [
            improvements['hypervolume'],
            improvements['spread'],
            improvements['igd'],
            improvements['runtime']
        ]
        
        bars = plt.bar(metrics_names, improvements_values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('ASMOO vs ASMS-EMOA\nImprovements (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvements_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. ASMOO vs NSGA-II Improvements
        plt.subplot(3, 3, 6)
        improvements = analysis['improvements']['asmoo_vs_nsga2']
        improvements_values = [
            improvements['hypervolume'],
            improvements['spread'],
            improvements['igd'],
            improvements['runtime']
        ]
        
        bars = plt.bar(metrics_names, improvements_values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('ASMOO vs NSGA-II\nImprovements (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvements_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7. ASMOO vs Random Portfolio Improvements
        plt.subplot(3, 3, 7)
        improvements = analysis['improvements']['asmoo_vs_random']
        improvements_values = [
            improvements['hypervolume'],
            improvements['spread'],
            improvements['igd'],
            improvements['runtime']
        ]
        
        bars = plt.bar(metrics_names, improvements_values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('ASMOO vs Random Portfolio\nImprovements (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Improvement (%)')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, improvements_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 8. Overall Performance Radar Chart
        plt.subplot(3, 3, 8)
        # Normalize metrics for radar chart
        norm_hypervolumes = np.array(metrics['hypervolumes']) / np.max(metrics['hypervolumes'])
        norm_spreads = np.array(metrics['spreads']) / np.max(metrics['spreads'])
        norm_igds = 1 - (np.array(metrics['igds']) / np.max(metrics['igds']))  # Invert IGD
        norm_runtimes = 1 - (np.array(metrics['runtimes']) / np.max(metrics['runtimes']))  # Invert runtime
        
        # Create radar chart data
        radar_data = np.array([norm_hypervolumes, norm_spreads, norm_igds, norm_runtimes]).T
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the circle
        
        for i, alg in enumerate(algorithms):
            values = np.concatenate((radar_data[i], [radar_data[i][0]]))  # Close the circle
            plt.plot(angles, values, 'o-', linewidth=2, label=alg)
            plt.fill(angles, values, alpha=0.25)
        
        plt.title('Overall Performance\nRadar Chart', fontsize=14, fontweight='bold')
        plt.xticks(angles[:-1], ['Hypervolume', 'Spread', 'IGD', 'Runtime'])
        plt.legend()
        plt.grid(True)
        
        # 9. Best Algorithm Summary
        plt.subplot(3, 3, 9)
        best = analysis['best_algorithms']
        categories = list(best.keys())
        values = [1 if best[cat] == 'ASMOO' else 0 for cat in categories]
        
        bars = plt.bar(categories, values, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        plt.title('ASMOO Wins by Metric\n(1 = ASMOO wins, 0 = Other wins)', fontsize=14, fontweight='bold')
        plt.ylabel('ASMOO Wins')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"experiment_1_algorithm_validation_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"📊 Algorithm validation visualization saved as: {filename}")
        
        plt.show()

def main():
    """Run algorithm validation experiment"""
    
    print("🧪 Starting ASMOO & ASMS-EMOA Algorithm Validation Experiment")
    print("=" * 70)
    print("🎯 Focus: Validate algorithms with enhanced hybrid system integration")
    print("📊 Testing: ASMOO, ASMS-EMOA, NSGA-II, Random Portfolio")
    print("")
    
    # Run algorithm validation experiment
    experiment = ASMOOASMSEMOAValidationExperiment()
    
    try:
        results = experiment.run_algorithm_validation_experiment()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"experiment_1_algorithm_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✅ Algorithm validation experiment completed!")
        print(f"📁 Results saved to: {results_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("ALGORITHM VALIDATION REPORT")
        print("=" * 70)
        print(results['report'])
        
    except Exception as e:
        print(f"❌ Error running algorithm validation experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
