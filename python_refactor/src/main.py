#!/usr/bin/env python3
"""
Main execution script for portfolio optimization.

This script demonstrates the integration of all components:
- Asset data loading
- Portfolio calculations
- NSGA-II optimization
- Results analysis
"""

import sys
import argparse
import numpy as np
from typing import List

from .portfolio.portfolio import Portfolio
from .algorithms.nsga2 import run_nsga2, get_pareto_front, evaluate_population_statistics
from .algorithms.sms_emoa import run_sms_emoa, get_sms_emoa_pareto_front, evaluate_sms_emoa_statistics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization using NSGA-II')
    
    parser.add_argument('--market', type=str, default='ftse', 
                       help='Market identifier')
    parser.add_argument('--num_assets', type=int, default=5,
                       help='Number of assets to optimize')
    parser.add_argument('--algorithm', type=str, default='NSGA2',
                       choices=['NSGA2', 'SMS_EMOA'],
                       help='Optimization algorithm')
    parser.add_argument('--regularization', type=str, default='L1',
                       choices=['L1', 'L2'],
                       help='Regularization type')
    parser.add_argument('--robustness', type=int, default=0,
                       choices=[0, 1],
                       help='Use robust methods (0=no, 1=yes)')
    parser.add_argument('--max_cardinality', type=int, default=3,
                       help='Maximum number of assets in portfolio')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2020-12-31',
                       help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--training_percentage', type=float, default=0.7,
                       help='Percentage of data for training')
    parser.add_argument('--window_size', type=int, default=20,
                       help='Window size for calculations')
    parser.add_argument('--population_size', type=int, default=50,
                       help='Population size for genetic algorithm')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations')
    parser.add_argument('--mutation_rate', type=float, default=0.1,
                       help='Mutation rate')
    parser.add_argument('--crossover_rate', type=float, default=0.9,
                       help='Crossover rate')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--data_dir', type=str, default='../data/ftse-original',
                       help='Directory containing CSV data files')
    
    return parser.parse_args()


def load_sample_data(num_assets: int, start_date: str, end_date: str, 
                    training_percentage: float) -> np.ndarray:
    """
    Load sample data for demonstration.
    
    In a real implementation, this would load actual CSV files.
    """
    print(f"Loading sample data for {num_assets} assets...")
    
    # Generate synthetic returns data
    np.random.seed(42)
    n_days = 252  # Trading days in a year
    returns_data = np.random.normal(0.001, 0.02, (n_days, num_assets))
    
    # Add some correlation structure
    correlation_matrix = np.eye(num_assets) * 0.8 + np.ones((num_assets, num_assets)) * 0.2
    correlation_matrix = correlation_matrix / correlation_matrix.sum(axis=1, keepdims=True)
    
    # Apply correlation
    returns_data = returns_data @ correlation_matrix.T
    
    return returns_data


def setup_portfolio_data(returns_data: np.ndarray, args):
    """Setup portfolio static data."""
    print("Setting up portfolio data...")
    
    # Set portfolio parameters
    Portfolio.available_assets_size = args.num_assets
    Portfolio.window_size = args.window_size
    Portfolio.max_cardinality = args.max_cardinality
    Portfolio.robustness = bool(args.robustness)
    
    # Compute statistics
    Portfolio.compute_statistics(returns_data)
    
    print(f"Portfolio setup complete:")
    print(f"  - Assets: {Portfolio.available_assets_size}")
    print(f"  - Window size: {Portfolio.window_size}")
    print(f"  - Max cardinality: {Portfolio.max_cardinality}")
    print(f"  - Robustness: {Portfolio.robustness}")


def run_optimization(args) -> List:
    """Run the optimization algorithm."""
    print(f"\nRunning {args.algorithm} optimization...")
    print(f"  - Population size: {args.population_size}")
    print(f"  - Generations: {args.generations}")
    print(f"  - Mutation rate: {args.mutation_rate}")
    print(f"  - Crossover rate: {args.crossover_rate}")
    
    if args.algorithm == 'NSGA2':
        # Run NSGA-II
        population = run_nsga2(
            num_generations=args.generations,
            population_size=args.population_size,
            num_assets=args.num_assets,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            random_seed=args.random_seed
        )
    elif args.algorithm == 'SMS_EMOA':
        # Run SMS-EMOA
        from .algorithms.solution import Solution
        initial_population = [Solution(args.num_assets) for _ in range(args.population_size)]
        
        population = run_sms_emoa(
            initial_population=initial_population,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            tournament_size=2,
            reference_point=(-1.0, 10.0)
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    return population


def analyze_results(population: List, args):
    """Analyze and display optimization results."""
    print(f"\nAnalyzing results...")
    
    if args.algorithm == 'NSGA2':
        # Get Pareto front
        pareto_front = get_pareto_front(population)
        
        # Calculate statistics
        stats = evaluate_population_statistics(population)
        
        # Display results
        print(f"\nOptimization Results:")
        print(f"  - Total population: {stats['population_size']}")
        print(f"  - Pareto front size: {stats['pareto_front_size']}")
        print(f"  - ROI range: {stats['roi_min']:.4f} - {stats['roi_max']:.4f}")
        print(f"  - Risk range: {stats['risk_min']:.4f} - {stats['risk_max']:.4f}")
        print(f"  - Average cardinality: {stats['cardinality_mean']:.2f}")
        
        # Display Pareto front solutions
        print(f"\nPareto Front Solutions:")
        print(f"{'Rank':<4} {'ROI':<8} {'Risk':<8} {'Cardinality':<12} {'Crowding':<10}")
        print("-" * 50)
        
        for i, solution in enumerate(pareto_front[:10]):  # Show first 10
            print(f"{i+1:<4} {solution.P.ROI:<8.4f} {solution.P.risk:<8.4f} "
                  f"{solution.P.cardinality:<12.1f} {solution.cd:<10.4f}")
    
    elif args.algorithm == 'SMS_EMOA':
        # Get Pareto front
        pareto_front = get_sms_emoa_pareto_front(population)
        
        # Calculate statistics
        stats = evaluate_sms_emoa_statistics(population)
        
        # Display results
        print(f"\nOptimization Results:")
        print(f"  - Total population: {stats['population_size']}")
        print(f"  - Pareto front size: {stats['pareto_front_size']}")
        print(f"  - Hypervolume: {stats['hypervolume']:.6f}")
        print(f"  - Mean ROI: {stats['mean_roi']:.4f}")
        print(f"  - Mean Risk: {stats['mean_risk']:.4f}")
        print(f"  - Average cardinality: {stats['mean_cardinality']:.2f}")
        
        # Display Pareto front solutions
        print(f"\nPareto Front Solutions:")
        print(f"{'Rank':<4} {'ROI':<8} {'Risk':<8} {'Cardinality':<12} {'Delta-S':<10}")
        print("-" * 50)
        
        for i, solution in enumerate(pareto_front[:10]):  # Show first 10
            print(f"{i+1:<4} {solution.P.ROI:<8.4f} {solution.P.risk:<8.4f} "
                  f"{solution.P.cardinality:<12.1f} {solution.Delta_S:<10.4f}")


def save_results(population: List, args, output_file: str = "optimization_results.txt"):
    """Save optimization results to file."""
    print(f"\nSaving results to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("Portfolio Optimization Results\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Parameters:\n")
        f.write(f"  Market: {args.market}\n")
        f.write(f"  Algorithm: {args.algorithm}\n")
        f.write(f"  Assets: {args.num_assets}\n")
        f.write(f"  Population: {args.population_size}\n")
        f.write(f"  Generations: {args.generations}\n")
        f.write(f"  Robustness: {args.robustness}\n\n")
        
        # Save Pareto front
        if args.algorithm == 'NSGA2':
            pareto_front = get_pareto_front(population)
        elif args.algorithm == 'SMS_EMOA':
            pareto_front = get_sms_emoa_pareto_front(population)
        
        f.write(f"Pareto Front ({len(pareto_front)} solutions):\n")
        f.write(f"{'ROI':<8} {'Risk':<8} {'Cardinality':<12} {'Weights'}\n")
        f.write("-" * 50 + "\n")
        
        for solution in pareto_front:
            weights_str = " ".join([f"{w:.3f}" for w in solution.P.investment])
            f.write(f"{solution.P.ROI:<8.4f} {solution.P.risk:<8.4f} "
                   f"{solution.P.cardinality:<12.1f} [{weights_str}]\n")


def main():
    """Main execution function."""
    print("Portfolio Optimization using NSGA-II")
    print("=" * 40)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load data
        returns_data = load_sample_data(
            args.num_assets, args.start_date, args.end_date, args.training_percentage
        )
        
        # Setup portfolio
        setup_portfolio_data(returns_data, args)
        
        # Run optimization
        population = run_optimization(args)
        
        # Analyze results
        analyze_results(population, args)
        
        # Save results
        save_results(population, args)
        
        print(f"\nOptimization completed successfully!")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 