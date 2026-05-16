#!/usr/bin/env python3
"""
Portfolio Optimization Experiment Runner

Main script to run comprehensive portfolio optimization experiments
with anticipatory learning and extended FTSE data.
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.experiments import ExperimentManager

def create_baseline_experiments():
    """Create baseline experiments without anticipatory learning."""
    experiments = []
    
    # NSGA-II Baseline Experiments
    experiments.extend([
        {
            "name": "NSGA2_Baseline_Original_2012",
            "description": "NSGA-II baseline with original 2012 data",
            "data": {
                "asset_files": ["../legacy-cpp/executable/data/ftse-original/table (0).csv"],
                "date_range": {"start": "2012-11-20", "end": "2012-11-21"},
                "assets": ["Asset_0"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 50,
                    "generations": 100,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "NSGA2_Baseline_Extended_2024",
            "description": "NSGA-II baseline with extended 2024 data",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    # SMS-EMOA Baseline Experiments
    experiments.extend([
        {
            "name": "SMSEMOA_Baseline_Original_2012",
            "description": "SMS-EMOA baseline with original 2012 data",
            "data": {
                "asset_files": ["../legacy-cpp/executable/data/ftse-original/table (0).csv"],
                "date_range": {"start": "2012-11-20", "end": "2012-11-21"},
                "assets": ["Asset_0"]
            },
            "algorithm": {
                "name": "sms_emoa",
                "parameters": {
                    "population_size": 50,
                    "generations": 100,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "SMSEMOA_Baseline_Extended_2024",
            "description": "SMS-EMOA baseline with extended 2024 data",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "sms_emoa",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    return experiments

def create_learning_experiments():
    """Create experiments with anticipatory learning."""
    experiments = []
    
    # NSGA-II with Learning
    experiments.extend([
        {
            "name": "NSGA2_Learning_Single_Extended_2024",
            "description": "NSGA-II with single solution learning",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "single_solution"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "NSGA2_Learning_Population_Extended_2024",
            "description": "NSGA-II with population learning",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "population"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    # SMS-EMOA with Learning
    experiments.extend([
        {
            "name": "SMSEMOA_Learning_Single_Extended_2024",
            "description": "SMS-EMOA with single solution learning",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "sms_emoa",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "single_solution"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "SMSEMOA_Learning_Population_Extended_2024",
            "description": "SMS-EMOA with population learning",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2012-11-21", "end": "2024-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "sms_emoa",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "population"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    return experiments

def create_market_condition_experiments():
    """Create experiments for different market conditions."""
    experiments = []
    
    # Bull Market (2017-2019)
    experiments.extend([
        {
            "name": "NSGA2_Bull_Market_2017_2019",
            "description": "NSGA-II in bull market conditions",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2017-01-01", "end": "2019-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "NSGA2_Learning_Bull_Market_2017_2019",
            "description": "NSGA-II with learning in bull market conditions",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2017-01-01", "end": "2019-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "population"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    # Bear Market (2020-2022)
    experiments.extend([
        {
            "name": "NSGA2_Bear_Market_2020_2022",
            "description": "NSGA-II in bear market conditions",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2020-01-01", "end": "2022-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {"enabled": False},
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        },
        {
            "name": "NSGA2_Learning_Bear_Market_2020_2022",
            "description": "NSGA-II with learning in bear market conditions",
            "data": {
                "asset_files": ["data/ftse-updated/FTSE_100_20121121_20241231.csv"],
                "date_range": {"start": "2020-01-01", "end": "2022-12-31"},
                "assets": ["FTSE_100"]
            },
            "algorithm": {
                "name": "nsga2",
                "parameters": {
                    "population_size": 100,
                    "generations": 200,
                    "crossover_rate": 0.9,
                    "mutation_rate": 0.1
                }
            },
            "learning": {
                "enabled": True,
                "parameters": {
                    "learning_rate": 0.01,
                    "prediction_horizon": 30,
                    "monte_carlo_simulations": 1000,
                    "state_observation_frequency": 10,
                    "learning_type": "population"
                }
            },
            "portfolio_selection": "hypervolume",
            "evaluation_period": "full"
        }
    ])
    
    return experiments

def main():
    """Main function to run the experiments."""
    print("=" * 80)
    print("PORTFOLIO OPTIMIZATION EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Create experiment manager
    config = {
        "experiment_name": "anticipatory_learning_portfolio_optimization",
        "description": "Comprehensive portfolio optimization with anticipatory learning",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }
    
    manager = ExperimentManager(config)
    
    try:
        # Create experiment suites
        print("\n📋 Creating experiment configurations...")
        
        # Baseline experiments
        baseline_experiments = create_baseline_experiments()
        baseline_suite = {
            "name": "Baseline_Experiments",
            "description": "Baseline experiments without anticipatory learning",
            "experiments": baseline_experiments
        }
        
        # Learning experiments
        learning_experiments = create_learning_experiments()
        learning_suite = {
            "name": "Learning_Experiments", 
            "description": "Experiments with anticipatory learning",
            "experiments": learning_experiments
        }
        
        # Market condition experiments
        market_experiments = create_market_condition_experiments()
        market_suite = {
            "name": "Market_Condition_Experiments",
            "description": "Experiments across different market conditions",
            "experiments": market_experiments
        }
        
        # Run baseline experiments
        print(f"\n🧪 Running Baseline Experiments ({len(baseline_experiments)} experiments)...")
        baseline_results = manager.run_experiment_suite(baseline_suite)
        
        # Run learning experiments
        print(f"\n🧠 Running Learning Experiments ({len(learning_experiments)} experiments)...")
        learning_results = manager.run_experiment_suite(learning_suite)
        
        # Run market condition experiments
        print(f"\n📈 Running Market Condition Experiments ({len(market_experiments)} experiments)...")
        market_results = manager.run_experiment_suite(market_suite)
        
        # Generate final summary
        print("\n📊 Generating Final Summary...")
        
        all_results = {
            "baseline": baseline_results,
            "learning": learning_results,
            "market_conditions": market_results,
            "summary": {
                "total_experiments": len(baseline_experiments) + len(learning_experiments) + len(market_experiments),
                "baseline_experiments": len(baseline_experiments),
                "learning_experiments": len(learning_experiments),
                "market_experiments": len(market_experiments),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Save final results
        results_dir = Path("experiments/final_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "final_experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n✅ All experiments completed successfully!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📊 Total experiments run: {all_results['summary']['total_experiments']}")
        
        # Print key findings
        print("\n🎯 KEY FINDINGS:")
        print("-" * 50)
        
        if baseline_results['experiments']:
            baseline_returns = [exp['final_metrics']['summary']['total_return'] 
                              for exp in baseline_results['experiments']]
            print(f"Baseline Average Return: {sum(baseline_returns)/len(baseline_returns):.4f}")
        
        if learning_results['experiments']:
            learning_returns = [exp['final_metrics']['summary']['total_return'] 
                              for exp in learning_results['experiments']]
            print(f"Learning Average Return: {sum(learning_returns)/len(learning_returns):.4f}")
        
        print(f"\n⏱️  Total execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error during experiment execution: {str(e)}")
        manager.logger.log_error(e)
        raise
    
    finally:
        # Cleanup
        manager.cleanup()

if __name__ == "__main__":
    main() 