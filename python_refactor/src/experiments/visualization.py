"""
Visualization Module for Portfolio Optimization Experiments

Provides comprehensive visualization capabilities for:
- Stochastic Pareto frontiers evolution
- Anticipative distributions and predictions
- Expected future hypervolume metrics
- Learning progress and adaptive rates
- Portfolio rebalancing decisions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StochasticParetoVisualizer:
    """Visualization tools for stochastic Pareto frontiers and anticipatory learning."""
    
    def __init__(self, output_dir: str = "experiments/visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Setup matplotlib and seaborn plotting styles."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set figure size and DPI for high-quality plots
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
    
    def plot_stochastic_pareto_evolution(self, stochastic_frontiers: List[Dict[str, Any]], 
                                       anticipative_frontiers: List[Dict[str, Any]],
                                       save_path: Optional[str] = None):
        """
        Plot evolution of stochastic Pareto frontiers over generations.
        
        Args:
            stochastic_frontiers: List of stochastic Pareto frontiers
            anticipative_frontiers: List of anticipative Pareto frontiers
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stochastic Pareto Frontier Evolution', fontsize=16, fontweight='bold')
        
        # Plot 1: Current vs Predicted ROI
        ax1 = axes[0, 0]
        generations = range(len(stochastic_frontiers))
        
        for gen_idx, frontier_data in enumerate(stochastic_frontiers):
            frontier = frontier_data['frontier']
            if frontier:
                rois = [point['roi'] for point in frontier]
                predicted_rois = [point['roi_prediction'] for point in frontier]
                
                ax1.scatter(rois, predicted_rois, alpha=0.6, 
                           label=f'Gen {gen_idx}' if gen_idx % 10 == 0 else "")
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Prediction')
        ax1.set_xlabel('Current ROI')
        ax1.set_ylabel('Predicted ROI')
        ax1.set_title('Current vs Predicted ROI')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Risk Uncertainty Evolution
        ax2 = axes[0, 1]
        risk_uncertainties = []
        for frontier_data in stochastic_frontiers:
            frontier = frontier_data['frontier']
            if frontier:
                avg_risk_var = np.mean([point['risk_variance'] for point in frontier])
                risk_uncertainties.append(avg_risk_var)
        
        ax2.plot(generations[:len(risk_uncertainties)], risk_uncertainties, 'b-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Average Risk Variance')
        ax2.set_title('Risk Uncertainty Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Learning Rate Evolution
        ax3 = axes[1, 0]
        learning_rates = []
        for frontier_data in stochastic_frontiers:
            frontier = frontier_data['frontier']
            if frontier:
                avg_alpha = np.mean([point['alpha'] for point in frontier])
                learning_rates.append(avg_alpha)
        
        ax3.plot(generations[:len(learning_rates)], learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Learning Rate (α)')
        ax3.set_title('Adaptive Learning Rate Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Prediction Error Evolution
        ax4 = axes[1, 1]
        prediction_errors = []
        for frontier_data in stochastic_frontiers:
            frontier = frontier_data['frontier']
            if frontier:
                avg_error = np.mean([point['prediction_error'] for point in frontier])
                prediction_errors.append(avg_error)
        
        ax4.plot(generations[:len(prediction_errors)], prediction_errors, 'r-', linewidth=2)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('Average Prediction Error')
        ax4.set_title('Prediction Error Evolution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_anticipative_distributions(self, anticipative_frontiers: List[Dict[str, Any]], 
                                      generation_idx: int = -1, save_path: Optional[str] = None):
        """
        Plot anticipative distributions for a specific generation.
        
        Args:
            anticipative_frontiers: List of anticipative Pareto frontiers
            generation_idx: Generation index to plot (-1 for latest)
            save_path: Path to save the plot
        """
        if not anticipative_frontiers:
            print("No anticipative frontiers data available.")
            return
        
        frontier_data = anticipative_frontiers[generation_idx]
        frontier = frontier_data['frontier']
        
        if not frontier:
            print("No frontier data available for the specified generation.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Anticipative Distributions - Generation {frontier_data["current_time"]}', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        rois = [point['roi'] for point in frontier]
        risks = [point['risk'] for point in frontier]
        roi_vars = [point['roi_variance'] for point in frontier]
        risk_vars = [point['risk_variance'] for point in frontier]
        covariances = [point['covariance'] for point in frontier]
        
        # Plot 1: ROI vs Risk with uncertainty ellipses
        ax1 = axes[0, 0]
        scatter = ax1.scatter(rois, risks, c=roi_vars, cmap='viridis', alpha=0.7, s=100)
        ax1.set_xlabel('ROI')
        ax1.set_ylabel('Risk')
        ax1.set_title('ROI vs Risk with Uncertainty')
        plt.colorbar(scatter, ax=ax1, label='ROI Variance')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Variance evolution
        ax2 = axes[0, 1]
        ax2.scatter(roi_vars, risk_vars, alpha=0.7, s=100)
        ax2.set_xlabel('ROI Variance')
        ax2.set_ylabel('Risk Variance')
        ax2.set_title('Variance Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Covariance distribution
        ax3 = axes[1, 0]
        ax3.hist(covariances, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('ROI-Risk Covariance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Covariance Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Uncertainty heatmap
        ax4 = axes[1, 1]
        uncertainty_matrix = np.array([[roi_vars[i], covariances[i]] for i in range(len(roi_vars))])
        if len(uncertainty_matrix) > 1:
            # Create a 2D histogram of uncertainty
            ax4.scatter(roi_vars, risk_vars, c=covariances, cmap='coolwarm', alpha=0.7, s=100)
            ax4.set_xlabel('ROI Variance')
            ax4.set_ylabel('Risk Variance')
            ax4.set_title('Uncertainty Heatmap')
            plt.colorbar(ax4.collections[0], ax=ax4, label='Covariance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_expected_future_hypervolume_evolution(self, hypervolume_history: List[float],
                                                 expected_future_hypervolume_history: List[float],
                                                 save_path: Optional[str] = None):
        """
        Plot evolution of expected future hypervolume metrics.
        
        Args:
            hypervolume_history: History of current hypervolume
            expected_future_hypervolume_history: History of expected future hypervolume
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Expected Future Hypervolume Evolution', fontsize=16, fontweight='bold')
        
        generations = range(len(hypervolume_history))
        
        # Plot 1: Current vs Expected Future Hypervolume
        ax1 = axes[0, 0]
        ax1.plot(generations, hypervolume_history, 'b-', linewidth=2, label='Current Hypervolume')
        if expected_future_hypervolume_history:
            ax1.plot(generations[:len(expected_future_hypervolume_history)], 
                    expected_future_hypervolume_history, 'r-', linewidth=2, 
                    label='Expected Future Hypervolume')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('Hypervolume Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Hypervolume Ratio (Future/Current)
        ax2 = axes[0, 1]
        if expected_future_hypervolume_history:
            ratios = [future / current if current > 0 else 0 
                     for future, current in zip(expected_future_hypervolume_history, 
                                              hypervolume_history[:len(expected_future_hypervolume_history)])]
            ax2.plot(generations[:len(ratios)], ratios, 'g-', linewidth=2)
            ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No Change')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Future/Current Hypervolume Ratio')
            ax2.set_title('Predictive Robustness Ratio')
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hypervolume Improvement
        ax3 = axes[1, 0]
        if expected_future_hypervolume_history:
            improvements = [future - current 
                          for future, current in zip(expected_future_hypervolume_history, 
                                                   hypervolume_history[:len(expected_future_hypervolume_history)])]
            ax3.plot(generations[:len(improvements)], improvements, 'm-', linewidth=2)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Hypervolume Improvement')
            ax3.set_title('Expected Future Improvement')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Improvement
        ax4 = axes[1, 1]
        if expected_future_hypervolume_history:
            cumulative_improvement = np.cumsum(improvements)
            ax4.plot(generations[:len(cumulative_improvement)], cumulative_improvement, 'c-', linewidth=2)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Cumulative Improvement')
            ax4.set_title('Cumulative Expected Improvement')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_progress(self, learning_history: List[Dict[str, Any]], 
                             save_path: Optional[str] = None):
        """
        Plot learning progress and adaptive rates.
        
        Args:
            learning_history: History of learning events
            save_path: Path to save the plot
        """
        if not learning_history:
            print("No learning history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Anticipatory Learning Progress', fontsize=16, fontweight='bold')
        
        # Extract data
        alphas = [entry['alpha'] for entry in learning_history]
        prediction_errors = [entry['prediction_error'] for entry in learning_history]
        state_qualities = [entry['state_quality'] for entry in learning_history]
        nd_probabilities = [entry['nd_probability'] for entry in learning_history]
        anticipative_confidences = [entry.get('anticipative_confidence', 0.0) for entry in learning_history]
        
        events = range(len(learning_history))
        
        # Plot 1: Learning Rate Evolution
        ax1 = axes[0, 0]
        ax1.plot(events, alphas, 'b-', linewidth=2)
        ax1.set_xlabel('Learning Event')
        ax1.set_ylabel('Learning Rate (α)')
        ax1.set_title('Adaptive Learning Rate Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error vs State Quality
        ax2 = axes[0, 1]
        scatter = ax2.scatter(prediction_errors, state_qualities, c=alphas, cmap='viridis', alpha=0.7, s=100)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('State Quality')
        ax2.set_title('Error vs Quality with Learning Rate')
        plt.colorbar(scatter, ax=ax2, label='Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Non-dominance Probability Evolution
        ax3 = axes[1, 0]
        ax3.plot(events, nd_probabilities, 'g-', linewidth=2)
        ax3.set_xlabel('Learning Event')
        ax3.set_ylabel('Non-dominance Probability')
        ax3.set_title('Non-dominance Probability Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Anticipative Confidence
        ax4 = axes[1, 1]
        ax4.plot(events, anticipative_confidences, 'r-', linewidth=2)
        ax4.set_xlabel('Learning Event')
        ax4.set_ylabel('Anticipative Confidence')
        ax4.set_title('Anticipative Distribution Confidence')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_portfolio_rebalancing_analysis(self, stochastic_frontiers: List[Dict[str, Any]], 
                                          save_path: Optional[str] = None):
        """
        Plot portfolio rebalancing analysis.
        
        Args:
            stochastic_frontiers: List of stochastic Pareto frontiers
            save_path: Path to save the plot
        """
        if not stochastic_frontiers:
            print("No stochastic frontiers data available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Rebalancing Analysis', fontsize=16, fontweight='bold')
        
        # Extract weight data
        all_weights = []
        generations = []
        
        for gen_idx, frontier_data in enumerate(stochastic_frontiers):
            frontier = frontier_data['frontier']
            if frontier:
                for point in frontier:
                    all_weights.append(point['weights'])
                    generations.append(gen_idx)
        
        if not all_weights:
            print("No weight data available.")
            return
        
        all_weights = np.array(all_weights)
        generations = np.array(generations)
        
        # Plot 1: Weight evolution over generations
        ax1 = axes[0, 0]
        for asset_idx in range(min(5, all_weights.shape[1])):  # Plot first 5 assets
            asset_weights = all_weights[:, asset_idx]
            ax1.plot(generations, asset_weights, alpha=0.7, label=f'Asset {asset_idx}')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Portfolio Weight')
        ax1.set_title('Portfolio Weight Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Weight concentration over time
        ax2 = axes[0, 1]
        concentrations = []
        for weights in all_weights:
            concentration = np.sum(weights ** 2)  # Herfindahl index
            concentrations.append(concentration)
        
        ax2.plot(generations, concentrations, 'r-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Portfolio Concentration')
        ax2.set_title('Portfolio Concentration Evolution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Weight variance over time
        ax3 = axes[1, 0]
        weight_variances = np.var(all_weights, axis=1)
        ax3.plot(generations, weight_variances, 'g-', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Weight Variance')
        ax3.set_title('Portfolio Weight Variance')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Weight correlation matrix (latest generation)
        ax4 = axes[1, 1]
        latest_weights = all_weights[generations == max(generations)]
        if len(latest_weights) > 1:
            correlation_matrix = np.corrcoef(latest_weights.T)
            im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_title('Latest Generation Weight Correlations')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, experiment_results: Dict[str, Any], 
                                  save_dir: str = "experiments/visualizations"):
        """
        Create a comprehensive visualization report.
        
        Args:
            experiment_results: Complete experiment results
            save_dir: Directory to save visualizations
        """
        print("Creating comprehensive visualization report...")
        
        # Extract data from results
        stochastic_frontiers = experiment_results.get('stochastic_pareto_frontiers', [])
        anticipative_frontiers = experiment_results.get('anticipative_pareto_frontiers', [])
        hypervolume_history = experiment_results.get('hypervolume_history', [])
        expected_future_hypervolume_history = experiment_results.get('expected_future_hypervolume_history', [])
        learning_history = experiment_results.get('learning_history', [])
        
        # Create visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Stochastic Pareto Evolution
        if stochastic_frontiers and anticipative_frontiers:
            self.plot_stochastic_pareto_evolution(
                stochastic_frontiers, anticipative_frontiers,
                save_path=f"{save_dir}/stochastic_pareto_evolution_{timestamp}.png"
            )
        
        # 2. Anticipative Distributions
        if anticipative_frontiers:
            self.plot_anticipative_distributions(
                anticipative_frontiers,
                save_path=f"{save_dir}/anticipative_distributions_{timestamp}.png"
            )
        
        # 3. Expected Future Hypervolume
        if hypervolume_history and expected_future_hypervolume_history:
            self.plot_expected_future_hypervolume_evolution(
                hypervolume_history, expected_future_hypervolume_history,
                save_path=f"{save_dir}/expected_future_hypervolume_{timestamp}.png"
            )
        
        # 4. Learning Progress
        if learning_history:
            self.plot_learning_progress(
                learning_history,
                save_path=f"{save_dir}/learning_progress_{timestamp}.png"
            )
        
        # 5. Portfolio Rebalancing
        if stochastic_frontiers:
            self.plot_portfolio_rebalancing_analysis(
                stochastic_frontiers,
                save_path=f"{save_dir}/portfolio_rebalancing_{timestamp}.png"
            )
        
        print(f"Visualization report saved to {save_dir}")
    
    def plot_3d_stochastic_frontier(self, stochastic_frontiers: List[Dict[str, Any]], 
                                   generation_idx: int = -1, save_path: Optional[str] = None):
        """
        Create 3D visualization of stochastic Pareto frontier.
        
        Args:
            stochastic_frontiers: List of stochastic Pareto frontiers
            generation_idx: Generation index to plot (-1 for latest)
            save_path: Path to save the plot
        """
        if not stochastic_frontiers:
            print("No stochastic frontiers data available.")
            return
        
        frontier_data = stochastic_frontiers[generation_idx]
        frontier = frontier_data['frontier']
        
        if not frontier:
            print("No frontier data available for the specified generation.")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data
        rois = [point['roi'] for point in frontier]
        risks = [point['risk'] for point in frontier]
        roi_vars = [point['roi_variance'] for point in frontier]
        
        # Create 3D scatter plot
        scatter = ax.scatter(rois, risks, roi_vars, c=roi_vars, cmap='viridis', alpha=0.7, s=100)
        
        ax.set_xlabel('ROI')
        ax.set_ylabel('Risk')
        ax.set_zlabel('ROI Variance')
        ax.set_title(f'3D Stochastic Pareto Frontier - Generation {frontier_data["current_time"]}')
        
        plt.colorbar(scatter, ax=ax, label='ROI Variance')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 