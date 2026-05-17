"""
Portfolio Evaluator Module

Evaluates portfolio performance and calculates comprehensive metrics for
portfolio optimization experiments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PortfolioEvaluator:
    """Portfolio evaluator for performance assessment."""
    
    def __init__(self):
        """Initialize the portfolio evaluator."""
        self.evaluation_history = []
    
    def evaluate_portfolio(self, solution, data: Dict[str, Any], 
                         evaluation_period: str = 'full') -> Dict[str, Any]:
        """
        Evaluate portfolio performance for a given solution.
        
        Args:
            solution: Portfolio solution to evaluate
            data: Market data dictionary
            evaluation_period: Period for evaluation ('full', 'train', 'test')
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract portfolio weights
        weights = self._extract_weights(solution)
        
        # Get asset returns
        asset_returns = self._get_asset_returns(data)
        
        if asset_returns.empty:
            return self._empty_evaluation_result()
        
        # Filter by evaluation period
        filtered_returns = self._filter_by_period(asset_returns, evaluation_period)
        
        # Calculate portfolio performance
        performance = self._calculate_portfolio_performance(weights, filtered_returns)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(weights, filtered_returns)
        
        # Calculate additional metrics
        additional_metrics = self._calculate_additional_metrics(weights, filtered_returns)
        
        # Combine results
        results = {
            'weights': weights,
            'performance': performance,
            'risk_metrics': risk_metrics,
            'additional_metrics': additional_metrics,
            'evaluation_period': evaluation_period,
            'final_value': performance['final_value']
        }
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now().isoformat(),
            'weights': weights,
            'performance': performance,
            'evaluation_period': evaluation_period
        })
        
        return results
    
    def _extract_weights(self, solution) -> Dict[str, float]:
        """Extract portfolio weights from solution."""
        if hasattr(solution, 'P') and hasattr(solution.P, 'investment'):
            weights = solution.P.investment
            # Convert to dictionary with asset names
            asset_names = [f'Asset_{i}' for i in range(len(weights))]
            return dict(zip(asset_names, weights))
        else:
            # Default equal weights
            return {'Asset_0': 1.0}
    
    def _get_asset_returns(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Get asset returns from data.

        W11-2 (closes W10-CARRY-1): two-layer guard for the
        downstream `(1 + portfolio_returns).cumprod()` overflow:

        1. Price-level guard (defensive): if the input median |value|
           > 1.5, treat as price-level and convert via pct_change.
           The W9-2 data_loader pivot DOES return returns (not prices)
           for the standard path, but this protects callers that
           supply raw prices.

        2. Returns sanitation (the actual W10-CARRY-1 root cause):
           the real 98-CSV paper-window dataset contains Inf returns
           (pct_change crosses a zero-price quote — 2 occurrences in
           the FTSE archive) plus implausible outliers (97999, 42149,
           28927 — one-day "returns" from data-source quirks).
           Without sanitation, portfolio_value = Σ weight × Inf = Inf.
           Cap any |return| > 1.0 (100% daily move) at ±1.0 and replace
           Inf with 0.0 (no-information treatment).
        """
        if 'assets' in data and isinstance(data['assets'], pd.DataFrame):
            df = data['assets']
            if df.empty:
                return df
            # Layer 1: price-level detection.
            try:
                median_magnitude = df.abs().median().median()
            except Exception:
                median_magnitude = 0.0
            if median_magnitude > 1.5:
                df = df.pct_change().dropna(how='all').fillna(0.0)
            # Layer 2: returns sanitation — eliminate Inf and clip
            # implausible outliers. Operates on returns-shaped output
            # regardless of which path produced it.
            import numpy as _np
            df = df.replace([_np.inf, -_np.inf], 0.0)
            df = df.clip(lower=-1.0, upper=1.0)
            return df
        elif 'assets' in data and isinstance(data['assets'], dict):
            # Combine multiple asset dataframes
            combined_data = []
            for asset_name, asset_df in data['assets'].items():
                if not asset_df.empty and 'Return' in asset_df.columns:
                    asset_df_copy = asset_df.copy()
                    asset_df_copy.columns = [asset_name]
                    combined_data.append(asset_df_copy)
            
            if combined_data:
                return pd.concat(combined_data, axis=1)
        
        return pd.DataFrame()
    
    def _filter_by_period(self, returns: pd.DataFrame, period: str) -> pd.DataFrame:
        """Filter returns by evaluation period."""
        if returns.empty:
            return returns
        
        if period == 'full':
            return returns
        
        elif period == 'train':
            # Use first 70% of data
            split_idx = int(len(returns) * 0.7)
            return returns.iloc[:split_idx]
        
        elif period == 'test':
            # Use last 30% of data
            split_idx = int(len(returns) * 0.7)
            return returns.iloc[split_idx:]
        
        else:
            return returns
    
    def _calculate_portfolio_performance(self, weights: Dict[str, float], 
                                       returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        if returns.empty:
            return self._empty_performance_result()
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, returns)
        
        # Calculate cumulative performance
        cumulative_returns = (1 + portfolio_returns).cumprod()
        final_value = cumulative_returns.iloc[-1]
        
        # Calculate performance metrics
        total_return = final_value - 1
        annualized_return = self._calculate_annualized_return(portfolio_returns)
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # 2% annual risk-free rate
        excess_returns = portfolio_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'final_value': final_value,
            'portfolio_returns': portfolio_returns.tolist(),
            'cumulative_returns': cumulative_returns.tolist()
        }
    
    def _calculate_portfolio_returns(self, weights: Dict[str, float], 
                                   returns: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns."""
        if returns.empty:
            return pd.Series()
        
        # Align weights with returns columns
        aligned_weights = []
        for col in returns.columns:
            if col in weights:
                aligned_weights.append(weights[col])
            else:
                aligned_weights.append(0.0)
        
        # Ensure weights sum to 1
        weight_sum = sum(aligned_weights)
        if weight_sum > 0:
            aligned_weights = [w / weight_sum for w in aligned_weights]
        else:
            # Equal weights if no weights provided
            aligned_weights = [1.0 / len(returns.columns)] * len(returns.columns)
        
        # Calculate weighted returns
        portfolio_returns = returns.dot(aligned_weights)
        
        return portfolio_returns
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if returns.empty:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        num_years = len(returns) / 252  # Assuming daily data
        
        if num_years > 0:
            return (1 + total_return) ** (1 / num_years) - 1
        else:
            return 0.0
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if cumulative_returns.empty:
            return 0.0
        
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        return drawdown.min()
    
    def _calculate_risk_metrics(self, weights: Dict[str, float], 
                              returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics."""
        if returns.empty:
            return {}
        
        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(weights, returns)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Downside deviation
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        risk_free_rate = 0.02 / 252  # Daily risk-free rate
        excess_returns = portfolio_returns - risk_free_rate
        sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'downside_deviation': downside_deviation,
            'sortino_ratio': sortino_ratio
        }
    
    def _calculate_additional_metrics(self, weights: Dict[str, float], 
                                    returns: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional portfolio metrics."""
        if returns.empty:
            return {}
        
        # Portfolio concentration (Herfindahl index)
        weight_values = list(weights.values())
        concentration = sum(w**2 for w in weight_values)
        
        # Portfolio diversification
        diversification = 1 - concentration
        
        # Number of assets
        num_assets = len(weights)
        
        # Effective number of assets
        effective_assets = 1 / concentration if concentration > 0 else num_assets
        
        # Portfolio turnover (simplified)
        turnover = 0.0  # Would need historical weights to calculate properly
        
        return {
            'concentration': concentration,
            'diversification': diversification,
            'num_assets': num_assets,
            'effective_assets': effective_assets,
            'turnover': turnover
        }
    
    def _empty_evaluation_result(self) -> Dict[str, Any]:
        """Return empty evaluation result."""
        return {
            'weights': {},
            'performance': self._empty_performance_result(),
            'risk_metrics': {},
            'additional_metrics': {},
            'evaluation_period': 'full',
            'final_value': 1.0
        }
    
    def _empty_performance_result(self) -> Dict[str, float]:
        """Return empty performance result."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'final_value': 1.0,
            'portfolio_returns': [],
            'cumulative_returns': []
        }
    
    def compare_portfolios(self, portfolios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple portfolios.
        
        Args:
            portfolios: List of portfolio evaluation results
            
        Returns:
            Comparison results
        """
        if not portfolios:
            return {}
        
        comparison = {
            'total_return': [p['performance']['total_return'] for p in portfolios],
            'sharpe_ratio': [p['performance']['sharpe_ratio'] for p in portfolios],
            'max_drawdown': [p['performance']['max_drawdown'] for p in portfolios],
            'volatility': [p['performance']['volatility'] for p in portfolios],
            'calmar_ratio': [p['performance']['calmar_ratio'] for p in portfolios],
            'final_value': [p['performance']['final_value'] for p in portfolios]
        }
        
        # Calculate rankings
        rankings = {}
        for metric, values in comparison.items():
            if metric == 'max_drawdown':
                # Lower is better for drawdown
                rankings[metric] = np.argsort(values)
            else:
                # Higher is better for other metrics
                rankings[metric] = np.argsort(values)[::-1]
        
        comparison['rankings'] = rankings
        
        return comparison
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {}
        
        # Calculate summary statistics
        total_returns = [eval['performance']['total_return'] for eval in self.evaluation_history]
        sharpe_ratios = [eval['performance']['sharpe_ratio'] for eval in self.evaluation_history]
        max_drawdowns = [eval['performance']['max_drawdown'] for eval in self.evaluation_history]
        
        summary = {
            'total_evaluations': len(self.evaluation_history),
            'total_return': {
                'mean': np.mean(total_returns),
                'std': np.std(total_returns),
                'min': np.min(total_returns),
                'max': np.max(total_returns)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios)
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns)
            }
        }
        
        return summary 