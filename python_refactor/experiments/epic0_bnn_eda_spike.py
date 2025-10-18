#!/usr/bin/env python3
"""
EPIC 0: EDA Spike - Bayesian Neural Network Feasibility Analysis

This module performs exploratory data analysis (EDA) on real-world financial data
to determine if Bayesian Neural Networks (BNN) can make a positive contribution
to the results. It analyzes data characteristics, predictability, and BNN requirements.

Key Questions:
1. What is the role of BNN in the current system?
2. Can BNN effectively predict returns/risk or portfolio weights?
3. What are the minimal conditions and assumptions for BNN success?
4. Should BNN be deprecated if assumptions don't hold?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialDataAnalyzer:
    """Analyze financial data for BNN feasibility"""
    
    def __init__(self, data_path: str = "data/ftse-updated/"):
        self.data_path = Path(data_path)
        self.data_summary = None
        self.individual_assets = {}
        self.portfolio_data = {}
        self.analysis_results = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available financial data"""
        logger.info("Loading financial data...")
        
        data_files = list(self.data_path.glob("*.csv"))
        loaded_data = {}
        
        for file_path in data_files:
            if file_path.name == "data_summary.csv":
                self.data_summary = pd.read_csv(file_path)
                continue
                
            try:
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                
                # Calculate returns
                df['Returns'] = df['Adj Close'].pct_change()
                df['Log_Returns'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
                df['Volatility'] = df['Returns'].rolling(window=20).std()
                
                loaded_data[file_path.stem] = df
                logger.info(f"Loaded {file_path.stem}: {len(df)} rows")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")
        
        return loaded_data
    
    def analyze_data_characteristics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze basic data characteristics"""
        logger.info("Analyzing data characteristics...")
        
        characteristics = {
            'total_assets': len(data),
            'date_range': {},
            'data_quality': {},
            'return_statistics': {},
            'volatility_statistics': {},
            'correlation_analysis': {},
            'stationarity_tests': {},
            'autocorrelation_analysis': {}
        }
        
        # Date range analysis
        all_dates = []
        for name, df in data.items():
            all_dates.extend(df.index.tolist())
        
        characteristics['date_range'] = {
            'start': min(all_dates),
            'end': max(all_dates),
            'total_days': len(set(all_dates)),
            'avg_observations_per_asset': np.mean([len(df) for df in data.values()])
        }
        
        # Data quality analysis
        for name, df in data.items():
            missing_returns = df['Returns'].isna().sum()
            zero_returns = (df['Returns'] == 0).sum()
            extreme_returns = (np.abs(df['Returns']) > 0.2).sum()
            
            characteristics['data_quality'][name] = {
                'missing_returns': missing_returns,
                'zero_returns': zero_returns,
                'extreme_returns': extreme_returns,
                'data_completeness': 1 - (missing_returns / len(df))
            }
        
        # Return statistics
        all_returns = []
        for name, df in data.items():
            returns = df['Returns'].dropna()
            all_returns.extend(returns.tolist())
            
            characteristics['return_statistics'][name] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
        
        # Overall return statistics
        all_returns = np.array(all_returns)
        characteristics['overall_return_statistics'] = {
            'mean': np.mean(all_returns),
            'std': np.std(all_returns),
            'skewness': stats.skew(all_returns),
            'kurtosis': stats.kurtosis(all_returns),
            'min': np.min(all_returns),
            'max': np.max(all_returns)
        }
        
        return characteristics
    
    def analyze_predictability(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze predictability of financial data"""
        logger.info("Analyzing data predictability...")
        
        predictability_results = {
            'autocorrelation_analysis': {},
            'trend_analysis': {},
            'regime_analysis': {},
            'volatility_clustering': {},
            'cross_asset_correlations': {}
        }
        
        # Autocorrelation analysis
        for name, df in data.items():
            returns = df['Returns'].dropna()
            
            # Calculate autocorrelations for different lags
            autocorrs = []
            for lag in range(1, 21):  # 1 to 20 days
                if len(returns) > lag:
                    autocorr = returns.autocorr(lag=lag)
                    autocorrs.append(autocorr)
            
            predictability_results['autocorrelation_analysis'][name] = {
                'lags_1_5': autocorrs[:5] if len(autocorrs) >= 5 else autocorrs,
                'lags_6_10': autocorrs[5:10] if len(autocorrs) >= 10 else autocorrs[5:],
                'lags_11_20': autocorrs[10:20] if len(autocorrs) >= 20 else autocorrs[10:],
                'max_autocorr': max(np.abs(autocorrs)) if autocorrs else 0,
                'significant_autocorr_count': sum(1 for ac in autocorrs if abs(ac) > 0.1)
            }
        
        # Trend analysis
        for name, df in data.items():
            prices = df['Adj Close'].dropna()
            
            # Linear trend
            x = np.arange(len(prices))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
            
            # Rolling trend analysis
            rolling_trends = []
            window = 252  # 1 year
            for i in range(window, len(prices)):
                window_prices = prices.iloc[i-window:i]
                window_x = np.arange(len(window_prices))
                window_slope, _, _, _, _ = stats.linregress(window_x, window_prices)
                rolling_trends.append(window_slope)
            
            predictability_results['trend_analysis'][name] = {
                'overall_trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'rolling_trend_volatility': np.std(rolling_trends) if rolling_trends else 0,
                'trend_consistency': 1 - np.std(rolling_trends) / abs(np.mean(rolling_trends)) if rolling_trends and np.mean(rolling_trends) != 0 else 0
            }
        
        # Volatility clustering analysis
        for name, df in data.items():
            returns = df['Returns'].dropna()
            volatility = df['Volatility'].dropna()
            
            # GARCH-like analysis
            volatility_autocorr = volatility.autocorr(lag=1)
            returns_squared_autocorr = (returns**2).autocorr(lag=1)
            
            predictability_results['volatility_clustering'][name] = {
                'volatility_autocorr': volatility_autocorr,
                'returns_squared_autocorr': returns_squared_autocorr,
                'volatility_persistence': abs(volatility_autocorr) > 0.1,
                'volatility_clustering_strength': abs(returns_squared_autocorr)
            }
        
        return predictability_results
    
    def analyze_bnn_requirements(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze if data meets BNN requirements"""
        logger.info("Analyzing BNN requirements...")
        
        bnn_requirements = {
            'data_sufficiency': {},
            'noise_characteristics': {},
            'nonlinearity_analysis': {},
            'uncertainty_quantification_potential': {},
            'feature_engineering_opportunities': {}
        }
        
        # Data sufficiency analysis
        for name, df in data.items():
            returns = df['Returns'].dropna()
            
            # Minimum data requirements for BNN
            min_samples_for_bnn = 1000  # Conservative estimate
            sufficient_data = len(returns) >= min_samples_for_bnn
            
            # Training/validation split analysis
            train_size = int(0.7 * len(returns))
            val_size = int(0.15 * len(returns))
            test_size = len(returns) - train_size - val_size
            
            bnn_requirements['data_sufficiency'][name] = {
                'total_samples': len(returns),
                'sufficient_for_bnn': sufficient_data,
                'train_samples': train_size,
                'val_samples': val_size,
                'test_samples': test_size,
                'samples_per_parameter_estimate': len(returns) / 100  # Rough estimate
            }
        
        # Noise characteristics analysis
        for name, df in data.items():
            returns = df['Returns'].dropna()
            
            # Signal-to-noise ratio estimation
            # Use rolling mean as signal, residuals as noise
            rolling_mean = returns.rolling(window=20).mean()
            noise = returns - rolling_mean
            signal_power = np.var(rolling_mean.dropna())
            noise_power = np.var(noise.dropna())
            snr = signal_power / noise_power if noise_power > 0 else 0
            
            # Noise distribution analysis
            noise_skewness = stats.skew(noise.dropna())
            noise_kurtosis = stats.kurtosis(noise.dropna())
            
            bnn_requirements['noise_characteristics'][name] = {
                'signal_to_noise_ratio': snr,
                'noise_skewness': noise_skewness,
                'noise_kurtosis': noise_kurtosis,
                'noise_heteroscedasticity': self._test_heteroscedasticity(returns),
                'noise_autocorrelation': noise.autocorr(lag=1)
            }
        
        # Nonlinearity analysis
        for name, df in data.items():
            returns = df['Returns'].dropna()
            
            # Test for nonlinear patterns
            nonlinearity_score = self._test_nonlinearity(returns)
            
            # Volatility-return relationship
            volatility = df['Volatility'].dropna()
            if len(volatility) > 0 and len(returns) > 0:
                # Align indices
                common_idx = returns.index.intersection(volatility.index)
                if len(common_idx) > 10:
                    aligned_returns = returns.loc[common_idx]
                    aligned_volatility = volatility.loc[common_idx]
                    
                    # Test for volatility clustering (nonlinear relationship)
                    vol_return_corr = np.corrcoef(aligned_returns, aligned_volatility)[0, 1]
                else:
                    vol_return_corr = 0
            else:
                vol_return_corr = 0
            
            bnn_requirements['nonlinearity_analysis'][name] = {
                'nonlinearity_score': nonlinearity_score,
                'volatility_return_correlation': vol_return_corr,
                'has_nonlinear_patterns': nonlinearity_score > 0.1,
                'complexity_estimate': self._estimate_complexity(returns)
            }
        
        return bnn_requirements
    
    def _test_heteroscedasticity(self, returns: pd.Series) -> float:
        """Test for heteroscedasticity in returns"""
        try:
            # Engle's ARCH test approximation
            returns_squared = returns**2
            lagged_squared = returns_squared.shift(1).dropna()
            returns_squared_aligned = returns_squared.iloc[1:]
            
            if len(lagged_squared) > 10 and len(returns_squared_aligned) > 10:
                correlation = np.corrcoef(lagged_squared, returns_squared_aligned)[0, 1]
                return abs(correlation)
            else:
                return 0
        except:
            return 0
    
    def _test_nonlinearity(self, returns: pd.Series) -> float:
        """Test for nonlinearity in returns"""
        try:
            # Simple nonlinearity test using squared returns
            returns_clean = returns.dropna()
            if len(returns_clean) < 50:
                return 0
            
            # Test if squared returns have different autocorrelation than returns
            returns_autocorr = returns_clean.autocorr(lag=1)
            squared_returns_autocorr = (returns_clean**2).autocorr(lag=1)
            
            # Nonlinearity score based on difference in autocorrelation patterns
            nonlinearity_score = abs(squared_returns_autocorr - returns_autocorr)
            return nonlinearity_score
        except:
            return 0
    
    def _estimate_complexity(self, returns: pd.Series) -> float:
        """Estimate complexity of return series"""
        try:
            returns_clean = returns.dropna()
            if len(returns_clean) < 20:
                return 0
            
            # Use sample entropy as complexity measure
            def _sample_entropy(data, m=2, r=0.2):
                N = len(data)
                B = 0.0
                A = 0.0
                
                # Split data into patterns
                patterns = np.array([data[i:i+m] for i in range(N-m+1)])
                
                for i in range(N-m):
                    template_i = patterns[i]
                    for j in range(i+1, N-m+1):
                        template_j = patterns[j]
                        if np.max(np.abs(template_i - template_j)) <= r * np.std(data):
                            B += 1
                            if j < N-m:
                                if np.max(np.abs(template_i - template_j)) <= r * np.std(data):
                                    A += 1
            
            if B > 0:
                return -np.log(A/B) if A > 0 else 0
            else:
                return 0
        except:
            return 0
    
    def analyze_bnn_role_and_alternatives(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze current BNN role and propose alternatives"""
        logger.info("Analyzing BNN role and alternatives...")
        
        role_analysis = {
            'current_bnn_role': {},
            'proposed_bnn_roles': {},
            'alternative_approaches': {},
            'integration_opportunities': {}
        }
        
        # Current BNN role analysis (from uncertainty_aware_asmsoa.py)
        role_analysis['current_bnn_role'] = {
            'description': 'Ensemble of MLPRegressor for uncertainty quantification',
            'inputs': 'Multi-scale features (returns, volatility, correlations)',
            'outputs': 'ROI and risk predictions with uncertainty bounds',
            'limitations': [
                'Not a true Bayesian Neural Network',
                'Uses deterministic ensemble instead of variational inference',
                'Limited integration with thesis framework',
                'No principled uncertainty quantification'
            ]
        }
        
        # Proposed BNN roles
        role_analysis['proposed_bnn_roles'] = {
            'role_1_portfolio_weight_prediction': {
                'description': 'Predict optimal portfolio weights directly',
                'inputs': 'Market features, historical weights, risk factors',
                'outputs': 'Portfolio weight distributions',
                'advantages': [
                    'Direct optimization target',
                    'Natural uncertainty quantification',
                    'End-to-end learning'
                ],
                'challenges': [
                    'Requires large amount of data',
                    'Weight constraints difficult to enforce',
                    'May not capture market dynamics well'
                ]
            },
            'role_2_return_risk_prediction': {
                'description': 'Predict asset returns and risk with uncertainty',
                'inputs': 'Market features, technical indicators, macro factors',
                'outputs': 'Return and risk distributions',
                'advantages': [
                    'Natural fit for BNN uncertainty',
                    'Can integrate with existing portfolio optimization',
                    'Interpretable outputs'
                ],
                'challenges': [
                    'Financial data is notoriously hard to predict',
                    'High noise-to-signal ratio',
                    'Non-stationary nature of markets'
                ]
            },
            'role_3_market_regime_detection': {
                'description': 'Detect market regimes and adjust predictions accordingly',
                'inputs': 'Market features, volatility, correlations',
                'outputs': 'Regime probabilities and regime-specific predictions',
                'advantages': [
                    'Addresses non-stationarity',
                    'Can improve prediction accuracy',
                    'Natural uncertainty in regime classification'
                ],
                'challenges': [
                    'Regime definitions are subjective',
                    'Regime changes are hard to detect in real-time',
                    'Requires historical regime labels'
                ]
            }
        }
        
        # Alternative approaches
        role_analysis['alternative_approaches'] = {
            'enhanced_kalman_filter': {
                'description': 'Improve existing Kalman filter with better state space models',
                'advantages': [
                    'Already integrated with thesis framework',
                    'Proven track record in finance',
                    'Natural uncertainty quantification',
                    'Computationally efficient'
                ],
                'implementation': 'Extend current Kalman filter with regime-switching or time-varying parameters'
            },
            'gaussian_process_regression': {
                'description': 'Use Gaussian Processes for return/risk prediction',
                'advantages': [
                    'Natural uncertainty quantification',
                    'Non-parametric approach',
                    'Can capture complex patterns',
                    'Well-established theory'
                ],
                'challenges': [
                    'Computational complexity O(n³)',
                    'May not scale well with large datasets',
                    'Requires careful kernel selection'
                ]
            },
            'ensemble_methods': {
                'description': 'Improve current ensemble with better base models',
                'advantages': [
                    'Can combine multiple approaches',
                    'Robust to individual model failures',
                    'Easier to implement and debug',
                    'Can incorporate domain knowledge'
                ],
                'implementation': 'Combine Kalman filter, ARIMA, and other time series models'
            }
        }
        
        # Integration opportunities
        role_analysis['integration_opportunities'] = {
            'with_tip_calculation': {
                'description': 'Use BNN to improve TIP (Temporal Incomparability Probability) calculation',
                'approach': 'BNN predicts future objective distributions for TIP calculation',
                'benefits': 'More accurate TIP, better anticipatory learning'
            },
            'with_belief_coefficient': {
                'description': 'Use BNN uncertainty to adjust belief coefficients',
                'approach': 'BNN uncertainty directly influences belief coefficient calculation',
                'benefits': 'Dynamic belief adjustment based on prediction confidence'
            },
            'with_dirichlet_model': {
                'description': 'Use BNN to predict Dirichlet concentration parameters',
                'approach': 'BNN learns optimal concentration parameters for different market conditions',
                'benefits': 'Adaptive Dirichlet model, better portfolio weight predictions'
            }
        }
        
        return role_analysis
    
    def generate_bnn_feasibility_report(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive BNN feasibility report"""
        logger.info("Generating BNN feasibility report...")
        
        # Run all analyses
        characteristics = self.analyze_data_characteristics(data)
        predictability = self.analyze_predictability(data)
        bnn_requirements = self.analyze_bnn_requirements(data)
        role_analysis = self.analyze_bnn_role_and_alternatives(data)
        
        # Generate feasibility assessment
        feasibility_assessment = self._assess_bnn_feasibility(
            characteristics, predictability, bnn_requirements
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            feasibility_assessment, role_analysis
        )
        
        report = {
            'executive_summary': self._generate_executive_summary(feasibility_assessment),
            'data_characteristics': characteristics,
            'predictability_analysis': predictability,
            'bnn_requirements_analysis': bnn_requirements,
            'role_analysis': role_analysis,
            'feasibility_assessment': feasibility_assessment,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _assess_bnn_feasibility(self, characteristics: Dict, predictability: Dict, 
                               bnn_requirements: Dict) -> Dict[str, Any]:
        """Assess overall BNN feasibility"""
        
        # Data sufficiency score
        sufficient_assets = sum(1 for asset in bnn_requirements['data_sufficiency'].values() 
                              if asset['sufficient_for_bnn'])
        data_sufficiency_score = sufficient_assets / len(bnn_requirements['data_sufficiency'])
        
        # Predictability score
        avg_autocorr = np.mean([
            stats['max_autocorr'] for stats in predictability['autocorrelation_analysis'].values()
        ])
        predictability_score = min(avg_autocorr * 5, 1.0)  # Scale to 0-1
        
        # Noise characteristics score
        avg_snr = np.mean([
            stats['signal_to_noise_ratio'] for stats in bnn_requirements['noise_characteristics'].values()
        ])
        noise_score = min(avg_snr / 2, 1.0)  # Scale to 0-1
        
        # Nonlinearity score
        avg_nonlinearity = np.mean([
            stats['nonlinearity_score'] for stats in bnn_requirements['nonlinearity_analysis'].values()
        ])
        nonlinearity_score = min(avg_nonlinearity * 10, 1.0)  # Scale to 0-1
        
        # Overall feasibility score
        overall_score = (data_sufficiency_score * 0.3 + 
                        predictability_score * 0.2 + 
                        noise_score * 0.2 + 
                        nonlinearity_score * 0.3)
        
        # Feasibility assessment
        if overall_score >= 0.7:
            feasibility_level = "HIGH"
            recommendation = "Proceed with BNN implementation"
        elif overall_score >= 0.4:
            feasibility_level = "MEDIUM"
            recommendation = "Proceed with caution, consider alternatives"
        else:
            feasibility_level = "LOW"
            recommendation = "Deprecate BNN, focus on alternative approaches"
        
        return {
            'overall_feasibility_score': overall_score,
            'feasibility_level': feasibility_level,
            'recommendation': recommendation,
            'component_scores': {
                'data_sufficiency': data_sufficiency_score,
                'predictability': predictability_score,
                'noise_characteristics': noise_score,
                'nonlinearity': nonlinearity_score
            },
            'detailed_assessment': {
                'data_sufficiency': {
                    'score': data_sufficiency_score,
                    'sufficient_assets': sufficient_assets,
                    'total_assets': len(bnn_requirements['data_sufficiency']),
                    'interpretation': 'Higher is better - more assets have sufficient data for BNN training'
                },
                'predictability': {
                    'score': predictability_score,
                    'avg_autocorrelation': avg_autocorr,
                    'interpretation': 'Higher is better - more predictable patterns in the data'
                },
                'noise_characteristics': {
                    'score': noise_score,
                    'avg_signal_to_noise': avg_snr,
                    'interpretation': 'Higher is better - cleaner signal relative to noise'
                },
                'nonlinearity': {
                    'score': nonlinearity_score,
                    'avg_nonlinearity': avg_nonlinearity,
                    'interpretation': 'Higher is better - more complex patterns that BNN can capture'
                }
            }
        }
    
    def _generate_recommendations(self, feasibility_assessment: Dict, 
                                role_analysis: Dict) -> Dict[str, Any]:
        """Generate specific recommendations based on analysis"""
        
        recommendations = {
            'primary_recommendation': feasibility_assessment['recommendation'],
            'implementation_strategy': {},
            'alternative_approaches': {},
            'integration_opportunities': {},
            'risk_mitigation': {}
        }
        
        if feasibility_assessment['feasibility_level'] == "HIGH":
            recommendations['implementation_strategy'] = {
                'approach': 'Full BNN implementation',
                'priority_roles': [
                    'return_risk_prediction',
                    'market_regime_detection'
                ],
                'implementation_order': [
                    'Implement true BNN with variational inference',
                    'Integrate with TIP calculation',
                    'Add belief coefficient adjustment',
                    'Compare with thesis method'
                ]
            }
        elif feasibility_assessment['feasibility_level'] == "MEDIUM":
            recommendations['implementation_strategy'] = {
                'approach': 'Hybrid implementation',
                'priority_roles': [
                    'market_regime_detection',
                    'uncertainty_quantification_for_existing_methods'
                ],
                'implementation_order': [
                    'Implement BNN for regime detection only',
                    'Use BNN uncertainty to enhance existing methods',
                    'Gradual integration with thesis framework',
                    'Continuous monitoring and evaluation'
                ]
            }
        else:  # LOW feasibility
            recommendations['implementation_strategy'] = {
                'approach': 'Deprecate BNN, enhance existing methods',
                'priority_alternatives': [
                    'enhanced_kalman_filter',
                    'gaussian_process_regression',
                    'ensemble_methods'
                ],
                'implementation_order': [
                    'Improve existing Kalman filter',
                    'Add regime-switching capabilities',
                    'Enhance ensemble methods',
                    'Focus on thesis method optimization'
                ]
            }
        
        # Risk mitigation strategies
        recommendations['risk_mitigation'] = {
            'data_risks': [
                'Implement robust data validation',
                'Use multiple data sources',
                'Handle missing data gracefully'
            ],
            'model_risks': [
                'Implement model validation framework',
                'Use cross-validation extensively',
                'Monitor model performance continuously'
            ],
            'integration_risks': [
                'Maintain backward compatibility',
                'Implement fallback mechanisms',
                'Gradual rollout with A/B testing'
            ]
        }
        
        return recommendations
    
    def _generate_executive_summary(self, feasibility_assessment: Dict) -> str:
        """Generate executive summary of the analysis"""
        
        score = feasibility_assessment['overall_feasibility_score']
        level = feasibility_assessment['feasibility_level']
        recommendation = feasibility_assessment['recommendation']
        
        summary = f"""
BNN Feasibility Analysis - Executive Summary
============================================

Overall Feasibility Score: {score:.3f} ({level})

Recommendation: {recommendation}

Key Findings:
- Data Sufficiency: {feasibility_assessment['component_scores']['data_sufficiency']:.3f}
- Predictability: {feasibility_assessment['component_scores']['predictability']:.3f}
- Noise Characteristics: {feasibility_assessment['component_scores']['noise_characteristics']:.3f}
- Nonlinearity: {feasibility_assessment['component_scores']['nonlinearity']:.3f}

Assessment: {'BNN shows strong potential for positive contribution' if level == 'HIGH' 
            else 'BNN shows moderate potential with careful implementation' if level == 'MEDIUM'
            else 'BNN is not recommended - focus on alternative approaches'}

Next Steps: {'Proceed with full BNN implementation' if level == 'HIGH'
            else 'Implement hybrid approach with BNN for specific roles' if level == 'MEDIUM'
            else 'Deprecate BNN and enhance existing methods'}
        """
        
        return summary.strip()
    
    def save_report(self, report: Dict[str, Any], output_path: str = "epic0_bnn_feasibility_report.json"):
        """Save the feasibility report"""
        logger.info(f"Saving report to {output_path}")
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Report saved successfully")
    
    def create_visualizations(self, data: Dict[str, pd.DataFrame], 
                            report: Dict[str, Any], output_dir: str = "epic0_visualizations"):
        """Create visualizations for the analysis"""
        logger.info("Creating visualizations...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Data characteristics visualization
        self._plot_data_characteristics(data, output_path)
        
        # 2. Predictability analysis visualization
        self._plot_predictability_analysis(data, output_path)
        
        # 3. BNN feasibility score visualization
        self._plot_feasibility_scores(report, output_path)
        
        # 4. Return distribution analysis
        self._plot_return_distributions(data, output_path)
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_data_characteristics(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Plot data characteristics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Data completeness
        completeness = []
        asset_names = []
        for name, df in data.items():
            completeness.append(1 - df['Returns'].isna().sum() / len(df))
            asset_names.append(name[:15])  # Truncate long names
        
        axes[0, 0].bar(range(len(completeness)), completeness)
        axes[0, 0].set_title('Data Completeness by Asset')
        axes[0, 0].set_ylabel('Completeness Ratio')
        axes[0, 0].set_xticks(range(len(asset_names)))
        axes[0, 0].set_xticklabels(asset_names, rotation=45, ha='right')
        
        # Plot 2: Return statistics
        all_returns = []
        for df in data.values():
            all_returns.extend(df['Returns'].dropna().tolist())
        
        axes[0, 1].hist(all_returns, bins=50, alpha=0.7, density=True)
        axes[0, 1].set_title('Overall Return Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Density')
        
        # Plot 3: Volatility over time (sample)
        sample_asset = list(data.keys())[0]
        sample_df = data[sample_asset]
        axes[1, 0].plot(sample_df.index, sample_df['Volatility'])
        axes[1, 0].set_title(f'Volatility Over Time - {sample_asset}')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Volatility')
        
        # Plot 4: Correlation heatmap (sample of assets)
        sample_assets = list(data.keys())[:10]  # First 10 assets
        returns_matrix = pd.DataFrame({
            name: data[name]['Returns'] for name in sample_assets
        }).dropna()
        
        if len(returns_matrix) > 0:
            correlation_matrix = returns_matrix.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 1], fmt='.2f')
            axes[1, 1].set_title('Return Correlations (Sample)')
        
        plt.tight_layout()
        plt.savefig(output_path / 'data_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_predictability_analysis(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Plot predictability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Autocorrelation analysis
        autocorrs = []
        asset_names = []
        for name, df in data.items():
            returns = df['Returns'].dropna()
            if len(returns) > 20:
                autocorr = returns.autocorr(lag=1)
                autocorrs.append(autocorr)
                asset_names.append(name[:15])
        
        axes[0, 0].bar(range(len(autocorrs)), autocorrs)
        axes[0, 0].set_title('First-Order Autocorrelation by Asset')
        axes[0, 0].set_ylabel('Autocorrelation')
        axes[0, 0].set_xticks(range(len(asset_names)))
        axes[0, 0].set_xticklabels(asset_names, rotation=45, ha='right')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: Volatility clustering
        vol_clustering = []
        for name, df in data.items():
            returns = df['Returns'].dropna()
            if len(returns) > 20:
                vol_cluster = (returns**2).autocorr(lag=1)
                vol_clustering.append(vol_cluster)
        
        axes[0, 1].bar(range(len(vol_clustering)), vol_clustering)
        axes[0, 1].set_title('Volatility Clustering (Squared Returns Autocorr)')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].set_xticks(range(len(asset_names)))
        axes[0, 1].set_xticklabels(asset_names, rotation=45, ha='right')
        
        # Plot 3: Trend analysis
        trend_slopes = []
        for name, df in data.items():
            prices = df['Adj Close'].dropna()
            if len(prices) > 100:
                x = np.arange(len(prices))
                slope, _, _, _, _ = stats.linregress(x, prices)
                trend_slopes.append(slope)
        
        axes[1, 0].bar(range(len(trend_slopes)), trend_slopes)
        axes[1, 0].set_title('Price Trend Slopes')
        axes[1, 0].set_ylabel('Trend Slope')
        axes[1, 0].set_xticks(range(len(asset_names)))
        axes[1, 0].set_xticklabels(asset_names, rotation=45, ha='right')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 4: Signal-to-noise ratio
        snr_values = []
        for name, df in data.items():
            returns = df['Returns'].dropna()
            if len(returns) > 50:
                rolling_mean = returns.rolling(window=20).mean()
                noise = returns - rolling_mean
                signal_power = np.var(rolling_mean.dropna())
                noise_power = np.var(noise.dropna())
                snr = signal_power / noise_power if noise_power > 0 else 0
                snr_values.append(snr)
        
        axes[1, 1].bar(range(len(snr_values)), snr_values)
        axes[1, 1].set_title('Signal-to-Noise Ratio by Asset')
        axes[1, 1].set_ylabel('SNR')
        axes[1, 1].set_xticks(range(len(asset_names)))
        axes[1, 1].set_xticklabels(asset_names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'predictability_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feasibility_scores(self, report: Dict[str, Any], output_path: Path):
        """Plot BNN feasibility scores"""
        feasibility = report['feasibility_assessment']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Component scores
        components = list(feasibility['component_scores'].keys())
        scores = list(feasibility['component_scores'].values())
        colors = ['green' if s >= 0.7 else 'orange' if s >= 0.4 else 'red' for s in scores]
        
        bars = ax1.bar(components, scores, color=colors, alpha=0.7)
        ax1.set_title('BNN Feasibility Component Scores')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High (≥0.7)')
        ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Medium (≥0.4)')
        ax1.legend()
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 2: Overall feasibility
        overall_score = feasibility['overall_feasibility_score']
        feasibility_level = feasibility['feasibility_level']
        
        color = 'green' if feasibility_level == 'HIGH' else 'orange' if feasibility_level == 'MEDIUM' else 'red'
        
        ax2.bar(['Overall Feasibility'], [overall_score], color=color, alpha=0.7)
        ax2.set_title(f'Overall BNN Feasibility: {feasibility_level}')
        ax2.set_ylabel('Score (0-1)')
        ax2.set_ylim(0, 1)
        ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5)
        
        # Add value label
        ax2.text(0, overall_score + 0.01, f'{overall_score:.3f}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'bnn_feasibility_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_return_distributions(self, data: Dict[str, pd.DataFrame], output_path: Path):
        """Plot return distributions for key assets"""
        # Select a few representative assets
        sample_assets = list(data.keys())[:6]  # First 6 assets
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, asset_name in enumerate(sample_assets):
            if i >= 6:
                break
                
            df = data[asset_name]
            returns = df['Returns'].dropna()
            
            # Plot histogram with normal overlay
            axes[i].hist(returns, bins=50, alpha=0.7, density=True, label='Actual')
            
            # Overlay normal distribution
            mu, sigma = returns.mean(), returns.std()
            x = np.linspace(returns.min(), returns.max(), 100)
            normal_dist = stats.norm.pdf(x, mu, sigma)
            axes[i].plot(x, normal_dist, 'r-', linewidth=2, label='Normal')
            
            axes[i].set_title(f'{asset_name}\nSkew: {stats.skew(returns):.3f}, Kurt: {stats.kurtosis(returns):.3f}')
            axes[i].set_xlabel('Returns')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'return_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function to run the EDA spike analysis"""
    logger.info("Starting EPIC 0: BNN EDA Spike Analysis")
    
    # Initialize analyzer
    analyzer = FinancialDataAnalyzer()
    
    # Load data
    data = analyzer.load_data()
    logger.info(f"Loaded {len(data)} assets")
    
    if len(data) == 0:
        logger.error("No data loaded. Exiting.")
        return
    
    # Generate comprehensive report
    report = analyzer.generate_bnn_feasibility_report(data)
    
    # Save report
    analyzer.save_report(report, "epic0_bnn_feasibility_report.json")
    
    # Create visualizations
    analyzer.create_visualizations(data, report)
    
    # Print executive summary
    print("\n" + "="*80)
    print(report['executive_summary'])
    print("="*80)
    
    # Print key recommendations
    print("\nKey Recommendations:")
    print("-" * 40)
    for key, value in report['recommendations']['implementation_strategy'].items():
        if isinstance(value, list):
            print(f"{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")
    
    logger.info("EPIC 0: BNN EDA Spike Analysis completed successfully")
    
    return report


if __name__ == "__main__":
    report = main()



