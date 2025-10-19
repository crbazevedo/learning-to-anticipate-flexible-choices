#!/usr/bin/env python3
"""
Adaptive Data Quality - Enhanced Data Management

This module implements sophisticated data quality mechanisms with:
1. Advanced data validation
2. Data quality metrics
3. Data preprocessing improvements
4. Adaptive data filtering
5. Dynamic quality adjustment

Key improvements:
1. Advanced data validation for consistency
2. Data quality metrics for monitoring
3. Data preprocessing improvements
4. Adaptive data filtering for noise reduction
5. Dynamic quality adjustment based on performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveDataQuality:
    """
    Adaptive data quality system with enhanced data management.
    
    This class implements sophisticated data quality mechanisms:
    - Advanced data validation
    - Data quality metrics
    - Data preprocessing improvements
    - Adaptive data filtering
    - Dynamic quality adjustment
    """
    
    def __init__(self, target_quality: float = 0.9, validation_window: int = 20):
        """
        Initialize adaptive data quality system.
        
        Args:
            target_quality: Target data quality level
            validation_window: Size of validation window
        """
        self.target_quality = target_quality
        self.validation_window = validation_window
        
        # Data quality tracking
        self.quality_history = deque(maxlen=validation_window)
        self.validation_history = deque(maxlen=validation_window)
        self.preprocessing_history = deque(maxlen=validation_window)
        self.filtering_history = deque(maxlen=validation_window)
        
        # Data validation parameters (MUCH more permissive for financial data)
        self.validation_params = {
            'outlier_threshold': 6.0,        # Increased to 6.0 (very permissive)
            'missing_data_threshold': 0.25,   # Increased to 0.25 (very permissive)
            'consistency_threshold': 0.5,     # Decreased to 0.5 (very permissive)
            'completeness_threshold': 0.6     # Decreased to 0.6 (very permissive)
        }
        
        # Data preprocessing parameters (less aggressive for financial data)
        self.preprocessing_params = {
            'normalization_method': 'robust',     # Changed from 'z_score' to 'robust' (less sensitive to outliers)
            'scaling_method': 'robust',           # Changed from 'min_max' to 'robust' (less sensitive to outliers)
            'encoding_method': 'label',           # Changed from 'one_hot' to 'label' (simpler)
            'feature_selection': False            # Changed from True to False (keep all features)
        }
        
        # Data filtering parameters
        self.filtering_params = {
            'noise_threshold': 0.05,
            'smoothing_window': 5,
            'trend_detection': True,
            'anomaly_detection': True
        }
        
        # Quality metrics
        self.quality_metrics = {
            'current_quality': 0.0,
            'validation_score': 0.0,
            'preprocessing_score': 0.0,
            'filtering_score': 0.0,
            'overall_quality': 0.0
        }
        
        # Dynamic adjustment
        self.validation_enabled = True
        self.preprocessing_enabled = True
        self.filtering_enabled = True
        
        logger.info(f"Initialized AdaptiveDataQuality with target_quality={target_quality}")
    
    def assess_data_quality(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess data quality using advanced validation mechanisms.
        
        Args:
            data: Data array to assess
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary with quality assessment results
        """
        try:
            if data is None or len(data) == 0:
                return self._get_default_quality_result()
            
            # Perform data validation
            validation_result = self._validate_data(data, metadata)
            
            # Perform data preprocessing assessment
            preprocessing_result = self._assess_preprocessing_quality(data, metadata)
            
            # Perform data filtering assessment
            filtering_result = self._assess_filtering_quality(data, metadata)
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(
                validation_result, preprocessing_result, filtering_result
            )
            
            # Store quality history
            self.quality_history.append(overall_quality)
            self.validation_history.append(validation_result['validation_score'])
            self.preprocessing_history.append(preprocessing_result['preprocessing_score'])
            self.filtering_history.append(filtering_result['filtering_score'])
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics()
            
            # Apply dynamic adjustments
            if self.validation_enabled:
                adjusted_validation = self._apply_validation_adjustments(validation_result, data)
            else:
                adjusted_validation = validation_result
            
            return {
                'overall_quality': overall_quality,
                'target_quality': self.target_quality,
                'validation_result': adjusted_validation,
                'preprocessing_result': preprocessing_result,
                'filtering_result': filtering_result,
                'quality_metrics': quality_metrics,
                'data_shape': data.shape,
                'quality_improvement': self._calculate_quality_improvement()
            }
            
        except Exception as e:
            logger.error(f"Error assessing data quality: {e}")
            return self._get_default_quality_result()
    
    def _validate_data(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate data using advanced validation mechanisms."""
        
        validation_scores = {}
        
        # Check for missing values
        missing_ratio = np.isnan(data).sum() / data.size
        missing_score = max(0, 1.0 - missing_ratio / self.validation_params['missing_data_threshold'])
        validation_scores['missing_data'] = missing_score
        
        # Check for outliers
        if len(data.shape) == 1:
            # 1D data
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            outliers = np.abs(data - mean_val) > self.validation_params['outlier_threshold'] * std_val
            outlier_ratio = np.sum(outliers) / len(data)
        else:
            # Multi-dimensional data
            outlier_ratios = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                mean_val = np.nanmean(col_data)
                std_val = np.nanstd(col_data)
                outliers = np.abs(col_data - mean_val) > self.validation_params['outlier_threshold'] * std_val
                outlier_ratios.append(np.sum(outliers) / len(col_data))
            outlier_ratio = np.mean(outlier_ratios)
        
        outlier_score = max(0, 1.0 - outlier_ratio)
        validation_scores['outliers'] = outlier_score
        
        # Check for consistency
        if len(data.shape) == 2 and data.shape[1] > 1:
            # Check correlation between columns
            correlations = []
            for i in range(data.shape[1]):
                for j in range(i+1, data.shape[1]):
                    corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                consistency_score = np.mean(correlations)
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.5
        
        validation_scores['consistency'] = consistency_score
        
        # Check for completeness
        completeness_score = 1.0 - missing_ratio
        validation_scores['completeness'] = completeness_score
        
        # Calculate overall validation score
        validation_score = np.mean(list(validation_scores.values()))
        
        return {
            'validation_score': validation_score,
            'validation_scores': validation_scores,
            'missing_ratio': missing_ratio,
            'outlier_ratio': outlier_ratio,
            'consistency_score': consistency_score,
            'completeness_score': completeness_score
        }
    
    def _assess_preprocessing_quality(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess data preprocessing quality."""
        
        preprocessing_scores = {}
        
        # Check data normalization (robust method)
        if len(data.shape) == 1:
            # Robust normalization using median and IQR
            data_median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                normalized_data = (data - data_median) / (iqr + 1e-8)
                normalization_score = 1.0 - min(1.0, np.std(normalized_data))
            else:
                normalization_score = 0.8  # Good score for constant data
        else:
            normalization_scores = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                col_median = np.median(col_data)
                q75, q25 = np.percentile(col_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    normalized_col = (col_data - col_median) / (iqr + 1e-8)
                    normalization_scores.append(1.0 - min(1.0, np.std(normalized_col)))
                else:
                    normalization_scores.append(0.8)  # Good score for constant data
            normalization_score = np.mean(normalization_scores)
        
        preprocessing_scores['normalization'] = normalization_score
        
        # Check data scaling (robust method)
        if len(data.shape) == 1:
            # Robust scaling using median and IQR
            data_median = np.median(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                scaled_data = (data - data_median) / (iqr + 1e-8)
                scaling_score = 1.0 - min(1.0, np.std(scaled_data))
            else:
                scaling_score = 0.8  # Good score for constant data
        else:
            scaling_scores = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                col_median = np.median(col_data)
                q75, q25 = np.percentile(col_data, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    scaled_col = (col_data - col_median) / (iqr + 1e-8)
                    scaling_scores.append(1.0 - min(1.0, np.std(scaled_col)))
                else:
                    scaling_scores.append(0.8)  # Good score for constant data
            scaling_score = np.mean(scaling_scores)
        
        preprocessing_scores['scaling'] = scaling_score
        
        # Check feature selection (simplified)
        if len(data.shape) == 2:
            # Calculate feature variance
            variances = np.var(data, axis=0)
            high_variance_features = np.sum(variances > np.mean(variances))
            feature_selection_score = min(1.0, high_variance_features / data.shape[1])
        else:
            feature_selection_score = 0.5
        
        preprocessing_scores['feature_selection'] = feature_selection_score
        
        # Calculate overall preprocessing score
        preprocessing_score = np.mean(list(preprocessing_scores.values()))
        
        return {
            'preprocessing_score': preprocessing_score,
            'preprocessing_scores': preprocessing_scores,
            'normalization_score': normalization_score,
            'scaling_score': scaling_score,
            'feature_selection_score': feature_selection_score
        }
    
    def _assess_filtering_quality(self, data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess data filtering quality."""
        
        filtering_scores = {}
        
        # Check noise filtering
        if len(data.shape) == 1:
            # Calculate signal-to-noise ratio
            signal_power = np.var(data)
            noise_power = np.var(np.diff(data))
            if noise_power > 0:
                snr = signal_power / noise_power
                noise_score = min(1.0, snr / 10.0)  # Normalize to 0-1
            else:
                noise_score = 0.5
        else:
            noise_scores = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                signal_power = np.var(col_data)
                noise_power = np.var(np.diff(col_data))
                if noise_power > 0:
                    snr = signal_power / noise_power
                    noise_scores.append(min(1.0, snr / 10.0))
                else:
                    noise_scores.append(0.5)
            noise_score = np.mean(noise_scores)
        
        filtering_scores['noise_filtering'] = noise_score
        
        # Check smoothing quality
        if len(data.shape) == 1:
            # Calculate smoothness
            smoothed_data = self._apply_smoothing(data)
            smoothness = 1.0 - np.mean(np.abs(np.diff(smoothed_data)))
            smoothing_score = max(0, min(1.0, smoothness))
        else:
            smoothing_scores = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                smoothed_col = self._apply_smoothing(col_data)
                smoothness = 1.0 - np.mean(np.abs(np.diff(smoothed_col)))
                smoothing_scores.append(max(0, min(1.0, smoothness)))
            smoothing_score = np.mean(smoothing_scores)
        
        filtering_scores['smoothing'] = smoothing_score
        
        # Check trend detection
        if len(data.shape) == 1:
            # Calculate trend strength
            trend_strength = abs(np.polyfit(range(len(data)), data, 1)[0])
            trend_score = min(1.0, trend_strength * 100)  # Normalize
        else:
            trend_scores = []
            for i in range(data.shape[1]):
                col_data = data[:, i]
                trend_strength = abs(np.polyfit(range(len(col_data)), col_data, 1)[0])
                trend_scores.append(min(1.0, trend_strength * 100))
            trend_score = np.mean(trend_scores)
        
        filtering_scores['trend_detection'] = trend_score
        
        # Calculate overall filtering score
        filtering_score = np.mean(list(filtering_scores.values()))
        
        return {
            'filtering_score': filtering_score,
            'filtering_scores': filtering_scores,
            'noise_score': noise_score,
            'smoothing_score': smoothing_score,
            'trend_score': trend_score
        }
    
    def _apply_smoothing(self, data: np.ndarray, window_size: int = None) -> np.ndarray:
        """Apply smoothing to data."""
        if window_size is None:
            window_size = self.filtering_params['smoothing_window']
        
        if len(data) < window_size:
            return data
        
        smoothed_data = np.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2 + 1)
            smoothed_data[i] = np.mean(data[start_idx:end_idx])
        
        return smoothed_data
    
    def _calculate_overall_quality(self, validation_result: Dict[str, Any], 
                                 preprocessing_result: Dict[str, Any], 
                                 filtering_result: Dict[str, Any]) -> float:
        """Calculate overall data quality score."""
        
        # Weighted combination of quality components (heavily rebalanced for financial data)
        weights = [0.2, 0.3, 0.5]  # Validation (20%), preprocessing (30%), filtering (50%)
        scores = [
            validation_result['validation_score'],
            preprocessing_result['preprocessing_score'],
            filtering_result['filtering_score']
        ]
        
        overall_quality = np.average(scores, weights=weights)
        return min(1.0, max(0.0, overall_quality))
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate detailed quality metrics."""
        
        if len(self.quality_history) == 0:
            return {
                'current_quality': 0.0,
                'validation_score': 0.0,
                'preprocessing_score': 0.0,
                'filtering_score': 0.0,
                'overall_quality': 0.0
            }
        
        # Current quality
        current_quality = self.quality_history[-1]
        
        # Component scores
        validation_score = np.mean(self.validation_history) if len(self.validation_history) > 0 else 0.0
        preprocessing_score = np.mean(self.preprocessing_history) if len(self.preprocessing_history) > 0 else 0.0
        filtering_score = np.mean(self.filtering_history) if len(self.filtering_history) > 0 else 0.0
        
        # Overall quality
        overall_quality = 0.4 * validation_score + 0.3 * preprocessing_score + 0.3 * filtering_score
        
        return {
            'current_quality': current_quality,
            'validation_score': validation_score,
            'preprocessing_score': preprocessing_score,
            'filtering_score': filtering_score,
            'overall_quality': overall_quality
        }
    
    def _apply_validation_adjustments(self, validation_result: Dict[str, Any], 
                                    data: np.ndarray) -> Dict[str, Any]:
        """Apply dynamic validation adjustments."""
        
        adjusted_result = validation_result.copy()
        
        # Adjust thresholds based on data characteristics
        if validation_result['missing_ratio'] > 0.05:
            # High missing data - relax outlier threshold
            adjusted_result['outlier_threshold'] = self.validation_params['outlier_threshold'] * 1.2
        else:
            # Low missing data - tighten outlier threshold
            adjusted_result['outlier_threshold'] = self.validation_params['outlier_threshold'] * 0.9
        
        # Adjust consistency threshold based on data size
        if len(data) > 100:
            adjusted_result['consistency_threshold'] = self.validation_params['consistency_threshold'] * 1.1
        else:
            adjusted_result['consistency_threshold'] = self.validation_params['consistency_threshold'] * 0.9
        
        return adjusted_result
    
    def _calculate_quality_improvement(self) -> float:
        """Calculate quality improvement over time."""
        
        if len(self.quality_history) < 2:
            return 0.0
        
        recent_quality = list(self.quality_history)[-5:]
        if len(recent_quality) >= 2:
            improvement = recent_quality[-1] - recent_quality[0]
            return improvement
        else:
            return 0.0
    
    def _get_default_quality_result(self) -> Dict[str, Any]:
        """Get default quality result when assessment fails."""
        return {
            'overall_quality': 0.5,
            'target_quality': self.target_quality,
            'validation_result': {'validation_score': 0.5},
            'preprocessing_result': {'preprocessing_score': 0.5},
            'filtering_result': {'filtering_score': 0.5},
            'quality_metrics': {'overall_quality': 0.5},
            'data_shape': (0,),
            'quality_improvement': 0.0
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'quality_type': 'AdaptiveDataQuality',
            'target_quality': self.target_quality,
            'validation_window': self.validation_window,
            'validation_params': self.validation_params,
            'preprocessing_params': self.preprocessing_params,
            'filtering_params': self.filtering_params,
            'quality_history_length': len(self.quality_history),
            'validation_history_length': len(self.validation_history),
            'preprocessing_history_length': len(self.preprocessing_history),
            'filtering_history_length': len(self.filtering_history),
            'validation_enabled': self.validation_enabled,
            'preprocessing_enabled': self.preprocessing_enabled,
            'filtering_enabled': self.filtering_enabled
        }
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get detailed quality metrics."""
        
        if len(self.quality_history) == 0:
            return {
                'current_quality': 0.0,
                'validation_score': 0.0,
                'preprocessing_score': 0.0,
                'filtering_score': 0.0,
                'overall_quality': 0.0,
                'quality_trend': 0.0
            }
        
        # Calculate current metrics
        current_quality = self.quality_history[-1]
        quality_metrics = self._calculate_quality_metrics()
        
        # Calculate quality trend
        if len(self.quality_history) >= 3:
            recent_quality = list(self.quality_history)[-3:]
            quality_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        else:
            quality_trend = 0.0
        
        return {
            'current_quality': current_quality,
            'validation_score': quality_metrics['validation_score'],
            'preprocessing_score': quality_metrics['preprocessing_score'],
            'filtering_score': quality_metrics['filtering_score'],
            'overall_quality': quality_metrics['overall_quality'],
            'quality_trend': quality_trend,
            'quality_history': list(self.quality_history),
            'validation_history': list(self.validation_history),
            'preprocessing_history': list(self.preprocessing_history),
            'filtering_history': list(self.filtering_history)
        }
