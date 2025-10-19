#!/usr/bin/env python3
"""
Advanced Regime Detection V2 - Enhanced Ensemble Methods

This module implements sophisticated ensemble methods for regime detection
with temporal consistency checks, advanced feature engineering, and
regime transition probabilities.

Key improvements:
1. Ensemble methods with multiple models
2. Temporal consistency checks for regime stability
3. Advanced feature engineering for regime detection
4. Regime transition probabilities
5. Adaptive confidence thresholds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRegimeDetectorV2:
    """
    Advanced regime detector with ensemble methods and temporal consistency.
    
    This class implements sophisticated regime detection using:
    - Multiple ensemble models (RF, GB, SVM, MLP)
    - Temporal consistency checks
    - Advanced feature engineering
    - Regime transition probabilities
    - Adaptive confidence thresholds
    """
    
    def __init__(self, window_size: int = 20, confidence_threshold: float = 0.7):
        """
        Initialize advanced regime detector.
        
        Args:
            window_size: Size of the sliding window for regime detection
            confidence_threshold: Minimum confidence threshold for regime detection
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Initialize ensemble models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                random_state=42,
                max_iter=1000
            )
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Regime labels
        self.regime_labels = ['bull_market', 'bear_market', 'sideways_market', 'recovery_market']
        
        # Historical data for temporal consistency
        self.regime_history = []
        self.confidence_history = []
        self.feature_history = []
        
        # Regime transition probabilities
        self.transition_matrix = np.ones((4, 4)) * 0.25  # Initialize uniform
        self.regime_counts = np.zeros(4)
        
        # Adaptive confidence thresholds
        self.adaptive_thresholds = {
            'bull_market': 0.7,
            'bear_market': 0.7,
            'sideways_market': 0.6,
            'recovery_market': 0.7
        }
        
        # Feature importance tracking
        self.feature_importance = {}
        
        logger.info(f"Initialized AdvancedRegimeDetectorV2 with window_size={window_size}")
    
    def extract_advanced_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract advanced features for regime detection.
        
        Args:
            data: Financial data array with shape (n_periods, 2) [ROI, risk]
            
        Returns:
            Feature array with shape (n_periods, n_features)
        """
        if len(data) < self.window_size:
            # Pad with zeros if insufficient data
            padded_data = np.zeros((self.window_size, 2))
            padded_data[-len(data):] = data
            data = padded_data
        
        features = []
        
        for i in range(len(data)):
            # Basic features
            roi, risk = data[i]
            
            # Technical indicators
            if i >= 5:
                # Moving averages
                sma_5 = np.mean(data[i-4:i+1, 0])
                sma_10 = np.mean(data[i-9:i+1, 0]) if i >= 9 else sma_5
                sma_20 = np.mean(data[i-19:i+1, 0]) if i >= 19 else sma_10
                
                # Volatility
                volatility = np.std(data[i-4:i+1, 0])
                
                # Momentum
                momentum_5 = data[i, 0] - data[i-4, 0]
                momentum_10 = data[i, 0] - data[i-9, 0] if i >= 9 else momentum_5
                
                # Risk-adjusted returns
                risk_adjusted_return = roi / (risk + 1e-8)
                
                # Trend strength
                trend_strength = (sma_5 - sma_20) / (sma_20 + 1e-8)
                
                # Volatility ratio
                volatility_ratio = volatility / (np.mean(data[i-9:i+1, 0]) + 1e-8) if i >= 9 else 1.0
                
                # Risk momentum
                risk_momentum = risk - data[i-4, 1] if i >= 4 else 0.0
                
                # Regime-specific features
                if i >= 20:
                    # Long-term trend
                    long_trend = (data[i, 0] - data[i-19, 0]) / 20
                    
                    # Regime persistence
                    regime_persistence = self._calculate_regime_persistence(i)
                    
                    # Market stress indicator
                    market_stress = self._calculate_market_stress(data[i-19:i+1])
                    
                    # Recovery indicator
                    recovery_indicator = self._calculate_recovery_indicator(data[i-19:i+1])
                    
                    # Volatility clustering
                    volatility_clustering = self._calculate_volatility_clustering(data[i-19:i+1])
                    
                    # Mean reversion
                    mean_reversion = self._calculate_mean_reversion(data[i-19:i+1])
                    
                    # Regime transition features
                    transition_features = self._calculate_transition_features(i)
                    
                    feature_vector = np.array([
                        roi, risk, sma_5, sma_10, sma_20, volatility, momentum_5, momentum_10,
                        risk_adjusted_return, trend_strength, volatility_ratio, risk_momentum,
                        long_trend, regime_persistence, market_stress, recovery_indicator,
                        volatility_clustering, mean_reversion, *transition_features
                    ])
                else:
                    feature_vector = np.array([
                        roi, risk, sma_5, sma_10, sma_20, volatility, momentum_5, momentum_10,
                        risk_adjusted_return, trend_strength, volatility_ratio, risk_momentum,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])
            else:
                # Insufficient data - use basic features
                feature_vector = np.array([
                    roi, risk, roi, roi, roi, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_regime_persistence(self, current_index: int) -> float:
        """Calculate regime persistence based on recent history."""
        if len(self.regime_history) < 5:
            return 0.5
        
        recent_regimes = self.regime_history[-5:]
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        max_count = max(regime_counts.values())
        return max_count / len(recent_regimes)
    
    def _calculate_market_stress(self, data: np.ndarray) -> float:
        """Calculate market stress indicator."""
        if len(data) < 5:
            return 0.0
        
        roi_data = data[:, 0]
        risk_data = data[:, 1]
        
        # High volatility and negative returns indicate stress
        volatility = np.std(roi_data)
        negative_returns = np.sum(roi_data < 0) / len(roi_data)
        high_risk = np.mean(risk_data)
        
        stress = (volatility * 0.4 + negative_returns * 0.4 + high_risk * 0.2)
        return min(stress, 1.0)
    
    def _calculate_recovery_indicator(self, data: np.ndarray) -> float:
        """Calculate recovery indicator."""
        if len(data) < 10:
            return 0.0
        
        roi_data = data[:, 0]
        
        # Check for recovery pattern (negative to positive trend)
        first_half = np.mean(roi_data[:len(roi_data)//2])
        second_half = np.mean(roi_data[len(roi_data)//2:])
        
        if first_half < 0 and second_half > first_half:
            return min((second_half - first_half) / abs(first_half + 1e-8), 1.0)
        
        return 0.0
    
    def _calculate_volatility_clustering(self, data: np.ndarray) -> float:
        """Calculate volatility clustering indicator."""
        if len(data) < 10:
            return 0.0
        
        roi_data = data[:, 0]
        returns = np.diff(roi_data)
        
        if len(returns) < 5:
            return 0.0
        
        # Calculate autocorrelation of squared returns
        squared_returns = returns ** 2
        autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        
        return max(autocorr, 0.0) if not np.isnan(autocorr) else 0.0
    
    def _calculate_mean_reversion(self, data: np.ndarray) -> float:
        """Calculate mean reversion indicator."""
        if len(data) < 10:
            return 0.0
        
        roi_data = data[:, 0]
        mean_roi = np.mean(roi_data)
        
        # Calculate how much the current value deviates from mean
        current_deviation = abs(roi_data[-1] - mean_roi)
        historical_deviation = np.mean(np.abs(roi_data - mean_roi))
        
        if historical_deviation > 0:
            return min(current_deviation / historical_deviation, 1.0)
        
        return 0.0
    
    def _calculate_transition_features(self, current_index: int) -> np.ndarray:
        """Calculate regime transition features."""
        if len(self.regime_history) < 2:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        # Get transition probabilities from current regime
        current_regime = self.regime_history[-1]
        if current_regime in self.regime_labels:
            regime_index = self.regime_labels.index(current_regime)
            return self.transition_matrix[regime_index]
        else:
            return np.array([0.25, 0.25, 0.25, 0.25])
    
    def detect_regime_advanced(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect regime using advanced ensemble methods.
        
        Args:
            data: Financial data array with shape (n_periods, 2) [ROI, risk]
            
        Returns:
            Dictionary with regime detection results
        """
        try:
            # Extract advanced features
            features = self.extract_advanced_features(data)
            
            if len(features) == 0:
                return self._get_default_regime_result()
            
            # Use the most recent features
            current_features = features[-1].reshape(1, -1)
            
            # Scale features
            if hasattr(self, 'feature_scaler'):
                current_features = self.feature_scaler.transform(current_features)
            else:
                # Initialize scaler with current features
                self.feature_scaler = StandardScaler()
                current_features = self.feature_scaler.fit_transform(current_features)
            
            # Get predictions from all models
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    try:
                        # Get prediction and probability
                        prediction = model.predict(current_features)[0]
                        probabilities = model.predict_proba(current_features)[0]
                        
                        model_predictions[model_name] = prediction
                        model_probabilities[model_name] = probabilities
                    except Exception as e:
                        logger.warning(f"Error in {model_name} prediction: {e}")
                        # Use default prediction
                        model_predictions[model_name] = 'sideways_market'
                        model_probabilities[model_name] = np.array([0.25, 0.25, 0.25, 0.25])
                else:
                    # Model not trained yet
                    model_predictions[model_name] = 'sideways_market'
                    model_probabilities[model_name] = np.array([0.25, 0.25, 0.25, 0.25])
            
            # Ensemble prediction using weighted voting
            ensemble_result = self._ensemble_prediction(model_predictions, model_probabilities)
            
            # Apply temporal consistency check
            consistent_result = self._apply_temporal_consistency(ensemble_result)
            
            # Update regime history
            self.regime_history.append(consistent_result['regime'])
            self.confidence_history.append(consistent_result['confidence'])
            self.feature_history.append(current_features[0])
            
            # Update transition matrix
            self._update_transition_matrix()
            
            # Update adaptive thresholds
            self._update_adaptive_thresholds()
            
            return consistent_result
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._get_default_regime_result()
    
    def _ensemble_prediction(self, model_predictions: Dict, model_probabilities: Dict) -> Dict[str, Any]:
        """Perform ensemble prediction with weighted voting."""
        
        # Calculate weights based on model performance (simplified)
        model_weights = {
            'random_forest': 0.3,
            'gradient_boosting': 0.3,
            'svm': 0.2,
            'mlp': 0.2
        }
        
        # Weighted voting
        regime_votes = {regime: 0.0 for regime in self.regime_labels}
        
        for model_name, prediction in model_predictions.items():
            weight = model_weights.get(model_name, 0.25)
            regime_votes[prediction] += weight
        
        # Get ensemble prediction
        ensemble_regime = max(regime_votes, key=regime_votes.get)
        ensemble_confidence = regime_votes[ensemble_regime]
        
        # Calculate weighted probabilities
        weighted_probabilities = np.zeros(len(self.regime_labels))
        for model_name, probabilities in model_probabilities.items():
            weight = model_weights.get(model_name, 0.25)
            weighted_probabilities += weight * probabilities
        
        # Normalize probabilities
        weighted_probabilities = weighted_probabilities / np.sum(weighted_probabilities)
        
        return {
            'regime': ensemble_regime,
            'confidence': ensemble_confidence,
            'probabilities': weighted_probabilities,
            'model_predictions': model_predictions,
            'model_probabilities': model_probabilities
        }
    
    def _apply_temporal_consistency(self, ensemble_result: Dict) -> Dict[str, Any]:
        """Apply temporal consistency check to regime detection."""
        
        regime = ensemble_result['regime']
        confidence = ensemble_result['confidence']
        
        # Check temporal consistency
        if len(self.regime_history) >= 3:
            recent_regimes = self.regime_history[-3:]
            regime_counts = {}
            for r in recent_regimes:
                regime_counts[r] = regime_counts.get(r, 0) + 1
            
            # If current regime is different from recent majority, apply consistency penalty
            majority_regime = max(regime_counts, key=regime_counts.get)
            if regime != majority_regime:
                # Apply consistency penalty
                consistency_penalty = 0.1
                confidence = max(confidence - consistency_penalty, 0.1)
                
                # If confidence is too low, use majority regime
                if confidence < self.confidence_threshold:
                    regime = majority_regime
                    confidence = 0.6  # Moderate confidence for consistency
        
        # Apply adaptive threshold
        adaptive_threshold = self.adaptive_thresholds.get(regime, self.confidence_threshold)
        if confidence < adaptive_threshold:
            # Use fallback regime
            regime = 'sideways_market'
            confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'probabilities': ensemble_result['probabilities'],
            'model_predictions': ensemble_result['model_predictions'],
            'model_probabilities': ensemble_result['model_probabilities'],
            'temporal_consistency': confidence > self.confidence_threshold
        }
    
    def _update_transition_matrix(self):
        """Update regime transition matrix based on history."""
        if len(self.regime_history) < 2:
            return
        
        # Count transitions
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i]
            to_regime = self.regime_history[i + 1]
            
            from_index = self.regime_labels.index(from_regime)
            to_index = self.regime_labels.index(to_regime)
            
            self.transition_matrix[from_index, to_index] += 1
        
        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1)
        for i in range(len(self.regime_labels)):
            if row_sums[i] > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / row_sums[i]
    
    def _update_adaptive_thresholds(self):
        """Update adaptive confidence thresholds based on performance."""
        if len(self.confidence_history) < 10:
            return
        
        # Calculate performance metrics
        recent_confidences = self.confidence_history[-10:]
        avg_confidence = np.mean(recent_confidences)
        
        # Adjust thresholds based on performance
        for regime in self.regime_labels:
            if regime in self.adaptive_thresholds:
                # Increase threshold if confidence is consistently high
                if avg_confidence > 0.8:
                    self.adaptive_thresholds[regime] = min(
                        self.adaptive_thresholds[regime] + 0.01, 0.9
                    )
                # Decrease threshold if confidence is consistently low
                elif avg_confidence < 0.6:
                    self.adaptive_thresholds[regime] = max(
                        self.adaptive_thresholds[regime] - 0.01, 0.5
                    )
    
    def _get_default_regime_result(self) -> Dict[str, Any]:
        """Get default regime result when detection fails."""
        return {
            'regime': 'sideways_market',
            'confidence': 0.5,
            'probabilities': np.array([0.25, 0.25, 0.25, 0.25]),
            'model_predictions': {},
            'model_probabilities': {},
            'temporal_consistency': False
        }
    
    def train_models(self, training_data: np.ndarray, training_labels: List[str]):
        """
        Train ensemble models on historical data.
        
        Args:
            training_data: Training features array
            training_labels: Training regime labels
        """
        try:
            # Extract features
            features = self.extract_advanced_features(training_data)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    model.fit(scaled_features, training_labels)
                    logger.info(f"Successfully trained {model_name}")
                except Exception as e:
                    logger.warning(f"Error training {model_name}: {e}")
            
            # Calculate feature importance
            self._calculate_feature_importance()
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from trained models."""
        try:
            feature_names = [
                'roi', 'risk', 'sma_5', 'sma_10', 'sma_20', 'volatility', 'momentum_5', 'momentum_10',
                'risk_adjusted_return', 'trend_strength', 'volatility_ratio', 'risk_momentum',
                'long_trend', 'regime_persistence', 'market_stress', 'recovery_indicator',
                'volatility_clustering', 'mean_reversion', 'transition_prob_0', 'transition_prob_1',
                'transition_prob_2', 'transition_prob_3'
            ]
            
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(zip(feature_names, model.feature_importances_))
                else:
                    self.feature_importance[model_name] = {name: 0.0 for name in feature_names}
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'regime_detector_type': 'AdvancedRegimeDetectorV2',
            'window_size': self.window_size,
            'confidence_threshold': self.confidence_threshold,
            'adaptive_thresholds': self.adaptive_thresholds,
            'regime_history_length': len(self.regime_history),
            'confidence_history_length': len(self.confidence_history),
            'transition_matrix': self.transition_matrix.tolist(),
            'feature_importance': self.feature_importance,
            'models_trained': all(hasattr(model, 'classes_') for model in self.models.values())
        }
