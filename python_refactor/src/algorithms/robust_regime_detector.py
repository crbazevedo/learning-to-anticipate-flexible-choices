#!/usr/bin/env python3
"""
Robust Regime Detector - Simplified and Robust Implementation

This module implements a robust regime detection system that focuses on
reliability and performance over complexity.

Key features:
1. Simple but effective feature engineering
2. Robust ensemble methods
3. Temporal consistency checks
4. Adaptive confidence thresholds
5. Error handling and fallback mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustRegimeDetector:
    """
    Robust regime detector with simplified but effective methods.
    
    This class implements reliable regime detection using:
    - Simple but effective feature engineering
    - Robust ensemble methods
    - Temporal consistency checks
    - Adaptive confidence thresholds
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, window_size: int = 20, confidence_threshold: float = 0.7):
        """
        Initialize robust regime detector.
        
        Args:
            window_size: Size of the sliding window for regime detection
            confidence_threshold: Minimum confidence threshold for regime detection
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # Initialize simple but effective model
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Regime labels
        self.regime_labels = ['bull_market', 'bear_market', 'sideways_market', 'recovery_market']
        
        # Historical data for temporal consistency
        self.regime_history = []
        self.confidence_history = []
        
        # Adaptive confidence thresholds
        self.adaptive_thresholds = {
            'bull_market': 0.7,
            'bear_market': 0.7,
            'sideways_market': 0.6,
            'recovery_market': 0.7
        }
        
        logger.info(f"Initialized RobustRegimeDetector with window_size={window_size}")
    
    def extract_robust_features(self, data: np.ndarray) -> np.ndarray:
        """
        Extract robust features for regime detection.
        
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
            
            # Simple but effective features
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
                volatility_ratio = volatility / (np.mean(data[i-9:i+1, 0]) if i >= 9 else 1.0)
                
                # Risk momentum
                risk_momentum = risk - data[i-4, 1] if i >= 4 else 0.0
                
                # Regime-specific features
                if i >= 20:
                    # Long-term trend
                    long_trend = (data[i, 0] - data[i-19, 0]) / 20
                    
                    # Regime persistence
                    regime_persistence = self._calculate_regime_persistence()
                    
                    # Market stress indicator
                    market_stress = self._calculate_market_stress(data[i-19:i+1])
                    
                    # Recovery indicator
                    recovery_indicator = self._calculate_recovery_indicator(data[i-19:i+1])
                    
                    feature_vector = np.array([
                        roi, risk, sma_5, sma_10, sma_20, volatility, momentum_5, momentum_10,
                        risk_adjusted_return, trend_strength, volatility_ratio, risk_momentum,
                        long_trend, regime_persistence, market_stress, recovery_indicator
                    ])
                else:
                    feature_vector = np.array([
                        roi, risk, sma_5, sma_10, sma_20, volatility, momentum_5, momentum_10,
                        risk_adjusted_return, trend_strength, volatility_ratio, risk_momentum,
                        0.0, 0.0, 0.0, 0.0
                    ])
            else:
                # Insufficient data - use basic features
                feature_vector = np.array([
                    roi, risk, roi, roi, roi, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 0.0
                ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _calculate_regime_persistence(self) -> float:
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
    
    def detect_regime_robust(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Detect regime using robust methods.
        
        Args:
            data: Financial data array with shape (n_periods, 2) [ROI, risk]
            
        Returns:
            Dictionary with regime detection results
        """
        try:
            # Extract robust features
            features = self.extract_robust_features(data)
            
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
            
            # Get prediction from model
            if hasattr(self.model, 'predict_proba'):
                try:
                    # Get prediction and probability
                    prediction = self.model.predict(current_features)[0]
                    probabilities = self.model.predict_proba(current_features)[0]
                    
                    # Calculate confidence
                    confidence = np.max(probabilities)
                    
                    # Apply temporal consistency check
                    consistent_result = self._apply_temporal_consistency(prediction, confidence)
                    
                    # Update regime history
                    self.regime_history.append(consistent_result['regime'])
                    self.confidence_history.append(consistent_result['confidence'])
                    
                    return consistent_result
                    
                except Exception as e:
                    logger.warning(f"Error in model prediction: {e}")
                    return self._get_default_regime_result()
            else:
                # Model not trained yet - use rule-based fallback
                return self._rule_based_regime_detection(data)
                
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._get_default_regime_result()
    
    def _apply_temporal_consistency(self, regime: str, confidence: float) -> Dict[str, Any]:
        """Apply temporal consistency check to regime detection."""
        
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
            'temporal_consistency': confidence > self.confidence_threshold
        }
    
    def _rule_based_regime_detection(self, data: np.ndarray) -> Dict[str, Any]:
        """Fallback rule-based regime detection."""
        if len(data) == 0:
            return self._get_default_regime_result()
        
        roi, risk = data[-1]
        
        # Simple rule-based detection
        if roi > 0.02 and risk < 0.12:
            regime = 'bull_market'
            confidence = 0.8
        elif roi < -0.01 and risk > 0.18:
            regime = 'bear_market'
            confidence = 0.8
        elif roi > 0.01 and risk < 0.15:
            regime = 'recovery_market'
            confidence = 0.7
        else:
            regime = 'sideways_market'
            confidence = 0.6
        
        # Apply temporal consistency
        consistent_result = self._apply_temporal_consistency(regime, confidence)
        
        # Update regime history
        self.regime_history.append(consistent_result['regime'])
        self.confidence_history.append(consistent_result['confidence'])
        
        return consistent_result
    
    def _get_default_regime_result(self) -> Dict[str, Any]:
        """Get default regime result when detection fails."""
        return {
            'regime': 'sideways_market',
            'confidence': 0.5,
            'temporal_consistency': False
        }
    
    def train_model(self, training_data: np.ndarray, training_labels: List[str]):
        """
        Train model on historical data.
        
        Args:
            training_data: Training features array
            training_labels: Training regime labels
        """
        try:
            # Extract features
            features = self.extract_robust_features(training_data)
            
            # Scale features
            self.feature_scaler = StandardScaler()
            scaled_features = self.feature_scaler.fit_transform(features)
            
            # Train model
            self.model.fit(scaled_features, training_labels)
            logger.info("Successfully trained robust regime detector")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'regime_detector_type': 'RobustRegimeDetector',
            'window_size': self.window_size,
            'confidence_threshold': self.confidence_threshold,
            'adaptive_thresholds': self.adaptive_thresholds,
            'regime_history_length': len(self.regime_history),
            'confidence_history_length': len(self.confidence_history),
            'model_trained': hasattr(self.model, 'classes_')
        }
