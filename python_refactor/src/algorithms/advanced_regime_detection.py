#!/usr/bin/env python3
"""
Advanced Regime Detection System

Implements ensemble regime detection with multiple algorithms and advanced
feature engineering for improved accuracy and stability.

Target: 80%+ regime detection accuracy, <20% exploration rate, 70%+ confidence
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class AdvancedRegimeResult:
    """Result of advanced regime detection"""
    regime: str
    confidence: float
    ensemble_confidence: float
    individual_predictions: Dict[str, str]
    individual_confidences: Dict[str, float]
    feature_importance: Dict[str, float]
    regret_bound: float
    timestamp: float

class TechnicalIndicators:
    """Technical indicators for financial data analysis"""
    
    def __init__(self):
        self.indicators = {}
    
    def extract_indicators(self, data: np.ndarray, window: int = 20) -> Dict[str, float]:
        """Extract comprehensive technical indicators"""
        
        if len(data) < window:
            window = len(data)
        
        roi_data = data[:, 0]
        risk_data = data[:, 1]
        
        indicators = {}
        
        # Price-based indicators
        indicators['roi_sma_short'] = np.mean(roi_data[-5:])
        indicators['roi_sma_long'] = np.mean(roi_data[-window:])
        indicators['roi_ema_short'] = self._calculate_ema(roi_data[-5:], 0.3)
        indicators['roi_ema_long'] = self._calculate_ema(roi_data[-window:], 0.1)
        
        # Volatility indicators
        indicators['roi_volatility'] = np.std(roi_data[-window:])
        indicators['risk_volatility'] = np.std(risk_data[-window:])
        indicators['volatility_ratio'] = indicators['roi_volatility'] / indicators['risk_volatility']
        
        # Momentum indicators
        if len(roi_data) >= 5:
            indicators['roi_momentum'] = (roi_data[-1] - roi_data[-5]) / 5
        else:
            indicators['roi_momentum'] = 0.0
        
        if len(risk_data) >= 5:
            indicators['risk_momentum'] = (risk_data[-1] - risk_data[-5]) / 5
        else:
            indicators['risk_momentum'] = 0.0
        # Momentum ratio (avoid division by zero)
        if abs(indicators['risk_momentum']) > 1e-8:
            indicators['momentum_ratio'] = indicators['roi_momentum'] / indicators['risk_momentum']
        else:
            indicators['momentum_ratio'] = 0.0
        
        # Trend indicators
        indicators['trend_strength'] = abs(indicators['roi_sma_short'] - indicators['roi_sma_long']) / indicators['roi_volatility']
        indicators['trend_direction'] = 1 if indicators['roi_sma_short'] > indicators['roi_sma_long'] else -1
        
        # Market regime indicators
        indicators['bull_indicator'] = 1 if (indicators['roi_sma_short'] > indicators['roi_sma_long'] and 
                                           indicators['roi_volatility'] < 0.02) else 0
        indicators['bear_indicator'] = 1 if (indicators['roi_sma_short'] < indicators['roi_sma_long'] and 
                                           indicators['roi_volatility'] > 0.03) else 0
        indicators['sideways_indicator'] = 1 if abs(indicators['roi_sma_short'] - indicators['roi_sma_long']) < 0.01 else 0
        
        # Risk-adjusted indicators
        indicators['sharpe_ratio'] = indicators['roi_momentum'] / indicators['roi_volatility'] if indicators['roi_volatility'] > 0 else 0
        indicators['risk_adjusted_return'] = indicators['roi_momentum'] / indicators['risk_volatility'] if indicators['risk_volatility'] > 0 else 0
        
        return indicators
    
    def _calculate_ema(self, data: np.ndarray, alpha: float) -> float:
        """Calculate exponential moving average"""
        if len(data) == 0:
            return 0.0
        
        ema = data[0]
        for value in data[1:]:
            ema = alpha * value + (1 - alpha) * ema
        
        return ema

class SVMMarketRegimeClassifier:
    """SVM-based market regime classifier"""
    
    def __init__(self):
        self.classifier = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.is_trained = False
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
    
    def train(self, features: np.ndarray, regimes: np.ndarray) -> None:
        """Train SVM classifier"""
        if len(features) == 0 or len(regimes) == 0:
            return
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train classifier
        self.classifier.fit(features_scaled, regimes)
        self.is_trained = True
        
        logger.info(f"Trained SVM classifier with {len(features)} samples")
    
    def predict_regime(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict market regime"""
        if not self.is_trained:
            return 'sideways_market', 0.5
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict regime and probability
        regime_proba = self.classifier.predict_proba(features_scaled)[0]
        regime_idx = np.argmax(regime_proba)
        confidence = regime_proba[regime_idx]
        
        return self.regime_names[regime_idx], confidence

class LSTMRegimePredictor:
    """LSTM-based regime predictor (simplified version)"""
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
        self.is_trained = False
    
    def train(self, sequences: np.ndarray, regimes: np.ndarray) -> None:
        """Train LSTM predictor (simplified)"""
        if len(sequences) == 0 or len(regimes) == 0:
            return
        
        # Simplified training (in practice, would use actual LSTM)
        self.is_trained = True
        logger.info(f"Trained LSTM predictor with {len(sequences)} sequences")
    
    def predict_regime(self, sequence: np.ndarray) -> Tuple[str, float]:
        """Predict regime from sequence"""
        if not self.is_trained or len(sequence) < self.sequence_length:
            return 'sideways_market', 0.5
        
        # Simplified prediction based on sequence patterns
        roi_trend = np.mean(sequence[-self.sequence_length:, 0])
        risk_trend = np.mean(sequence[-self.sequence_length:, 1])
        
        if roi_trend > 0.01 and risk_trend < 0.15:
            return 'bull_market', 0.8
        elif roi_trend < -0.005 and risk_trend > 0.2:
            return 'bear_market', 0.8
        else:
            return 'sideways_market', 0.6

class TechnicalAnalysisRegimeDetector:
    """Technical analysis-based regime detector"""
    
    def __init__(self):
        self.regime_names = ['bull_market', 'bear_market', 'sideways_market']
    
    def detect_regime(self, data: np.ndarray) -> Tuple[str, float]:
        """Detect regime using technical analysis"""
        if len(data) < 10:
            return 'sideways_market', 0.5
        
        roi_data = data[:, 0]
        risk_data = data[:, 1]
        
        # Calculate technical indicators
        sma_short = np.mean(roi_data[-5:])
        sma_long = np.mean(roi_data[-10:])
        volatility = np.std(roi_data[-10:])
        
        # Bull market criteria
        bull_score = 0
        if sma_short > sma_long:
            bull_score += 1
        if volatility < 0.02:
            bull_score += 1
        if roi_data[-1] > roi_data[-5]:
            bull_score += 1
        
        # Bear market criteria
        bear_score = 0
        if sma_short < sma_long:
            bear_score += 1
        if volatility > 0.03:
            bear_score += 1
        if roi_data[-1] < roi_data[-5]:
            bear_score += 1
        
        # Determine regime
        if bull_score >= 2:
            return 'bull_market', 0.8
        elif bear_score >= 2:
            return 'bear_market', 0.8
        else:
            return 'sideways_market', 0.6

class EnsembleRegimeDetector:
    """Ensemble regime detection with multiple algorithms"""
    
    def __init__(self):
        self.detectors = {
            'contextual_bandit': None,  # Will be set externally
            'svm_classifier': SVMMarketRegimeClassifier(),
            'lstm_predictor': LSTMRegimePredictor(),
            'technical_analyzer': TechnicalAnalysisRegimeDetector()
        }
        
        # Adaptive weights based on performance
        self.weights = {
            'contextual_bandit': 0.3,
            'svm_classifier': 0.25,
            'lstm_predictor': 0.25,
            'technical_analyzer': 0.2
        }
        
        # Performance tracking
        self.performance_history = {name: [] for name in self.detectors.keys()}
        self.feature_importance = {}
        
        # Technical indicators
        self.technical_indicators = TechnicalIndicators()
        
        logger.info("Initialized EnsembleRegimeDetector")
    
    def set_contextual_bandit(self, bandit) -> None:
        """Set contextual bandit detector"""
        self.detectors['contextual_bandit'] = bandit
    
    def extract_advanced_features(self, data: np.ndarray, window: int = 20) -> np.ndarray:
        """Extract advanced features for regime detection"""
        
        # Get technical indicators
        indicators = self.technical_indicators.extract_indicators(data, window)
        
        # Convert to feature vector
        features = np.array(list(indicators.values()))
        
        return features
    
    def detect_regime(self, data: np.ndarray, window: int = 20) -> AdvancedRegimeResult:
        """Ensemble regime detection"""
        
        # Extract advanced features
        features = self.extract_advanced_features(data, window)
        
        # Get predictions from all detectors
        predictions = {}
        confidences = {}
        
        for name, detector in self.detectors.items():
            if detector is None:
                continue
            
            try:
                if name == 'contextual_bandit':
                    # Use contextual bandit
                    result = detector.detect_regime(data, window)
                    predictions[name] = result.regime
                    confidences[name] = result.confidence
                
                elif name == 'svm_classifier':
                    # Use SVM classifier
                    regime, confidence = detector.predict_regime(features)
                    predictions[name] = regime
                    confidences[name] = confidence
                
                elif name == 'lstm_predictor':
                    # Use LSTM predictor
                    regime, confidence = detector.predict_regime(data)
                    predictions[name] = regime
                    confidences[name] = confidence
                
                elif name == 'technical_analyzer':
                    # Use technical analyzer
                    regime, confidence = detector.detect_regime(data)
                    predictions[name] = regime
                    confidences[name] = confidence
                
            except Exception as e:
                logger.warning(f"Error in {name} detector: {e}")
                predictions[name] = 'sideways_market'
                confidences[name] = 0.5
        
        # Ensemble voting with adaptive weights
        regime_votes = {}
        total_weight = 0
        
        for name, regime in predictions.items():
            if name in self.weights:
                weight = self.weights[name] * confidences[name]
                regime_votes[regime] = regime_votes.get(regime, 0) + weight
                total_weight += weight
        
        # Select regime with highest weighted vote
        if regime_votes:
            best_regime = max(regime_votes, key=regime_votes.get)
            ensemble_confidence = regime_votes[best_regime] / total_weight if total_weight > 0 else 0.5
        else:
            best_regime = 'sideways_market'
            ensemble_confidence = 0.5
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features, predictions)
        
        # Calculate regret bound
        regret_bound = self._calculate_ensemble_regret_bound(len(data))
        
        # Update performance tracking
        self._update_performance_tracking(predictions, confidences)
        
        return AdvancedRegimeResult(
            regime=best_regime,
            confidence=ensemble_confidence,
            ensemble_confidence=ensemble_confidence,
            individual_predictions=predictions,
            individual_confidences=confidences,
            feature_importance=feature_importance,
            regret_bound=regret_bound,
            timestamp=datetime.now().timestamp()
        )
    
    def _calculate_feature_importance(self, features: np.ndarray, predictions: Dict[str, str]) -> Dict[str, float]:
        """Calculate feature importance based on predictions"""
        
        # Simple feature importance calculation
        feature_names = [
            'roi_sma_short', 'roi_sma_long', 'roi_ema_short', 'roi_ema_long',
            'roi_volatility', 'risk_volatility', 'volatility_ratio',
            'roi_momentum', 'risk_momentum', 'momentum_ratio',
            'trend_strength', 'trend_direction',
            'bull_indicator', 'bear_indicator', 'sideways_indicator',
            'sharpe_ratio', 'risk_adjusted_return'
        ]
        
        importance = {}
        for i, name in enumerate(feature_names):
            if i < len(features):
                importance[name] = abs(features[i])
            else:
                importance[name] = 0.0
        
        return importance
    
    def _calculate_ensemble_regret_bound(self, T: int) -> float:
        """Calculate ensemble regret bound"""
        if T <= 0:
            return 0.0
        
        # Ensemble regret bound (simplified)
        return np.sqrt(len(self.detectors) * T * np.log(T + 1))
    
    def _update_performance_tracking(self, predictions: Dict[str, str], confidences: Dict[str, float]) -> None:
        """Update performance tracking for adaptive weights"""
        
        for name, confidence in confidences.items():
            if name in self.performance_history:
                self.performance_history[name].append(confidence)
                
                # Keep only recent history
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][-100:]
    
    def adapt_weights(self) -> None:
        """Adapt weights based on performance history"""
        
        for name, history in self.performance_history.items():
            if len(history) > 10:
                avg_performance = np.mean(history)
                
                # Adjust weight based on performance
                if avg_performance > 0.7:
                    self.weights[name] = min(0.4, self.weights[name] * 1.1)
                elif avg_performance < 0.5:
                    self.weights[name] = max(0.1, self.weights[name] * 0.9)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
    
    def train_ensemble(self, data_sequences: List[np.ndarray], regime_labels: List[str]) -> None:
        """Train ensemble detectors"""
        
        if not data_sequences or not regime_labels:
            return
        
        # Prepare training data
        features_list = []
        for data in data_sequences:
            features = self.extract_advanced_features(data)
            features_list.append(features)
        
        features_array = np.array(features_list)
        regimes_array = np.array(regime_labels)
        
        # Train SVM classifier
        self.detectors['svm_classifier'].train(features_array, regimes_array)
        
        # Train LSTM predictor
        self.detectors['lstm_predictor'].train(data_sequences, regimes_array)
        
        logger.info(f"Trained ensemble with {len(data_sequences)} sequences")
    
    def get_ensemble_status(self) -> Dict:
        """Get ensemble status"""
        
        return {
            'detectors': list(self.detectors.keys()),
            'weights': self.weights,
            'performance_history': {name: len(history) for name, history in self.performance_history.items()},
            'ensemble_ready': all(detector is not None for detector in self.detectors.values())
        }

def main():
    """Test the advanced regime detection system"""
    import matplotlib.pyplot as plt
    
    # Generate test data
    np.random.seed(42)
    n_periods = 200
    
    # Generate regime-specific data
    data = []
    true_regimes = []
    
    # Bull market (periods 0-60)
    for i in range(60):
        roi = 0.02 + np.random.normal(0, 0.01)
        risk = 0.10 + np.random.normal(0, 0.02)
        data.append([roi, risk])
        true_regimes.append('bull_market')
    
    # Bear market (periods 60-120)
    for i in range(60):
        roi = -0.01 + np.random.normal(0, 0.02)
        risk = 0.20 + np.random.normal(0, 0.03)
        data.append([roi, risk])
        true_regimes.append('bear_market')
    
    # Sideways market (periods 120-200)
    for i in range(80):
        roi = 0.005 + np.random.normal(0, 0.015)
        risk = 0.15 + np.random.normal(0, 0.025)
        data.append([roi, risk])
        true_regimes.append('sideways_market')
    
    data = np.array(data)
    
    # Initialize ensemble regime detector
    ensemble_detector = EnsembleRegimeDetector()
    
    # Test regime detection
    detected_regimes = []
    confidences = []
    
    for i in range(20, len(data)):
        window_data = data[:i+1]
        result = ensemble_detector.detect_regime(window_data)
        detected_regimes.append(result.regime)
        confidences.append(result.confidence)
    
    # Calculate accuracy
    accuracy = sum(1 for i, detected in enumerate(detected_regimes) 
                   if detected == true_regimes[i+20]) / len(detected_regimes)
    
    print(f"Advanced Regime Detection Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {np.mean(confidences):.2%}")
    print(f"Ensemble Status: {ensemble_detector.get_ensemble_status()}")

if __name__ == "__main__":
    main()
