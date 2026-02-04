"""
Anomaly Detection Module
Uses unsupervised ML to detect data quality issues
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Detects anomalies in data profiles using unsupervised ML.
    Supports multiple detection algorithms.
    """
    
    def __init__(self, contamination: float = 0.1, method: str = 'isolation_forest'):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (0.1 = 10%)
            method: Detection algorithm ('isolation_forest', 'one_class_svm', 'elliptic_envelope')
        """
        self.contamination = contamination
        self.method = method
        self.scaler = StandardScaler()
        self.detector = None
        self.is_trained = False
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize the anomaly detection model."""
        if self.method == 'isolation_forest':
            self.detector = IsolationForest(
                contamination=self.contamination,
                n_estimators=100,
                max_samples=256,
                random_state=42,
                n_jobs=-1
            )
        elif self.method == 'one_class_svm':
            self.detector = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='auto'
            )
        elif self.method == 'elliptic_envelope':
            self.detector = EllipticEnvelope(
                contamination=self.contamination,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def fit(self, feature_vectors: np.ndarray):
        """
        Train the anomaly detector on historical clean data.
        
        Args:
            feature_vectors: Array of feature vectors from data profiles
        """
        # Normalize features
        feature_vectors_scaled = self.scaler.fit_transform(feature_vectors)
        
        # Fit detector
        self.detector.fit(feature_vectors_scaled)
        self.is_trained = True
        
        print(f"✅ Anomaly detector trained on {len(feature_vectors)} samples")
        
    def detect(self, feature_vector: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect if a feature vector represents anomalous data.
        
        Args:
            feature_vector: Single feature vector to check
            
        Returns:
            (is_anomaly, anomaly_score, details)
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection. Call fit() first.")
        
        # Reshape if needed
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Normalize
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.detector.predict(feature_vector_scaled)[0]
        is_anomaly = prediction == -1
        
        # Get anomaly score
        if hasattr(self.detector, 'decision_function'):
            score = self.detector.decision_function(feature_vector_scaled)[0]
        elif hasattr(self.detector, 'score_samples'):
            score = self.detector.score_samples(feature_vector_scaled)[0]
        else:
            score = float(prediction)
        
        # Build details
        details = {
            'method': self.method,
            'anomaly_score': float(score),
            'threshold': self.contamination,
            'is_anomaly': bool(is_anomaly)
        }
        
        return is_anomaly, float(score), details
    
    def detect_batch(self, feature_vectors: np.ndarray) -> List[Dict]:
        """
        Detect anomalies in a batch of feature vectors.
        
        Args:
            feature_vectors: Array of feature vectors
            
        Returns:
            List of detection results
        """
        if not self.is_trained:
            raise ValueError("Detector must be trained before detection.")
        
        feature_vectors_scaled = self.scaler.transform(feature_vectors)
        predictions = self.detector.predict(feature_vectors_scaled)
        
        if hasattr(self.detector, 'decision_function'):
            scores = self.detector.decision_function(feature_vectors_scaled)
        elif hasattr(self.detector, 'score_samples'):
            scores = self.detector.score_samples(feature_vectors_scaled)
        else:
            scores = predictions.astype(float)
        
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            results.append({
                'index': i,
                'is_anomaly': pred == -1,
                'anomaly_score': float(score),
                'method': self.method
            })
        
        return results
    
    def save(self, filepath: str):
        """Save trained detector to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained detector.")
        
        model_data = {
            'detector': self.detector,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'method': self.method,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Detector saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained detector from disk."""
        model_data = joblib.load(filepath)
        
        self.detector = model_data['detector']
        self.scaler = model_data['scaler']
        self.contamination = model_data['contamination']
        self.method = model_data['method']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Detector loaded from {filepath}")


class DataQualityAnomalyDetector:
    """
    High-level detector that identifies specific types of data quality issues.
    """
    
    def __init__(self):
        self.detectors = {
            'missing_value_flood': self._detect_missing_flood,
            'value_explosion': self._detect_value_explosion,
            'distribution_drift': self._detect_distribution_drift,
            'schema_change': self._detect_schema_change,
            'duplicate_surge': self._detect_duplicate_surge
        }
    
    def detect_all(self, current_profile: Dict, historical_profiles: List[Dict]) -> Dict:
        """
        Run all anomaly detections on current profile.
        
        Args:
            current_profile: Current data profile
            historical_profiles: List of historical profiles for comparison
            
        Returns:
            Dictionary of detected anomalies
        """
        anomalies = {}
        
        for anomaly_type, detector_func in self.detectors.items():
            result = detector_func(current_profile, historical_profiles)
            if result['detected']:
                anomalies[anomaly_type] = result
        
        return anomalies
    
    def _detect_missing_flood(self, current: Dict, historical: List[Dict]) -> Dict:
        """Detect sudden increase in missing values."""
        current_missing_rate = current['missing_patterns']['missing_rate']
        
        if len(historical) > 0:
            historical_missing_rates = [p['missing_patterns']['missing_rate'] for p in historical]
            avg_historical = np.mean(historical_missing_rates)
            std_historical = np.std(historical_missing_rates)
            
            # Alert if current rate is >3 std above historical
            threshold = avg_historical + 3 * std_historical
            detected = current_missing_rate > threshold and current_missing_rate > 0.3
        else:
            # No history - alert if >30% missing
            detected = current_missing_rate > 0.3
        
        return {
            'detected': detected,
            'current_missing_rate': current_missing_rate,
            'threshold': threshold if len(historical) > 0 else 0.3,
            'severity': 'high' if current_missing_rate > 0.5 else 'medium'
        }
    
    def _detect_value_explosion(self, current: Dict, historical: List[Dict]) -> Dict:
        """Detect sudden spike in value ranges."""
        detected = False
        details = []
        
        current_numeric = current['numeric_features']
        
        if len(historical) > 0 and current_numeric:
            for col in current_numeric:
                current_max = current_numeric[col].get('max')
                
                if current_max is None:
                    continue
                
                # Get historical max values for this column
                historical_maxes = []
                for prof in historical:
                    if col in prof['numeric_features']:
                        max_val = prof['numeric_features'][col].get('max')
                        if max_val is not None:
                            historical_maxes.append(max_val)
                
                if historical_maxes:
                    avg_max = np.mean(historical_maxes)
                    
                    # Alert if current max is >10x historical average
                    if current_max > 10 * avg_max:
                        detected = True
                        details.append({
                            'column': col,
                            'current_max': current_max,
                            'historical_avg_max': avg_max,
                            'ratio': current_max / avg_max
                        })
        
        return {
            'detected': detected,
            'affected_columns': details,
            'severity': 'high' if detected else 'none'
        }
    
    def _detect_distribution_drift(self, current: Dict, historical: List[Dict]) -> Dict:
        """Detect drift in statistical distributions."""
        detected = False
        drifted_columns = []
        
        current_numeric = current['numeric_features']
        
        if len(historical) > 0 and current_numeric:
            # Use most recent historical profile
            recent_historical = historical[-1]
            
            for col in current_numeric:
                if col not in recent_historical['numeric_features']:
                    continue
                
                current_stats = current_numeric[col]
                historical_stats = recent_historical['numeric_features'][col]
                
                if current_stats['mean'] is None or historical_stats['mean'] is None:
                    continue
                
                # Check mean shift
                mean_change = abs(current_stats['mean'] - historical_stats['mean'])
                historical_std = historical_stats['std'] if historical_stats['std'] else 1
                
                # Alert if mean shifted by >3 standard deviations
                if mean_change > 3 * historical_std:
                    detected = True
                    drifted_columns.append({
                        'column': col,
                        'mean_change': mean_change,
                        'historical_std': historical_std
                    })
        
        return {
            'detected': detected,
            'drifted_columns': drifted_columns,
            'severity': 'medium' if detected else 'none'
        }
    
    def _detect_schema_change(self, current: Dict, historical: List[Dict]) -> Dict:
        """Detect schema changes (columns added/removed)."""
        detected = False
        changes = {}
        
        if len(historical) > 0:
            recent_historical = historical[-1]
            
            current_cols = set(current['columns'])
            historical_cols = set(recent_historical['columns'])
            
            added = current_cols - historical_cols
            removed = historical_cols - current_cols
            
            if added or removed:
                detected = True
                changes = {
                    'added_columns': list(added),
                    'removed_columns': list(removed)
                }
        
        return {
            'detected': detected,
            'changes': changes,
            'severity': 'high' if detected else 'none'
        }
    
    def _detect_duplicate_surge(self, current: Dict, historical: List[Dict]) -> Dict:
        """Detect sudden increase in duplicate patterns."""
        # This would require actual data, not just profile
        # For now, return placeholder
        return {
            'detected': False,
            'severity': 'none',
            'note': 'Requires raw data analysis'
        }


if __name__ == "__main__":
    # Example usage
    print("🔍 Anomaly Detector Example\n")
    
    # Generate synthetic feature vectors (representing clean data)
    np.random.seed(42)
    clean_features = np.random.randn(100, 10)  # 100 samples, 10 features
    
    # Create detector
    detector = AnomalyDetector(contamination=0.1, method='isolation_forest')
    
    # Train on clean data
    detector.fit(clean_features)
    
    # Test on clean sample
    test_clean = np.random.randn(10)
    is_anomaly, score, details = detector.detect(test_clean)
    print(f"Clean sample - Anomaly: {is_anomaly}, Score: {score:.3f}")
    
    # Test on anomalous sample (extreme values)
    test_anomaly = np.random.randn(10) * 10 + 20
    is_anomaly, score, details = detector.detect(test_anomaly)
    print(f"Anomalous sample - Anomaly: {is_anomaly}, Score: {score:.3f}")
    
    print("\n✅ Anomaly detection working!")