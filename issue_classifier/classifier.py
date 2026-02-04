"""
Issue Classification Module
Classifies detected anomalies into specific issue types using supervised ML
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class IssueClassifier:
    """
    Supervised classifier for categorizing data quality issues.
    
    Issue Types:
    - missing_values: Sudden increase in null/missing data
    - outliers: Extreme values outside normal range
    - drift: Distribution shift in feature statistics
    - schema_change: Column additions/removals or type changes
    - duplicates: Increase in duplicate records
    """
    
    ISSUE_TYPES = [
        'missing_values',
        'outliers',
        'drift',
        'schema_change',
        'duplicates',
        'normal'  # No issue
    ]
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize issue classifier.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.classifier = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the classification model."""
        if self.model_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Handle class imbalance
            )
        elif self.model_type == 'gradient_boosting':
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                learning_rate=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, anomaly_data: Dict) -> np.ndarray:
        """
        Convert anomaly detection data to feature vector for classification.
        
        Args:
            anomaly_data: Dictionary with anomaly characteristics
            
        Returns:
            Feature vector
        """
        features = []
        
        # Anomaly score features
        features.append(anomaly_data.get('anomaly_score', 0))
        
        # Missing value indicators
        features.append(anomaly_data.get('missing_rate', 0))
        features.append(anomaly_data.get('missing_rate_change', 0))
        
        # Statistical features
        features.append(anomaly_data.get('mean_change', 0))
        features.append(anomaly_data.get('std_change', 0))
        features.append(anomaly_data.get('skewness_change', 0))
        
        # Outlier indicators
        features.append(anomaly_data.get('max_value_ratio', 1))
        features.append(anomaly_data.get('min_value_ratio', 1))
        
        # Schema indicators
        features.append(anomaly_data.get('column_count_change', 0))
        features.append(anomaly_data.get('dtype_changes', 0))
        
        # Correlation features
        features.append(anomaly_data.get('correlation_change', 0))
        
        return np.array(features)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the issue classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (issue types)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        print(f"✅ Issue Classifier trained")
        print(f"Model: {self.model_type}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        return accuracy
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Predict issue type for given features.
        
        Args:
            features: Feature vector or matrix
            
        Returns:
            (predicted_issue, confidence, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction.")
        
        # Reshape if single sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Predict
        prediction_encoded = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        # Decode prediction
        predicted_issue = self.label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = probabilities[prediction_encoded]
        
        # Build probability dict
        prob_dict = {
            issue: float(prob) 
            for issue, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        return predicted_issue, float(confidence), prob_dict
    
    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """
        Predict issue types for multiple samples.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction.")
        
        predictions_encoded = self.classifier.predict(features)
        probabilities = self.classifier.predict_proba(features)
        
        results = []
        for pred_enc, probs in zip(predictions_encoded, probabilities):
            issue = self.label_encoder.inverse_transform([pred_enc])[0]
            confidence = probs[pred_enc]
            
            prob_dict = {
                issue_type: float(prob)
                for issue_type, prob in zip(self.label_encoder.classes_, probs)
            }
            
            results.append({
                'issue_type': issue,
                'confidence': float(confidence),
                'probabilities': prob_dict
            })
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        Only available for tree-based models.
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained first.")
        
        if not hasattr(self.classifier, 'feature_importances_'):
            raise ValueError("Feature importance not available for this model type.")
        
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(self.classifier.feature_importances_))],
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: str):
        """Save trained classifier to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier.")
        
        model_data = {
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Classifier saved to {filepath}")
    
    def load(self, filepath: str):
        """Load trained classifier from disk."""
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        print(f"✅ Classifier loaded from {filepath}")


def generate_synthetic_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for issue classification.
    
    This simulates different data quality issues with characteristic features.
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    samples_per_class = n_samples // 6
    
    # Class 1: Missing Values
    # High missing_rate, high missing_rate_change
    for _ in range(samples_per_class):
        features = np.random.randn(11)
        features[1] = np.random.uniform(0.3, 0.9)  # High missing rate
        features[2] = np.random.uniform(0.2, 0.5)  # High change
        X.append(features)
        y.append('missing_values')
    
    # Class 2: Outliers
    # High max_value_ratio, high anomaly_score
    for _ in range(samples_per_class):
        features = np.random.randn(11)
        features[0] = np.random.uniform(-5, -1)    # High anomaly score
        features[6] = np.random.uniform(5, 50)     # High max ratio
        X.append(features)
        y.append('outliers')
    
    # Class 3: Drift
    # High mean_change, std_change
    for _ in range(samples_per_class):
        features = np.random.randn(11)
        features[3] = np.random.uniform(2, 10)     # Mean change
        features[4] = np.random.uniform(2, 8)      # Std change
        features[10] = np.random.uniform(0.3, 0.8) # Correlation change
        X.append(features)
        y.append('drift')
    
    # Class 4: Schema Change
    # Non-zero column_count_change, dtype_changes
    for _ in range(samples_per_class):
        features = np.random.randn(11)
        features[8] = np.random.choice([-3, -2, -1, 1, 2, 3])  # Column change
        features[9] = np.random.randint(1, 5)      # Dtype changes
        X.append(features)
        y.append('schema_change')
    
    # Class 5: Duplicates
    # Moderate anomaly score, some correlation change
    for _ in range(samples_per_class):
        features = np.random.randn(11)
        features[0] = np.random.uniform(-2, -0.5)  # Moderate anomaly
        features[10] = np.random.uniform(0.1, 0.3) # Some correlation
        X.append(features)
        y.append('duplicates')
    
    # Class 6: Normal (no issues)
    for _ in range(samples_per_class):
        features = np.random.randn(11) * 0.5  # Small variations
        X.append(features)
        y.append('normal')
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("🎯 Issue Classifier Example\n")
    
    # Generate synthetic training data
    X_train, y_train = generate_synthetic_training_data(n_samples=1200)
    print(f"Generated {len(X_train)} training samples")
    print(f"Classes: {np.unique(y_train)}\n")
    
    # Train classifier
    classifier = IssueClassifier(model_type='random_forest')
    classifier.train(X_train, y_train)
    
    # Test prediction
    print("\n📊 Test Predictions:")
    
    # Test case 1: Missing values pattern
    test_missing = np.zeros(11)
    test_missing[1] = 0.6  # High missing rate
    test_missing[2] = 0.4  # High change
    
    issue, conf, probs = classifier.predict(test_missing)
    print(f"\nTest 1 - Missing Values Pattern:")
    print(f"Predicted: {issue} (confidence: {conf:.2%})")
    
    # Test case 2: Outlier pattern
    test_outlier = np.zeros(11)
    test_outlier[0] = -3.5  # High anomaly score
    test_outlier[6] = 20    # High max ratio
    
    issue, conf, probs = classifier.predict(test_outlier)
    print(f"\nTest 2 - Outlier Pattern:")
    print(f"Predicted: {issue} (confidence: {conf:.2%})")
    
    print("\n✅ Issue classification working!")