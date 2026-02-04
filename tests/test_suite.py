"""
Test Suite for Data Quality Healer
"""

import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from profiler.data_profiler import DataProfiler
from anomaly_detector.detector import AnomalyDetector
from issue_classifier.classifier import IssueClassifier, generate_synthetic_training_data
from healing_engine.healer import DataHealer


class TestDataProfiler(unittest.TestCase):
    """Test cases for DataProfiler."""
    
    def setUp(self):
        self.profiler = DataProfiler()
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
        })
    
    def test_profile_creation(self):
        """Test profile generation."""
        profile = self.profiler.profile(self.sample_data, "test_data")
        
        self.assertEqual(profile['n_rows'], 5)
        self.assertEqual(profile['n_cols'], 3)
        self.assertIn('numeric_features', profile)
        self.assertIn('categorical_features', profile)
    
    def test_feature_vector_extraction(self):
        """Test feature vector extraction."""
        profile = self.profiler.profile(self.sample_data, "test_data")
        feature_vector = self.profiler.extract_feature_vector(profile)
        
        self.assertIsInstance(feature_vector, np.ndarray)
        self.assertGreater(len(feature_vector), 0)
    
    def test_missing_value_detection(self):
        """Test missing value pattern detection."""
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'income'] = np.nan
        
        profile = self.profiler.profile(data_with_missing, "test_missing")
        
        self.assertGreater(profile['missing_patterns']['total_missing'], 0)
        self.assertGreater(profile['missing_patterns']['missing_rate'], 0)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector."""
    
    def setUp(self):
        self.detector = AnomalyDetector(contamination=0.1)
        # Generate clean training data
        self.clean_features = np.random.randn(100, 10)
    
    def test_training(self):
        """Test detector training."""
        self.detector.fit(self.clean_features)
        self.assertTrue(self.detector.is_trained)
    
    def test_anomaly_detection(self):
        """Test anomaly detection on clean vs anomalous data."""
        self.detector.fit(self.clean_features)
        
        # Clean sample
        clean_sample = np.random.randn(10)
        is_anomaly, score, details = self.detector.detect(clean_sample)
        
        self.assertIsInstance(is_anomaly, bool)
        self.assertIsInstance(score, float)
        
        # Anomalous sample
        anomalous_sample = np.random.randn(10) * 10 + 20
        is_anomaly_2, score_2, details_2 = self.detector.detect(anomalous_sample)
        
        # Anomalous should have lower (more negative) score
        self.assertLess(score_2, score)


class TestIssueClassifier(unittest.TestCase):
    """Test cases for IssueClassifier."""
    
    def setUp(self):
        self.classifier = IssueClassifier()
        self.X_train, self.y_train = generate_synthetic_training_data(n_samples=600)
    
    def test_training(self):
        """Test classifier training."""
        accuracy = self.classifier.train(self.X_train, self.y_train)
        
        self.assertTrue(self.classifier.is_trained)
        self.assertGreater(accuracy, 0.5)  # Should be better than random
    
    def test_prediction(self):
        """Test issue classification."""
        self.classifier.train(self.X_train, self.y_train)
        
        # Test missing values pattern
        test_features = np.zeros(11)
        test_features[1] = 0.7  # High missing rate
        
        issue_type, confidence, probs = self.classifier.predict(test_features)
        
        self.assertIn(issue_type, self.classifier.ISSUE_TYPES)
        self.assertGreaterEqual(confidence, 0)
        self.assertLessEqual(confidence, 1)


class TestDataHealer(unittest.TestCase):
    """Test cases for DataHealer."""
    
    def setUp(self):
        self.healer = DataHealer()
        self.sample_data = pd.DataFrame({
            'age': [25, 30, np.nan, 40, 45],
            'income': [50000, 60000, 70000, 1000000, 90000],  # Outlier
            'score': [85, 90, 95, 88, 92]
        })
    
    def test_missing_value_healing(self):
        """Test missing value imputation."""
        healed_data, report = self.healer.heal(
            self.sample_data,
            'missing_values',
            strategy='impute_simple',
            params={'method': 'median'}
        )
        
        self.assertTrue(report['success'])
        self.assertEqual(healed_data.isna().sum().sum(), 0)
    
    def test_outlier_healing(self):
        """Test outlier capping."""
        healed_data, report = self.healer.heal(
            self.sample_data,
            'outliers',
            strategy='winsorize',
            params={'lower': 0.01, 'upper': 0.99}
        )
        
        self.assertTrue(report['success'])
        # Income outlier should be capped
        self.assertLess(healed_data['income'].max(), 200000)
    
    def test_duplicate_healing(self):
        """Test duplicate removal."""
        data_with_dupes = pd.concat([self.sample_data, self.sample_data.iloc[:2]], ignore_index=True)
        
        healed_data, report = self.healer.heal(
            data_with_dupes,
            'duplicates',
            strategy='drop_duplicates'
        )
        
        self.assertTrue(report['success'])
        self.assertLess(len(healed_data), len(data_with_dupes))
    
    def test_recommendations(self):
        """Test healing recommendations."""
        profile = {
            'missing_patterns': {'missing_rate': 0.2},
            'numeric_features': {},
            'categorical_features': {}
        }
        
        recommendations = self.healer.recommend('missing_values', profile)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        self.assertIn('confidence', recommendations[0])


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def setUp(self):
        self.profiler = DataProfiler()
        self.detector = AnomalyDetector(contamination=0.1)
        self.classifier = IssueClassifier()
        self.healer = DataHealer()
        
        # Train models
        clean_features = np.random.randn(100, 10)
        self.detector.fit(clean_features)
        
        X_train, y_train = generate_synthetic_training_data(n_samples=600)
        self.classifier.train(X_train, y_train)
    
    def test_complete_pipeline(self):
        """Test end-to-end pipeline."""
        # Create test data with issues
        test_data = pd.DataFrame({
            'value1': np.random.randn(100),
            'value2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Inject issues
        test_data.loc[0:10, 'value1'] = np.nan
        
        # Profile
        profile = self.profiler.profile(test_data, "test")
        
        # Extract features
        feature_vector = self.profiler.extract_feature_vector(profile)
        
        # Detect anomaly
        is_anomaly, score, details = self.detector.detect(feature_vector)
        
        # Classify if anomaly
        if is_anomaly:
            classifier_features = np.random.randn(11)
            issue_type, confidence, probs = self.classifier.predict(classifier_features)
            
            self.assertIn(issue_type, self.classifier.ISSUE_TYPES)
        
        # Heal
        healed_data, report = self.healer.heal(test_data, 'missing_values')
        
        self.assertTrue(report['success'])


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProfiler))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestIssueClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestDataHealer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)