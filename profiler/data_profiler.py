"""
Data Profiler Module
Extracts statistical features from datasets for anomaly detection
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataProfiler:
    """
    Profiles datasets to extract statistical features.
    These features become the input for anomaly detection.
    """
    
    def __init__(self):
        self.profile_history = []
        
    def profile(self, data: pd.DataFrame, dataset_name: str = "dataset") -> Dict:
        """
        Generate comprehensive statistical profile of dataset.
        
        Args:
            data: Input DataFrame
            dataset_name: Name identifier for the dataset
            
        Returns:
            Dictionary containing statistical features
        """
        profile = {
            'dataset_name': dataset_name,
            'timestamp': pd.Timestamp.now(),
            'n_rows': len(data),
            'n_cols': len(data.columns),
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'numeric_features': self._profile_numeric(data),
            'categorical_features': self._profile_categorical(data),
            'missing_patterns': self._analyze_missing(data),
            'schema_hash': self._compute_schema_hash(data),
            'correlation_signature': self._compute_correlation_signature(data)
        }
        
        self.profile_history.append(profile)
        return profile
    
    def _profile_numeric(self, data: pd.DataFrame) -> Dict:
        """Extract statistical features from numeric columns."""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        features = {}
        
        for col in numeric_cols:
            series = data[col].dropna()
            
            if len(series) == 0:
                features[col] = self._empty_numeric_profile()
                continue
                
            features[col] = {
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'q25': float(series.quantile(0.25)),
                'q75': float(series.quantile(0.75)),
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'missing_rate': float(data[col].isna().mean()),
                'unique_count': int(series.nunique()),
                'zero_rate': float((series == 0).mean()),
                'negative_rate': float((series < 0).mean())
            }
            
        return features
    
    def _profile_categorical(self, data: pd.DataFrame) -> Dict:
        """Extract features from categorical columns."""
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        features = {}
        
        for col in cat_cols:
            series = data[col].dropna()
            
            if len(series) == 0:
                features[col] = self._empty_categorical_profile()
                continue
                
            value_counts = series.value_counts()
            
            features[col] = {
                'unique_count': int(series.nunique()),
                'missing_rate': float(data[col].isna().mean()),
                'mode': str(series.mode().iloc[0]) if len(series.mode()) > 0 else None,
                'mode_frequency': float(value_counts.iloc[0] / len(series)) if len(value_counts) > 0 else 0,
                'cardinality_ratio': float(series.nunique() / len(series)) if len(series) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
            
        return features
    
    def _analyze_missing(self, data: pd.DataFrame) -> Dict:
        """Analyze missing value patterns."""
        missing_counts = data.isna().sum()
        total_missing = missing_counts.sum()
        
        return {
            'total_missing': int(total_missing),
            'missing_rate': float(total_missing / (len(data) * len(data.columns))),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_correlation': self._missing_correlation(data)
        }
    
    def _missing_correlation(self, data: pd.DataFrame) -> float:
        """
        Detect if missing values are correlated across columns.
        High correlation suggests systematic missing data.
        """
        missing_matrix = data.isna().astype(int)
        if missing_matrix.sum().sum() == 0:
            return 0.0
            
        corr_matrix = missing_matrix.corr()
        # Get upper triangle (excluding diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        return float(upper_triangle.stack().mean())
    
    def _compute_schema_hash(self, data: pd.DataFrame) -> str:
        """Generate hash signature of schema (column names + types)."""
        schema_str = '_'.join([f"{col}:{dtype}" for col, dtype in zip(data.columns, data.dtypes)])
        return str(hash(schema_str))
    
    def _compute_correlation_signature(self, data: pd.DataFrame) -> Dict:
        """
        Compute correlation signature for numeric columns.
        Useful for detecting distribution drift.
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {}
            
        corr_matrix = numeric_data.corr()
        
        return {
            'mean_correlation': float(corr_matrix.abs().mean().mean()),
            'max_correlation': float(corr_matrix.abs().max().max()),
            'high_correlation_pairs': self._get_high_correlations(corr_matrix)
        }
    
    def _get_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Tuple]:
        """Find highly correlated column pairs."""
        high_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        float(corr_matrix.iloc[i, j])
                    ))
                    
        return high_corr
    
    def _empty_numeric_profile(self) -> Dict:
        """Return empty profile for numeric column with no data."""
        return {
            'mean': None, 'median': None, 'std': None,
            'min': None, 'max': None, 'q25': None, 'q75': None,
            'skewness': None, 'kurtosis': None, 'missing_rate': 1.0,
            'unique_count': 0, 'zero_rate': None, 'negative_rate': None
        }
    
    def _empty_categorical_profile(self) -> Dict:
        """Return empty profile for categorical column with no data."""
        return {
            'unique_count': 0, 'missing_rate': 1.0, 'mode': None,
            'mode_frequency': None, 'cardinality_ratio': None, 'top_5_values': {}
        }
    
    def extract_feature_vector(self, profile: Dict) -> np.ndarray:
        """
        Convert profile to feature vector for ML models.
        This is used as input to anomaly detector.
        """
        features = []
        
        # Dataset-level features
        features.extend([
            profile['n_rows'],
            profile['n_cols'],
            profile['missing_patterns']['missing_rate'],
            profile['missing_patterns']['missing_correlation']
        ])
        
        # Aggregate numeric features
        numeric_features = profile['numeric_features']
        if numeric_features:
            means = [f['mean'] for f in numeric_features.values() if f['mean'] is not None]
            stds = [f['std'] for f in numeric_features.values() if f['std'] is not None]
            skews = [f['skewness'] for f in numeric_features.values() if f['skewness'] is not None]
            
            features.extend([
                np.mean(means) if means else 0,
                np.std(means) if means else 0,
                np.mean(stds) if stds else 0,
                np.mean(skews) if skews else 0
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Aggregate categorical features
        cat_features = profile['categorical_features']
        if cat_features:
            cardinalities = [f['cardinality_ratio'] for f in cat_features.values() 
                           if f['cardinality_ratio'] is not None]
            features.append(np.mean(cardinalities) if cardinalities else 0)
        else:
            features.append(0)
            
        # Correlation signature
        if profile['correlation_signature']:
            features.append(profile['correlation_signature']['mean_correlation'])
        else:
            features.append(0)
            
        return np.array(features)
    
    def compare_profiles(self, profile1: Dict, profile2: Dict) -> Dict:
        """
        Compare two profiles to detect drift.
        Returns drift metrics.
        """
        drift_report = {
            'schema_changed': profile1['schema_hash'] != profile2['schema_hash'],
            'row_count_change': profile2['n_rows'] - profile1['n_rows'],
            'column_count_change': profile2['n_cols'] - profile1['n_cols'],
            'numeric_drifts': {},
            'categorical_drifts': {}
        }
        
        # Compare numeric columns
        for col in set(profile1['numeric_features'].keys()) & set(profile2['numeric_features'].keys()):
            p1 = profile1['numeric_features'][col]
            p2 = profile2['numeric_features'][col]
            
            if p1['mean'] is not None and p2['mean'] is not None:
                drift_report['numeric_drifts'][col] = {
                    'mean_change': abs(p2['mean'] - p1['mean']),
                    'std_change': abs(p2['std'] - p1['std']) if p1['std'] and p2['std'] else None
                }
        
        return drift_report


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'score': np.random.uniform(0, 100, 1000)
    })
    
    # Profile the data
    profiler = DataProfiler()
    profile = profiler.profile(sample_data, "sample_dataset")
    
    print("✅ Data Profile Generated")
    print(f"Rows: {profile['n_rows']}")
    print(f"Columns: {profile['n_cols']}")
    print(f"Missing Rate: {profile['missing_patterns']['missing_rate']:.2%}")
    
    # Extract feature vector
    feature_vector = profiler.extract_feature_vector(profile)
    print(f"\n📊 Feature Vector Shape: {feature_vector.shape}")
    print(f"Features: {feature_vector}")