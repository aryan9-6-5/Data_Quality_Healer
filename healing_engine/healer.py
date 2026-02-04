"""
Healing Engine Module
Recommends and applies data quality fixes based on detected issues
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')


class DataHealer:
    """
    Recommends and applies healing strategies for data quality issues.
    Learns from user feedback to improve recommendations.
    """
    
    def __init__(self):
        self.healing_strategies = {
            'missing_values': self._heal_missing_values,
            'outliers': self._heal_outliers,
            'drift': self._heal_drift,
            'schema_change': self._heal_schema_change,
            'duplicates': self._heal_duplicates
        }
        
        # Track healing success rates for learning
        self.healing_history = {
            'missing_values': {'attempts': 0, 'successes': 0},
            'outliers': {'attempts': 0, 'successes': 0},
            'drift': {'attempts': 0, 'successes': 0},
            'schema_change': {'attempts': 0, 'successes': 0},
            'duplicates': {'attempts': 0, 'successes': 0}
        }
    
    def recommend(self, issue_type: str, data_profile: Dict) -> List[Dict]:
        """
        Recommend healing strategies for a given issue type.
        
        Args:
            issue_type: Type of data quality issue
            data_profile: Statistical profile of the data
            
        Returns:
            List of recommended strategies with confidence scores
        """
        if issue_type not in self.healing_strategies:
            return [{
                'strategy': 'manual_review',
                'confidence': 0.5,
                'description': 'Issue type not recognized. Manual review recommended.'
            }]
        
        recommendations = []
        
        if issue_type == 'missing_values':
            missing_rate = data_profile.get('missing_patterns', {}).get('missing_rate', 0)
            
            if missing_rate < 0.05:
                recommendations.append({
                    'strategy': 'drop_rows',
                    'confidence': 0.9,
                    'description': 'Few missing values - safe to drop rows',
                    'params': {}
                })
            elif missing_rate < 0.30:
                recommendations.append({
                    'strategy': 'impute_smart',
                    'confidence': 0.8,
                    'description': 'Use intelligent imputation (KNN or iterative)',
                    'params': {'method': 'knn', 'n_neighbors': 5}
                })
            else:
                recommendations.append({
                    'strategy': 'impute_simple',
                    'confidence': 0.6,
                    'description': 'High missing rate - use simple imputation',
                    'params': {'method': 'median'}
                })
        
        elif issue_type == 'outliers':
            recommendations.append({
                'strategy': 'winsorize',
                'confidence': 0.85,
                'description': 'Cap extreme values at percentiles',
                'params': {'lower': 0.01, 'upper': 0.99}
            })
            recommendations.append({
                'strategy': 'clip_iqr',
                'confidence': 0.75,
                'description': 'Clip values outside IQR range',
                'params': {'multiplier': 1.5}
            })
        
        elif issue_type == 'drift':
            recommendations.append({
                'strategy': 'retrain_alert',
                'confidence': 0.9,
                'description': 'Distribution has shifted - model retraining required',
                'params': {'alert_severity': 'high'}
            })
            recommendations.append({
                'strategy': 'normalize',
                'confidence': 0.7,
                'description': 'Apply normalization to current data',
                'params': {'method': 'standard'}
            })
        
        elif issue_type == 'schema_change':
            recommendations.append({
                'strategy': 'align_schema',
                'confidence': 0.8,
                'description': 'Align current schema with expected schema',
                'params': {}
            })
        
        elif issue_type == 'duplicates':
            recommendations.append({
                'strategy': 'drop_duplicates',
                'confidence': 0.95,
                'description': 'Remove duplicate rows',
                'params': {'keep': 'first'}
            })
        
        # Add historical success rate to confidence
        if issue_type in self.healing_history:
            history = self.healing_history[issue_type]
            if history['attempts'] > 0:
                success_rate = history['successes'] / history['attempts']
                for rec in recommendations:
                    rec['confidence'] *= (0.5 + 0.5 * success_rate)
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    def heal(self, data: pd.DataFrame, issue_type: str, 
             strategy: Optional[str] = None, params: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply healing strategy to data.
        
        Args:
            data: DataFrame to heal
            issue_type: Type of issue to heal
            strategy: Specific strategy to use (if None, uses best recommendation)
            params: Strategy parameters
            
        Returns:
            (healed_data, healing_report)
        """
        if issue_type not in self.healing_strategies:
            return data, {
                'success': False,
                'message': f'Unknown issue type: {issue_type}'
            }
        
        # Get strategy
        if strategy is None:
            recommendations = self.recommend(issue_type, {})
            if not recommendations:
                return data, {
                    'success': False,
                    'message': 'No recommendations available'
                }
            strategy = recommendations[0]['strategy']
            params = recommendations[0].get('params', {})
        
        # Apply healing
        healing_func = self.healing_strategies[issue_type]
        healed_data, report = healing_func(data, strategy, params or {})
        
        # Update history
        self.healing_history[issue_type]['attempts'] += 1
        if report.get('success', False):
            self.healing_history[issue_type]['successes'] += 1
        
        return healed_data, report
    
    def _heal_missing_values(self, data: pd.DataFrame, strategy: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Heal missing values."""
        original_missing = data.isna().sum().sum()
        
        try:
            if strategy == 'drop_rows':
                healed = data.dropna()
                rows_dropped = len(data) - len(healed)
                
                return healed, {
                    'success': True,
                    'strategy': strategy,
                    'rows_dropped': rows_dropped,
                    'missing_before': int(original_missing),
                    'missing_after': 0
                }
            
            elif strategy == 'impute_simple':
                method = params.get('method', 'median')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                healed = data.copy()
                imputer = SimpleImputer(strategy=method)
                healed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                
                return healed, {
                    'success': True,
                    'strategy': strategy,
                    'method': method,
                    'missing_before': int(original_missing),
                    'missing_after': int(healed.isna().sum().sum())
                }
            
            elif strategy == 'impute_smart':
                method = params.get('method', 'knn')
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                healed = data.copy()
                
                if method == 'knn':
                    n_neighbors = params.get('n_neighbors', 5)
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    healed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                
                return healed, {
                    'success': True,
                    'strategy': strategy,
                    'method': method,
                    'missing_before': int(original_missing),
                    'missing_after': int(healed.isna().sum().sum())
                }
            
            else:
                return data, {
                    'success': False,
                    'message': f'Unknown strategy: {strategy}'
                }
                
        except Exception as e:
            return data, {
                'success': False,
                'message': f'Healing failed: {str(e)}'
            }
    
    def _heal_outliers(self, data: pd.DataFrame, strategy: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Heal outliers."""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            healed = data.copy()
            outliers_capped = 0
            
            if strategy == 'winsorize':
                lower = params.get('lower', 0.01)
                upper = params.get('upper', 0.99)
                
                for col in numeric_cols:
                    lower_bound = healed[col].quantile(lower)
                    upper_bound = healed[col].quantile(upper)
                    
                    outliers_capped += ((healed[col] < lower_bound) | (healed[col] > upper_bound)).sum()
                    
                    healed[col] = healed[col].clip(lower=lower_bound, upper=upper_bound)
                
                return healed, {
                    'success': True,
                    'strategy': strategy,
                    'outliers_capped': int(outliers_capped),
                    'percentiles': (lower, upper)
                }
            
            elif strategy == 'clip_iqr':
                multiplier = params.get('multiplier', 1.5)
                
                for col in numeric_cols:
                    Q1 = healed[col].quantile(0.25)
                    Q3 = healed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR
                    
                    outliers_capped += ((healed[col] < lower_bound) | (healed[col] > upper_bound)).sum()
                    
                    healed[col] = healed[col].clip(lower=lower_bound, upper=upper_bound)
                
                return healed, {
                    'success': True,
                    'strategy': strategy,
                    'outliers_capped': int(outliers_capped),
                    'iqr_multiplier': multiplier
                }
            
            else:
                return data, {
                    'success': False,
                    'message': f'Unknown strategy: {strategy}'
                }
                
        except Exception as e:
            return data, {
                'success': False,
                'message': f'Healing failed: {str(e)}'
            }
    
    def _heal_drift(self, data: pd.DataFrame, strategy: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Handle distribution drift."""
        if strategy == 'retrain_alert':
            return data, {
                'success': True,
                'strategy': strategy,
                'message': 'ALERT: Model retraining required due to distribution drift',
                'severity': params.get('alert_severity', 'high')
            }
        
        elif strategy == 'normalize':
            method = params.get('method', 'standard')
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            healed = data.copy()
            
            if method == 'standard':
                healed[numeric_cols] = (healed[numeric_cols] - healed[numeric_cols].mean()) / healed[numeric_cols].std()
            
            return healed, {
                'success': True,
                'strategy': strategy,
                'method': method,
                'message': 'Data normalized to standard scale'
            }
        
        return data, {
            'success': False,
            'message': f'Unknown strategy: {strategy}'
        }
    
    def _heal_schema_change(self, data: pd.DataFrame, strategy: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Handle schema changes."""
        return data, {
            'success': True,
            'strategy': strategy,
            'message': 'Schema change detected - manual review recommended',
            'action_required': True
        }
    
    def _heal_duplicates(self, data: pd.DataFrame, strategy: str, params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Remove duplicates."""
        try:
            original_len = len(data)
            keep = params.get('keep', 'first')
            
            healed = data.drop_duplicates(keep=keep)
            duplicates_removed = original_len - len(healed)
            
            return healed, {
                'success': True,
                'strategy': strategy,
                'duplicates_removed': duplicates_removed,
                'keep': keep
            }
            
        except Exception as e:
            return data, {
                'success': False,
                'message': f'Healing failed: {str(e)}'
            }
    
    def record_feedback(self, issue_type: str, success: bool):
        """
        Record user feedback on healing success.
        Used for learning and improving recommendations.
        """
        if issue_type in self.healing_history:
            if success:
                self.healing_history[issue_type]['successes'] += 1
    
    def get_healing_stats(self) -> pd.DataFrame:
        """Get statistics on healing performance."""
        stats_data = []
        
        for issue_type, history in self.healing_history.items():
            if history['attempts'] > 0:
                success_rate = history['successes'] / history['attempts']
            else:
                success_rate = 0.0
            
            stats_data.append({
                'issue_type': issue_type,
                'attempts': history['attempts'],
                'successes': history['successes'],
                'success_rate': success_rate
            })
        
        return pd.DataFrame(stats_data)


if __name__ == "__main__":
    print("🔧 Data Healer Example\n")
    
    # Create sample data with issues
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 15000, 100),
        'score': np.random.uniform(0, 100, 100)
    })
    
    # Inject missing values
    data.loc[np.random.choice(data.index, 15), 'income'] = np.nan
    
    # Inject outliers
    data.loc[np.random.choice(data.index, 5), 'income'] = 500000
    
    print("Original Data Issues:")
    print(f"Missing values: {data.isna().sum().sum()}")
    print(f"Income range: {data['income'].min():.0f} - {data['income'].max():.0f}")
    
    # Initialize healer
    healer = DataHealer()
    
    # Heal missing values
    print("\n🔧 Healing missing values...")
    healed_data, report = healer.heal(data, 'missing_values', strategy='impute_simple')
    print(f"Report: {report}")
    
    # Heal outliers
    print("\n🔧 Healing outliers...")
    healed_data, report = healer.heal(healed_data, 'outliers', strategy='winsorize')
    print(f"Report: {report}")
    
    print("\nHealed Data:")
    print(f"Missing values: {healed_data.isna().sum().sum()}")
    print(f"Income range: {healed_data['income'].min():.0f} - {healed_data['income'].max():.0f}")
    
    print("\n✅ Data healing working!")