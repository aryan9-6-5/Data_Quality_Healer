"""
Synthetic Data Generator
Generates realistic datasets with controlled data quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import os


class SyntheticDataGenerator:
    """
    Generates synthetic enterprise datasets with injected quality issues.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_transaction_data(self, 
                                  n_records: int = 10000,
                                  inject_issues: bool = False) -> pd.DataFrame:
        """
        Generate synthetic transaction/sales data.
        
        Args:
            n_records: Number of records to generate
            inject_issues: Whether to inject data quality issues
            
        Returns:
            DataFrame with transaction data
        """
        np.random.seed(self.seed)
        
        # Generate base data
        data = {
            'transaction_id': [f'TXN{str(i).zfill(6)}' for i in range(n_records)],
            'customer_id': np.random.randint(1000, 9999, n_records),
            'product_id': np.random.choice(['PROD_A', 'PROD_B', 'PROD_C', 'PROD_D'], n_records),
            'quantity': np.random.randint(1, 20, n_records),
            'unit_price': np.random.uniform(10, 500, n_records),
            'discount': np.random.uniform(0, 0.3, n_records),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'payment_method': np.random.choice(['Credit', 'Debit', 'Cash', 'PayPal'], n_records),
            'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        
        # Calculate total amount
        df['total_amount'] = df['quantity'] * df['unit_price'] * (1 - df['discount'])
        
        if inject_issues:
            df = self._inject_issues(df)
        
        return df
    
    def generate_sensor_data(self,
                            n_records: int = 5000,
                            inject_issues: bool = False) -> pd.DataFrame:
        """
        Generate synthetic IoT sensor data.
        """
        np.random.seed(self.seed)
        
        data = {
            'sensor_id': np.random.choice(['SENSOR_001', 'SENSOR_002', 'SENSOR_003'], n_records),
            'temperature': np.random.normal(25, 5, n_records),
            'humidity': np.random.uniform(30, 80, n_records),
            'pressure': np.random.normal(1013, 10, n_records),
            'vibration': np.random.exponential(2, n_records),
            'status': np.random.choice(['OK', 'WARNING', 'ERROR'], n_records, p=[0.8, 0.15, 0.05]),
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        
        if inject_issues:
            df = self._inject_issues(df)
        
        return df
    
    def generate_user_logs(self,
                          n_records: int = 8000,
                          inject_issues: bool = False) -> pd.DataFrame:
        """
        Generate synthetic user activity logs.
        """
        np.random.seed(self.seed)
        
        data = {
            'user_id': np.random.randint(10000, 99999, n_records),
            'session_id': [f'SESSION_{i}' for i in range(n_records)],
            'action': np.random.choice(['login', 'view', 'click', 'purchase', 'logout'], n_records),
            'page': np.random.choice(['home', 'product', 'cart', 'checkout', 'profile'], n_records),
            'duration_seconds': np.random.exponential(60, n_records),
            'device': np.random.choice(['desktop', 'mobile', 'tablet'], n_records),
            'browser': np.random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'], n_records),
            'timestamp': [datetime.now() - timedelta(minutes=i) for i in range(n_records)]
        }
        
        df = pd.DataFrame(data)
        
        if inject_issues:
            df = self._inject_issues(df)
        
        return df
    
    def _inject_issues(self, df: pd.DataFrame, 
                       issue_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inject various data quality issues into the dataframe.
        
        Available issue types:
        - missing_values
        - outliers
        - duplicates
        - drift
        - schema_change
        """
        if issue_types is None:
            issue_types = ['missing_values', 'outliers', 'duplicates']
        
        df_corrupted = df.copy()
        
        if 'missing_values' in issue_types:
            df_corrupted = self._inject_missing_values(df_corrupted)
        
        if 'outliers' in issue_types:
            df_corrupted = self._inject_outliers(df_corrupted)
        
        if 'duplicates' in issue_types:
            df_corrupted = self._inject_duplicates(df_corrupted)
        
        if 'drift' in issue_types:
            df_corrupted = self._inject_drift(df_corrupted)
        
        if 'schema_change' in issue_types:
            df_corrupted = self._inject_schema_change(df_corrupted)
        
        return df_corrupted
    
    def _inject_missing_values(self, df: pd.DataFrame, rate: float = 0.15) -> pd.DataFrame:
        """Inject missing values randomly."""
        df_copy = df.copy()
        
        # Select numeric and categorical columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            mask = np.random.random(len(df_copy)) < rate
            df_copy.loc[mask, col] = np.nan
        
        print(f"✓ Injected missing values (~{rate*100:.0f}% per column)")
        return df_copy
    
    def _inject_outliers(self, df: pd.DataFrame, rate: float = 0.05) -> pd.DataFrame:
        """Inject extreme outliers."""
        df_copy = df.copy()
        
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            n_outliers = int(len(df_copy) * rate)
            outlier_indices = np.random.choice(df_copy.index, n_outliers, replace=False)
            
            # Create extreme values (10-100x normal)
            multiplier = np.random.uniform(10, 100, n_outliers)
            df_copy.loc[outlier_indices, col] *= multiplier
        
        print(f"✓ Injected outliers (~{rate*100:.0f}% per column)")
        return df_copy
    
    def _inject_duplicates(self, df: pd.DataFrame, rate: float = 0.10) -> pd.DataFrame:
        """Inject duplicate rows."""
        n_duplicates = int(len(df) * rate)
        duplicate_indices = np.random.choice(df.index, n_duplicates, replace=True)
        duplicates = df.loc[duplicate_indices]
        
        df_copy = pd.concat([df, duplicates], ignore_index=True)
        
        print(f"✓ Injected {n_duplicates} duplicate rows")
        return df_copy
    
    def _inject_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject distribution drift in numeric columns."""
        df_copy = df.copy()
        
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        
        # Shift mean of some columns
        for col in numeric_cols[:2]:  # Drift first 2 numeric columns
            shift = df_copy[col].mean() * 2  # Shift by 200% of mean
            df_copy[col] = df_copy[col] + shift
        
        print(f"✓ Injected distribution drift in {min(2, len(numeric_cols))} columns")
        return df_copy
    
    def _inject_schema_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject schema changes."""
        df_copy = df.copy()
        
        # Add unexpected column
        df_copy['unexpected_column'] = np.random.randn(len(df_copy))
        
        # Change dtype of a column
        if 'customer_id' in df_copy.columns:
            df_copy['customer_id'] = df_copy['customer_id'].astype(str)
        
        print(f"✓ Injected schema changes (added column, changed dtype)")
        return df_copy
    
    def save_dataset(self, df: pd.DataFrame, name: str, output_dir: str = 'data/samples'):
        """Save dataset to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{name}.csv')
        df.to_csv(filepath, index=False)
        print(f"💾 Saved dataset to {filepath}")


if __name__ == "__main__":
    print("🏭 Synthetic Data Generator\n")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate clean transaction data
    print("Generating CLEAN transaction data...")
    clean_transactions = generator.generate_transaction_data(n_records=5000, inject_issues=False)
    generator.save_dataset(clean_transactions, 'transactions_clean')
    
    # Generate corrupted transaction data
    print("\nGenerating CORRUPTED transaction data...")
    corrupted_transactions = generator.generate_transaction_data(n_records=5000, inject_issues=True)
    generator.save_dataset(corrupted_transactions, 'transactions_corrupted')
    
    # Generate sensor data
    print("\nGenerating sensor data...")
    sensor_data = generator.generate_sensor_data(n_records=3000, inject_issues=True)
    generator.save_dataset(sensor_data, 'sensor_data')
    
    # Generate user logs
    print("\nGenerating user logs...")
    user_logs = generator.generate_user_logs(n_records=4000, inject_issues=True)
    generator.save_dataset(user_logs, 'user_logs')
    
    print("\n✅ All datasets generated successfully!")
    print(f"\nClean data shape: {clean_transactions.shape}")
    print(f"Corrupted data shape: {corrupted_transactions.shape}")
    print(f"\nSample data preview:")
    print(corrupted_transactions.head())