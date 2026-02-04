"""
Main Entry Point for Data Quality Healer
Command-line interface for running the healing pipeline
"""

import argparse
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from profiler.data_profiler import DataProfiler
from anomaly_detector.detector import AnomalyDetector, DataQualityAnomalyDetector
from issue_classifier.classifier import IssueClassifier, generate_synthetic_training_data
from healing_engine.healer import DataHealer


class DataQualityHealer:
    """
    Main orchestrator for the data quality healing pipeline.
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the healing system."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.profiler = DataProfiler()
        self.anomaly_detector = AnomalyDetector(
            contamination=self.config['anomaly_detection']['contamination'],
            method='isolation_forest'
        )
        self.issue_classifier = IssueClassifier(model_type='random_forest')
        self.healer = DataHealer()
        self.quality_detector = DataQualityAnomalyDetector()
        
        # Storage for profiles
        self.profile_history = []
        
        print("🚑 Data Quality Healer initialized")
    
    def train_models(self, historical_data_paths: list = None):
        """
        Train anomaly detector and issue classifier.
        
        Args:
            historical_data_paths: List of paths to clean historical data
        """
        print("\n📚 Training ML models...")
        
        # Generate synthetic training data for classifier
        X_train, y_train = generate_synthetic_training_data(n_samples=1200)
        accuracy = self.issue_classifier.train(X_train, y_train)
        
        # Train anomaly detector if historical data provided
        if historical_data_paths:
            feature_vectors = []
            for path in historical_data_paths:
                data = pd.read_csv(path)
                profile = self.profiler.profile(data)
                feature_vector = self.profiler.extract_feature_vector(profile)
                feature_vectors.append(feature_vector)
                self.profile_history.append(profile)
            
            if feature_vectors:
                self.anomaly_detector.fit(np.array(feature_vectors))
        else:
            # Generate synthetic clean data for anomaly detector
            print("No historical data provided. Using synthetic baseline...")
            synthetic_features = np.random.randn(100, 10)
            self.anomaly_detector.fit(synthetic_features)
        
        print("✅ Models trained successfully\n")
    
    def analyze(self, data_path: str, dataset_name: str = None) -> dict:
        """
        Analyze a dataset for quality issues.
        
        Args:
            data_path: Path to CSV file
            dataset_name: Name for the dataset
            
        Returns:
            Analysis report
        """
        print(f"\n🔍 Analyzing {data_path}...")
        
        # Load data
        data = pd.read_csv(data_path)
        
        if dataset_name is None:
            dataset_name = Path(data_path).stem
        
        # Step 1: Profile the data
        print("  ├─ Profiling data...")
        current_profile = self.profiler.profile(data, dataset_name)
        
        # Step 2: Extract feature vector
        feature_vector = self.profiler.extract_feature_vector(current_profile)
        
        # Step 3: Detect anomalies (ML-based)
        print("  ├─ Running anomaly detection...")
        is_anomaly, anomaly_score, anomaly_details = self.anomaly_detector.detect(feature_vector)
        
        # Step 4: Detect specific quality issues
        print("  ├─ Identifying specific issues...")
        quality_issues = self.quality_detector.detect_all(
            current_profile, 
            self.profile_history
        )
        
        # Step 5: Classify issue type if anomaly detected
        issue_classification = None
        if is_anomaly:
            print("  ├─ Classifying issue type...")
            # Prepare features for classifier
            classifier_features = np.array([
                anomaly_score,
                current_profile['missing_patterns']['missing_rate'],
                0,  # missing_rate_change (would need historical data)
                0, 0, 0,  # mean, std, skewness changes
                1, 1,  # value ratios
                0, 0,  # schema changes
                current_profile['correlation_signature'].get('mean_correlation', 0) if current_profile['correlation_signature'] else 0
            ])
            
            issue_type, confidence, probabilities = self.issue_classifier.predict(classifier_features)
            issue_classification = {
                'issue_type': issue_type,
                'confidence': confidence,
                'probabilities': probabilities
            }
        
        # Step 6: Generate healing recommendations
        print("  └─ Generating recommendations...")
        recommendations = []
        
        if issue_classification and issue_classification['issue_type'] != 'normal':
            recommendations = self.healer.recommend(
                issue_classification['issue_type'],
                current_profile
            )
        
        for issue_type, issue_data in quality_issues.items():
            if issue_data['detected']:
                recs = self.healer.recommend(issue_type, current_profile)
                recommendations.extend(recs)
        
        # Build report
        report = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'profile': {
                'missing_rate': current_profile['missing_patterns']['missing_rate'],
                'n_numeric_cols': len(current_profile['numeric_features']),
                'n_categorical_cols': len(current_profile['categorical_features'])
            },
            'anomaly_detected': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'quality_issues': quality_issues,
            'issue_classification': issue_classification,
            'recommendations': recommendations,
            'data_health_score': self._calculate_health_score(current_profile, is_anomaly)
        }
        
        return report
    
    def heal(self, data_path: str, output_path: str = None, auto_heal: bool = False):
        """
        Heal data quality issues in a dataset.
        
        Args:
            data_path: Path to input CSV
            output_path: Path to save healed data
            auto_heal: If True, automatically apply best recommendations
        """
        print(f"\n🔧 Healing {data_path}...")
        
        # Analyze first
        report = self.analyze(data_path)
        
        # Load data
        data = pd.read_csv(data_path)
        healed_data = data.copy()
        healing_actions = []
        
        # Apply healing
        if report['recommendations']:
            print(f"\nFound {len(report['recommendations'])} healing recommendations:")
            
            for i, rec in enumerate(report['recommendations'][:3], 1):  # Top 3
                print(f"\n{i}. {rec['strategy']} (confidence: {rec['confidence']:.1%})")
                print(f"   {rec['description']}")
                
                if auto_heal:
                    # Determine issue type from recommendation
                    issue_type = self._infer_issue_from_strategy(rec['strategy'])
                    
                    print(f"   ⚙️  Applying...")
                    healed_data, heal_report = self.healer.heal(
                        healed_data,
                        issue_type,
                        strategy=rec['strategy'],
                        params=rec.get('params', {})
                    )
                    
                    healing_actions.append({
                        'strategy': rec['strategy'],
                        'report': heal_report
                    })
                    
                    if heal_report['success']:
                        print(f"   ✅ Success: {heal_report.get('message', 'Applied')}")
                    else:
                        print(f"   ❌ Failed: {heal_report.get('message', 'Error')}")
        else:
            print("\n✅ No data quality issues detected!")
        
        # Save healed data
        if output_path:
            healed_data.to_csv(output_path, index=False)
            print(f"\n💾 Healed data saved to {output_path}")
        
        # Save report
        report_path = Path(data_path).stem + '_report.json'
        with open(f'logs/{report_path}', 'w') as f:
            json.dump({
                'analysis': report,
                'healing_actions': healing_actions
            }, f, indent=2, default=str)
        
        print(f"📄 Report saved to logs/{report_path}")
        
        return healed_data, report
    
    def _calculate_health_score(self, profile: dict, is_anomaly: bool) -> float:
        """Calculate overall data health score (0-100)."""
        score = 100.0
        
        # Penalize for missing values
        missing_rate = profile['missing_patterns']['missing_rate']
        score -= missing_rate * 30
        
        # Penalize for anomaly detection
        if is_anomaly:
            score -= 20
        
        # Penalize for high missing correlation (systematic issues)
        missing_corr = profile['missing_patterns']['missing_correlation']
        score -= abs(missing_corr) * 10
        
        return max(0, min(100, score))
    
    def _infer_issue_from_strategy(self, strategy: str) -> str:
        """Infer issue type from healing strategy name."""
        strategy_map = {
            'impute_simple': 'missing_values',
            'impute_smart': 'missing_values',
            'drop_rows': 'missing_values',
            'winsorize': 'outliers',
            'clip_iqr': 'outliers',
            'retrain_alert': 'drift',
            'normalize': 'drift',
            'align_schema': 'schema_change',
            'drop_duplicates': 'duplicates'
        }
        return strategy_map.get(strategy, 'missing_values')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🚑 Automated Data Quality Healer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save healed data')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze, do not heal')
    parser.add_argument('--auto-heal', action='store_true',
                       help='Automatically apply healing recommendations')
    parser.add_argument('--train', action='store_true',
                       help='Train models before analysis')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = DataQualityHealer(config_path=args.config)
    
    # Train if requested
    if args.train:
        system.train_models()
    
    # Run analysis or healing
    if args.analyze_only:
        report = system.analyze(args.input)
        print(f"\n📊 Data Health Score: {report['data_health_score']:.1f}/100")
        
        if report['anomaly_detected']:
            print(f"⚠️  Anomaly detected (score: {report['anomaly_score']:.3f})")
        
        if report['quality_issues']:
            print(f"\n🔍 Quality Issues Found:")
            for issue_type, data in report['quality_issues'].items():
                if data['detected']:
                    print(f"  • {issue_type}: {data.get('severity', 'unknown')} severity")
    else:
        healed_data, report = system.heal(
            args.input,
            output_path=args.output,
            auto_heal=args.auto_heal
        )
        
        print(f"\n📊 Final Data Health Score: {report['data_health_score']:.1f}/100")


if __name__ == "__main__":
    main()