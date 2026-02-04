"""
Complete Example: Data Quality Healer Usage
This demonstrates the full workflow of the system
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from profiler.data_profiler import DataProfiler
from anomaly_detector.detector import AnomalyDetector, DataQualityAnomalyDetector
from issue_classifier.classifier import IssueClassifier, generate_synthetic_training_data
from healing_engine.healer import DataHealer
from data.synthetic_generator import SyntheticDataGenerator


def example_1_basic_workflow():
    """Example 1: Basic workflow with synthetic data."""
    
    print("=" * 70)
    print("EXAMPLE 1: Basic Data Quality Workflow")
    print("=" * 70)
    
    # Step 1: Generate synthetic data with issues
    print("\n📊 Step 1: Generating synthetic data...")
    generator = SyntheticDataGenerator(seed=42)
    data = generator.generate_transaction_data(n_records=1000, inject_issues=True)
    
    print(f"Generated {len(data)} records with {len(data.columns)} columns")
    print(f"\nData preview:")
    print(data.head())
    
    # Step 2: Profile the data
    print("\n🔍 Step 2: Profiling data...")
    profiler = DataProfiler()
    profile = profiler.profile(data, "synthetic_transactions")
    
    print(f"Missing values: {profile['missing_patterns']['total_missing']}")
    print(f"Missing rate: {profile['missing_patterns']['missing_rate']:.2%}")
    print(f"Numeric features: {len(profile['numeric_features'])}")
    
    # Step 3: Train anomaly detector
    print("\n🧠 Step 3: Training anomaly detector...")
    detector = AnomalyDetector(contamination=0.1)
    
    # Generate baseline clean data for training
    clean_data = generator.generate_transaction_data(n_records=500, inject_issues=False)
    clean_profiles = []
    for _ in range(100):
        sample = clean_data.sample(100)
        p = profiler.profile(sample, "clean_sample")
        fv = profiler.extract_feature_vector(p)
        clean_profiles.append(fv)
    
    detector.fit(np.array(clean_profiles))
    
    # Step 4: Detect anomalies
    print("\n⚠️  Step 4: Detecting anomalies...")
    feature_vector = profiler.extract_feature_vector(profile)
    is_anomaly, anomaly_score, details = detector.detect(feature_vector)
    
    print(f"Anomaly detected: {is_anomaly}")
    print(f"Anomaly score: {anomaly_score:.3f}")
    
    # Step 5: Train issue classifier
    print("\n🎯 Step 5: Training issue classifier...")
    classifier = IssueClassifier()
    X_train, y_train = generate_synthetic_training_data(n_samples=1000)
    accuracy = classifier.train(X_train, y_train)
    
    # Step 6: Classify the issue
    if is_anomaly:
        print("\n🔬 Step 6: Classifying issue type...")
        classifier_features = np.array([
            anomaly_score,
            profile['missing_patterns']['missing_rate'],
            0, 0, 0, 0, 1, 1, 0, 0,
            profile['correlation_signature'].get('mean_correlation', 0) if profile['correlation_signature'] else 0
        ])
        
        issue_type, confidence, probabilities = classifier.predict(classifier_features)
        
        print(f"Predicted issue: {issue_type}")
        print(f"Confidence: {confidence:.1%}")
        print(f"\nTop 3 predictions:")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        for issue, prob in sorted_probs:
            print(f"  {issue}: {prob:.1%}")
    
    # Step 7: Get healing recommendations
    print("\n💊 Step 7: Getting healing recommendations...")
    healer = DataHealer()
    recommendations = healer.recommend('missing_values', profile)
    
    print(f"\nFound {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['strategy']}")
        print(f"   Confidence: {rec['confidence']:.1%}")
        print(f"   Description: {rec['description']}")
    
    # Step 8: Apply healing
    print("\n🔧 Step 8: Applying healing...")
    healed_data, report = healer.heal(data, 'missing_values', strategy='impute_simple')
    
    print(f"\nHealing Report:")
    print(f"  Success: {report['success']}")
    print(f"  Missing before: {report.get('missing_before', 'N/A')}")
    print(f"  Missing after: {report.get('missing_after', 'N/A')}")
    
    print("\n✅ Example 1 completed!")
    return healed_data


def example_2_specific_issues():
    """Example 2: Detecting and healing specific issue types."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Detecting Specific Quality Issues")
    print("=" * 70)
    
    # Create data with known issues
    print("\n📊 Creating data with specific issues...")
    
    # Missing values issue
    data_missing = pd.DataFrame({
        'value1': [1, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10],
        'value2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })
    
    print(f"\nMissing values: {data_missing.isna().sum().sum()}")
    
    # Outliers issue
    data_outliers = pd.DataFrame({
        'normal_values': [10, 12, 11, 13, 12, 11, 10, 12, 11, 13],
        'with_outliers': [10, 12, 11, 1000, 12, 11, 10, 12, 2000, 13]
    })
    
    print(f"Outlier range: {data_outliers['with_outliers'].min()} - {data_outliers['with_outliers'].max()}")
    
    # Duplicates issue
    data_dupes = pd.DataFrame({
        'id': [1, 2, 3, 3, 4, 5, 5, 6],  # Duplicates
        'value': [100, 200, 300, 300, 400, 500, 500, 600]
    })
    
    print(f"Total rows: {len(data_dupes)}, Unique rows: {data_dupes.drop_duplicates().shape[0]}")
    
    # Heal each issue
    healer = DataHealer()
    
    print("\n🔧 Healing missing values...")
    healed_missing, report1 = healer.heal(data_missing, 'missing_values', strategy='impute_simple')
    print(f"  Result: {report1['success']} - {report1.get('missing_after', 0)} missing values remain")
    
    print("\n🔧 Healing outliers...")
    healed_outliers, report2 = healer.heal(data_outliers, 'outliers', strategy='winsorize')
    print(f"  Result: {report2['success']} - {report2.get('outliers_capped', 0)} outliers capped")
    print(f"  New range: {healed_outliers['with_outliers'].min():.0f} - {healed_outliers['with_outliers'].max():.0f}")
    
    print("\n🔧 Healing duplicates...")
    healed_dupes, report3 = healer.heal(data_dupes, 'duplicates', strategy='drop_duplicates')
    print(f"  Result: {report3['success']} - {report3.get('duplicates_removed', 0)} duplicates removed")
    print(f"  Final rows: {len(healed_dupes)}")
    
    print("\n✅ Example 2 completed!")


def example_3_quality_monitoring():
    """Example 3: Continuous quality monitoring."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Continuous Quality Monitoring")
    print("=" * 70)
    
    print("\n📊 Simulating data quality over time...")
    
    profiler = DataProfiler()
    quality_detector = DataQualityAnomalyDetector()
    generator = SyntheticDataGenerator()
    
    # Generate baseline data
    baseline_data = generator.generate_transaction_data(n_records=1000, inject_issues=False)
    baseline_profile = profiler.profile(baseline_data, "baseline")
    
    print(f"✅ Baseline established:")
    print(f"   Missing rate: {baseline_profile['missing_patterns']['missing_rate']:.2%}")
    
    # Simulate degrading data quality
    print("\n⏰ Simulating 5 time periods...")
    
    profiles = [baseline_profile]
    
    for period in range(1, 6):
        # Progressively inject more issues
        data = generator.generate_transaction_data(n_records=1000, inject_issues=True)
        
        # Add more corruption each period
        corruption_rate = 0.05 * period
        mask = np.random.random(len(data)) < corruption_rate
        data.loc[mask, 'total_amount'] = np.nan
        
        profile = profiler.profile(data, f"period_{period}")
        profiles.append(profile)
        
        # Detect issues
        issues = quality_detector.detect_all(profile, profiles[:-1])
        
        print(f"\n📅 Period {period}:")
        print(f"   Missing rate: {profile['missing_patterns']['missing_rate']:.2%}")
        
        detected_issues = [k for k, v in issues.items() if v['detected']]
        if detected_issues:
            print(f"   ⚠️  Issues: {', '.join(detected_issues)}")
        else:
            print(f"   ✅ No issues")
    
    print("\n✅ Example 3 completed!")


def example_4_learning_from_feedback():
    """Example 4: Learning from user feedback."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Learning from Feedback")
    print("=" * 70)
    
    healer = DataHealer()
    
    print("\n📊 Initial healing statistics:")
    print(healer.get_healing_stats())
    
    # Simulate healing attempts with feedback
    print("\n🔄 Simulating healing attempts...")
    
    for i in range(10):
        issue_type = np.random.choice(['missing_values', 'outliers', 'duplicates'])
        success = np.random.random() > 0.3  # 70% success rate
        
        healer.record_feedback(issue_type, success)
    
    print("\n📊 Updated healing statistics:")
    stats = healer.get_healing_stats()
    print(stats)
    
    print("\n💡 Success rates:")
    for _, row in stats.iterrows():
        if row['attempts'] > 0:
            print(f"  {row['issue_type']}: {row['success_rate']:.1%}")
    
    print("\n✅ Example 4 completed!")


def main():
    """Run all examples."""
    print("\n" + "🚑" * 20)
    print("DATA QUALITY HEALER - COMPLETE EXAMPLES")
    print("🚑" * 20)
    
    # Run examples
    healed_data = example_1_basic_workflow()
    example_2_specific_issues()
    example_3_quality_monitoring()
    example_4_learning_from_feedback()
    
    print("\n" + "=" * 70)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run the dashboard: streamlit run dashboard/app.py")
    print("  2. Try the CLI: python main.py --input data/samples/transactions.csv")
    print("  3. Generate your own data: python data/synthetic_generator.py")
    print("  4. Run tests: python tests/test_suite.py")


if __name__ == "__main__":
    main()