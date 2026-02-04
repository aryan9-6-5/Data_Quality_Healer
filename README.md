#  Automated Data Quality Healer (ML-Driven)

##  Problem Statement

Modern ML and analytics systems fail not because of bad models, but because of **bad data**:
- Missing values
- Sudden distribution shifts  
- Schema changes
- Silent outliers

Most systems only **detect** issues — they don't **fix** them.

This project builds an intelligent ML-based system that:
-  Detects data quality issues automatically
-  Classifies the issue type
-  Suggests or applies corrective actions
-  Learns from past fixes

---

##  Project Objectives

1. **Detect** anomalies in incoming datasets
2. **Identify** type of data issue
3. **Recommend** healing strategies
4. **Track** data health over time
5. **Provide** explainable alerts

---

##  System Architecture

```
Incoming Data
     ↓
Data Profiler (Statistical Features)
     ↓
Anomaly Detector (Isolation Forest, One-Class SVM)
     ↓
Issue Classifier (Random Forest, Gradient Boosting)
     ↓
Healing Recommendation Engine
     ↓
Cleaned Data + Audit Logs
```

---

##  ML Components

### 1️ Data Profiler
Extracts statistical features:
- Missing value %
- Mean, std, skewness
- Unique counts
- Schema signature

###  Anomaly Detector (Unsupervised)
**Models**: Isolation Forest, One-Class SVM

**Detects**:
- Sudden spikes
- Value explosions
- Null floods
- Distribution drift

###  Issue Classifier (Supervised)
Classifies detected anomalies into:
- Missing Data
- Outliers
- Distribution Drift
- Schema Changes
- Duplicates

**Models**: Random Forest, Gradient Boosting

###  Healing Recommendation Engine
Maps issues to fixes:

| Issue Type | Suggested Fix |
|------------|---------------|
| Missing values | Mean/Median/KNN Imputation |
| Outliers | Winsorization/Capping |
| Drift | Retrain Alert |
| Schema Change | Version Rollback |
| Duplicates | Deduplication |

###  Feedback Loop
- User accepts/rejects fixes
- System updates confidence scores
- Learning from production

---

##  Evaluation Metrics

- **Anomaly Detection Precision**: How many detected anomalies are real?
- **False Positive Rate**: How often do we flag good data?
- **Time to Detection**: How quickly do we catch issues?
- **Data Recovery Success %**: How often does healing work?
- **User Acceptance Rate**: How often are fixes approved?

---

##  Tech Stack

- **Python 3.9+**
- **Pandas** / **NumPy**: Data manipulation
- **Scikit-learn**: ML models
- **Streamlit**: Interactive dashboard
- **SQLite**: Audit logs
- **Matplotlib** / **Seaborn**: Visualization

---

##  Project Structure

```
data-healer/
│
├── profiler/               # Statistical profiling
│   └── data_profiler.py
│
├── anomaly_detector/       # Unsupervised anomaly detection
│   └── detector.py
│
├── issue_classifier/       # Supervised issue classification
│   └── classifier.py
│
├── healing_engine/         # Healing recommendations
│   └── healer.py
│
├── dashboard/              # Streamlit UI
│   └── app.py
│
├── data/                   # Sample datasets
│   ├── synthetic_generator.py
│   └── samples/
│
├── models/                 # Trained models
│
├── logs/                   # Audit trails
│
├── tests/                  # Unit tests
│
├── requirements.txt
├── config.yaml
├── main.py                 # CLI entry point
└── README.md
```

---

##  Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd data-healer

# Install dependencies
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
python data/synthetic_generator.py
```

### Run the System

```bash
# CLI mode
python main.py --input data/samples/transactions.csv

# Dashboard mode
streamlit run dashboard/app.py
```

---

##  Usage Examples

### Example 1: Detect Anomalies

```python
from profiler.data_profiler import DataProfiler
from anomaly_detector.detector import AnomalyDetector

# Profile data
profiler = DataProfiler()
features = profiler.profile('data.csv')

# Detect anomalies
detector = AnomalyDetector()
anomalies = detector.detect(features)
```

### Example 2: Classify Issues

```python
from issue_classifier.classifier import IssueClassifier

classifier = IssueClassifier()
classifier.train(historical_data)
issue_type = classifier.predict(anomaly_features)
```

### Example 3: Heal Data

```python
from healing_engine.healer import DataHealer

healer = DataHealer()
cleaned_data = healer.heal(data, issue_type='missing_values')
```

---

##  How to Explain in Interviews

> *"Instead of building another prediction model, I focused on **data reliability**, which is a real bottleneck in production ML systems. My system not only detects data issues but also recommends corrective actions and **learns from past fixes**."*

Key talking points:
-  **Enterprise mindset**: Solves real production problems
-  **ML beyond classification**: Unsupervised + supervised learning
-  **Data engineering + ML**: End-to-end pipeline
-  **Not copied from Kaggle**: Original architecture
-  **Easy to extend**: Modular design

---

##  Learning Outcomes

1. **Unsupervised Learning**: Isolation Forest, One-Class SVM
2. **Supervised Classification**: Random Forest, Gradient Boosting
3. **Feature Engineering**: Statistical profiling
4. **Production ML**: Feedback loops, monitoring
5. **Data Quality**: Real-world MLOps challenges

---

##  Future Enhancements

- [ ] Real-time streaming data support
- [ ] Deep learning for complex drift detection
- [ ] Auto-healing with confidence thresholds
- [ ] Integration with data catalogs (dbt, Great Expectations)
- [ ] Multi-dataset correlation analysis

---


## Acknowledgments

This project addresses real production ML challenges and demonstrates enterprise-level thinking beyond typical Kaggle competitions.
