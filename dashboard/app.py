"""
Streamlit Dashboard for Data Quality Healer
Interactive UI for monitoring and healing data quality issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from profiler.data_profiler import DataProfiler
from anomaly_detector.detector import AnomalyDetector, DataQualityAnomalyDetector
from issue_classifier.classifier import IssueClassifier, generate_synthetic_training_data
from healing_engine.healer import DataHealer


# Page configuration
st.set_page_config(
    page_title="Data Quality Healer",
    page_icon="🚑",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
.issue-badge {
    padding: 5px 10px;
    border-radius: 5px;
    font-weight: bold;
}
.severity-high {
    background-color: #ff4b4b;
    color: white;
}
.severity-medium {
    background-color: #ffa500;
    color: white;
}
.severity-low {
    background-color: #90ee90;
    color: black;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
    st.session_state.profiler = DataProfiler()
    st.session_state.healer = DataHealer()
    st.session_state.quality_detector = DataQualityAnomalyDetector()
    st.session_state.reports = []


def initialize_system():
    """Initialize ML components."""
    with st.spinner("🔧 Initializing ML models..."):
        # Train classifier
        X_train, y_train = generate_synthetic_training_data(n_samples=800)
        st.session_state.classifier = IssueClassifier(model_type='random_forest')
        st.session_state.classifier.train(X_train, y_train)
        
        # Train anomaly detector
        synthetic_features = np.random.randn(100, 10)
        st.session_state.anomaly_detector = AnomalyDetector(
            contamination=0.1,
            method='isolation_forest'
        )
        st.session_state.anomaly_detector.fit(synthetic_features)
        
        st.session_state.system_initialized = True
        st.success("✅ System initialized!")


def analyze_data(data: pd.DataFrame, dataset_name: str):
    """Analyze uploaded data."""
    # Profile
    profile = st.session_state.profiler.profile(data, dataset_name)
    
    # Extract features
    feature_vector = st.session_state.profiler.extract_feature_vector(profile)
    
    # Detect anomalies
    is_anomaly, anomaly_score, anomaly_details = st.session_state.anomaly_detector.detect(feature_vector)
    
    # Detect specific issues
    quality_issues = st.session_state.quality_detector.detect_all(profile, [])
    
    # Classify if anomaly
    issue_classification = None
    if is_anomaly:
        classifier_features = np.array([
            anomaly_score,
            profile['missing_patterns']['missing_rate'],
            0, 0, 0, 0, 1, 1, 0, 0,
            profile['correlation_signature'].get('mean_correlation', 0) if profile['correlation_signature'] else 0
        ])
        
        issue_type, confidence, probabilities = st.session_state.classifier.predict(classifier_features)
        issue_classification = {
            'issue_type': issue_type,
            'confidence': confidence,
            'probabilities': probabilities
        }
    
    # Generate recommendations
    recommendations = []
    if issue_classification and issue_classification['issue_type'] != 'normal':
        recommendations = st.session_state.healer.recommend(
            issue_classification['issue_type'],
            profile
        )
    
    for issue_type, issue_data in quality_issues.items():
        if issue_data['detected']:
            recs = st.session_state.healer.recommend(issue_type, profile)
            recommendations.extend(recs)
    
    # Calculate health score
    health_score = calculate_health_score(profile, is_anomaly)
    
    return {
        'profile': profile,
        'is_anomaly': is_anomaly,
        'anomaly_score': anomaly_score,
        'quality_issues': quality_issues,
        'issue_classification': issue_classification,
        'recommendations': recommendations,
        'health_score': health_score
    }


def calculate_health_score(profile: dict, is_anomaly: bool) -> float:
    """Calculate data health score."""
    score = 100.0
    missing_rate = profile['missing_patterns']['missing_rate']
    score -= missing_rate * 30
    if is_anomaly:
        score -= 20
    return max(0, min(100, score))


def plot_health_gauge(score: float):
    """Create health score gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Health Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightcoral"},
                {'range': [50, 75], 'color': "lightyellow"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def plot_missing_values(profile: dict):
    """Plot missing values by column."""
    missing_data = profile['missing_patterns']['columns_with_missing']
    
    if not missing_data:
        return None
    
    df = pd.DataFrame({
        'Column': list(missing_data.keys()),
        'Missing Count': list(missing_data.values())
    })
    
    fig = px.bar(df, x='Column', y='Missing Count',
                 title='Missing Values by Column',
                 color='Missing Count',
                 color_continuous_scale='Reds')
    
    return fig


def plot_numeric_distributions(data: pd.DataFrame):
    """Plot distributions of numeric columns."""
    numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]  # First 4
    
    if len(numeric_cols) == 0:
        return None
    
    fig = go.Figure()
    
    for col in numeric_cols:
        fig.add_trace(go.Box(y=data[col], name=col))
    
    fig.update_layout(
        title='Numeric Columns Distribution',
        yaxis_title='Value',
        height=400
    )
    
    return fig


# Main App
def main():
    # Header
    st.markdown('<p class="main-header">🚑 Automated Data Quality Healer</p>', unsafe_allow_html=True)
    st.markdown("*ML-powered data quality detection and healing*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        if not st.session_state.system_initialized:
            if st.button("🚀 Initialize System"):
                initialize_system()
        else:
            st.success("✅ System Ready")
        
        st.markdown("---")
        
        st.header("📊 Upload Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            dataset_name = uploaded_file.name.replace('.csv', '')
            
            if st.button("🔍 Analyze"):
                if st.session_state.system_initialized:
                    st.session_state.current_data = pd.read_csv(uploaded_file)
                    st.session_state.current_analysis = analyze_data(
                        st.session_state.current_data,
                        dataset_name
                    )
                    st.session_state.reports.append({
                        'name': dataset_name,
                        'timestamp': datetime.now(),
                        'health_score': st.session_state.current_analysis['health_score']
                    })
                else:
                    st.error("Please initialize system first!")
    
    # Main content
    if 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        data = st.session_state.current_data
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            missing_pct = analysis['profile']['missing_patterns']['missing_rate'] * 100
            st.metric("Missing %", f"{missing_pct:.1f}%")
        with col4:
            st.metric("Health Score", f"{analysis['health_score']:.0f}/100")
        
        # Health gauge
        st.plotly_chart(plot_health_gauge(analysis['health_score']), use_container_width=True)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "🔍 Issues", "💊 Recommendations", "📊 Visualizations"])
        
        with tab1:
            st.header("Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                st.write(f"**Shape:** {data.shape}")
                st.write(f"**Numeric Columns:** {len(analysis['profile'].get('numeric_features', {}))}")
                st.write(f"**Categorical Columns:** {len(analysis['profile'].get('categorical_features', {}))}")
                st.write(f"**Missing Rate:** {missing_pct:.2f}%")
            
            with col2:
                st.subheader("Anomaly Detection")
                if analysis['is_anomaly']:
                    st.error(f"⚠️ Anomaly Detected!")
                    st.write(f"**Score:** {analysis['anomaly_score']:.3f}")
                else:
                    st.success("✅ No anomalies detected")
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(10), use_container_width=True)
        
        with tab2:
            st.header("Detected Issues")
            
            if analysis['quality_issues']:
                for issue_type, issue_data in analysis['quality_issues'].items():
                    if issue_data['detected']:
                        severity = issue_data.get('severity', 'unknown')
                        
                        with st.expander(f"❗ {issue_type.replace('_', ' ').title()} - {severity.upper()}", expanded=True):
                            st.json(issue_data)
            else:
                st.success("✅ No specific quality issues detected!")
            
            if analysis['issue_classification']:
                st.subheader("ML Classification")
                
                class_info = analysis['issue_classification']
                st.write(f"**Predicted Issue:** {class_info['issue_type']}")
                st.write(f"**Confidence:** {class_info['confidence']:.1%}")
                
                # Probability chart
                probs_df = pd.DataFrame({
                    'Issue Type': list(class_info['probabilities'].keys()),
                    'Probability': list(class_info['probabilities'].values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(probs_df, x='Issue Type', y='Probability',
                           title='Issue Classification Probabilities')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.header("Healing Recommendations")
            
            if analysis['recommendations']:
                for i, rec in enumerate(analysis['recommendations'], 1):
                    with st.expander(f"💊 Recommendation {i}: {rec['strategy']} (Confidence: {rec['confidence']:.0%})", 
                                   expanded=i==1):
                        st.write(f"**Strategy:** {rec['strategy']}")
                        st.write(f"**Description:** {rec['description']}")
                        st.write(f"**Confidence:** {rec['confidence']:.1%}")
                        
                        if st.button(f"Apply {rec['strategy']}", key=f"apply_{i}"):
                            with st.spinner("Applying healing strategy..."):
                                # Determine issue type
                                issue_type = 'missing_values'  # Default
                                if 'outlier' in rec['strategy'].lower():
                                    issue_type = 'outliers'
                                elif 'duplicate' in rec['strategy'].lower():
                                    issue_type = 'duplicates'
                                
                                healed_data, report = st.session_state.healer.heal(
                                    data,
                                    issue_type,
                                    strategy=rec['strategy'],
                                    params=rec.get('params', {})
                                )
                                
                                if report['success']:
                                    st.success(f"✅ {report.get('message', 'Healing applied!')}")
                                    st.session_state.current_data = healed_data
                                    
                                    # Re-analyze
                                    st.session_state.current_analysis = analyze_data(
                                        healed_data,
                                        "healed_data"
                                    )
                                    st.experimental_rerun()
                                else:
                                    st.error(f"❌ {report.get('message', 'Healing failed')}")
            else:
                st.success("✅ No recommendations - data quality looks good!")
        
        with tab4:
            st.header("Data Visualizations")
            
            # Missing values plot
            missing_fig = plot_missing_values(analysis['profile'])
            if missing_fig:
                st.plotly_chart(missing_fig, use_container_width=True)
            
            # Distribution plots
            dist_fig = plot_numeric_distributions(data)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
        
        # Download healed data
        if st.button("💾 Download Healed Data"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="healed_data.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.info("👈 Upload a CSV file from the sidebar to get started!")
        
        st.markdown("""
        ## How It Works
        
        1. **Upload** your CSV file
        2. **Initialize** the ML system
        3. **Analyze** to detect data quality issues
        4. **Review** detected issues and classifications
        5. **Apply** healing recommendations
        6. **Download** the cleaned data
        
        ## Features
        
        - ✅ **Anomaly Detection** using Isolation Forest
        - ✅ **Issue Classification** with Random Forest
        - ✅ **Smart Healing** recommendations
        - ✅ **Interactive** visualizations
        - ✅ **Automatic** or manual healing
        """)


if __name__ == "__main__":
    main()