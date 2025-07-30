import subprocess
import sys
import os
import importlib
from pathlib import Path

# Auto-install packages function
def install_and_import(package_name, import_name=None):
    """
    Install and import packages automatically
    """
    if import_name is None:
        import_name = package_name
    
    try:
        # Try to import the package
        return importlib.import_module(import_name)
    except ImportError:
        print(f"Installing {package_name}...")
        try:
            # Install the package
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"Successfully installed {package_name}")
            return importlib.import_module(import_name)
        except subprocess.CalledProcessError:
            print(f"Failed to install {package_name}")
            return None
        except ImportError:
            print(f"Failed to import {package_name} after installation")
            return None

# Install required packages
print("Setting up environment for URSC Satellite Anomaly Detection System...")

# Core packages
streamlit = install_and_import("streamlit")
pandas = install_and_import("pandas", "pandas")
numpy = install_and_import("numpy", "numpy")
plotly = install_and_import("plotly")
sklearn = install_and_import("scikit-learn", "sklearn")

# Advanced ML packages
try:
    tensorflow = install_and_import("tensorflow")
except:
    tensorflow = None

try:
    keras = install_and_import("keras")
except:
    keras = None

# Visualization and reporting
altair = install_and_import("altair")
matplotlib = install_and_import("matplotlib")
seaborn = install_and_import("seaborn")
fpdf = install_and_import("fpdf2", "fpdf")

# Additional utilities
scipy = install_and_import("scipy")
joblib = install_and_import("joblib")

# Import specific modules after installation
if streamlit:
    import streamlit as st
if pandas:
    import pandas as pd
if numpy:
    import numpy as np
if plotly:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
if sklearn:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
if scipy:
    from scipy import stats
if joblib:
    import joblib
if altair:
    import altair as alt
if matplotlib:
    import matplotlib.pyplot as plt
if seaborn:
    import seaborn as sns

import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Advanced LSTM Autoencoder for Anomaly Detection
class AdvancedAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
    def create_lstm_autoencoder(self, input_dim, sequence_length=10):
        """Create LSTM Autoencoder if TensorFlow is available"""
        if tensorflow and keras:
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Encoder
            encoder_inputs = keras.Input(shape=(sequence_length, input_dim))
            encoder_l1 = layers.LSTM(64, return_sequences=True)(encoder_inputs)
            encoder_l2 = layers.LSTM(32, return_sequences=True)(encoder_l1)
            encoder_l3 = layers.LSTM(16)(encoder_l2)
            
            # Decoder
            decoder_l1 = layers.RepeatVector(sequence_length)(encoder_l3)
            decoder_l2 = layers.LSTM(16, return_sequences=True)(decoder_l1)
            decoder_l3 = layers.LSTM(32, return_sequences=True)(decoder_l2)
            decoder_l4 = layers.LSTM(64, return_sequences=True)(decoder_l3)
            decoder_outputs = layers.TimeDistributed(layers.Dense(input_dim))(decoder_l4)
            
            autoencoder = keras.Model(encoder_inputs, decoder_outputs)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            return autoencoder
        else:
            return None
    
    def prepare_sequences(self, data, sequence_length=10):
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:(i + sequence_length)])
        return np.array(sequences)
    
    def detect_anomalies(self, data, method='isolation_forest'):
        """Detect anomalies using various methods"""
        results = {}
        
        # Isolation Forest
        if method in ['isolation_forest', 'all']:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(data)
            results['isolation_forest'] = anomalies
        
        # Statistical methods
        if method in ['statistical', 'all']:
            z_scores = np.abs(stats.zscore(data, axis=0))
            statistical_anomalies = (z_scores > 3).any(axis=1).astype(int)
            statistical_anomalies = np.where(statistical_anomalies == 1, -1, 1)
            results['statistical'] = statistical_anomalies
        
        # LSTM Autoencoder (if available)
        if method in ['lstm', 'all'] and tensorflow:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                sequences = self.prepare_sequences(scaled_data)
                if len(sequences) > 20:  # Minimum data requirement
                    autoencoder = self.create_lstm_autoencoder(data.shape[1])
                    autoencoder.fit(sequences, sequences, epochs=50, verbose=0)
                    
                    predictions = autoencoder.predict(sequences)
                    mse = np.mean(np.power(sequences - predictions, 2), axis=(1, 2))
                    threshold = np.percentile(mse, 95)
                    
                    lstm_anomalies = np.ones(len(data))
                    lstm_anomalies[:len(mse)] = np.where(mse > threshold, -1, 1)
                    results['lstm'] = lstm_anomalies
            except Exception as e:
                st.warning(f"LSTM method failed: {e}")
        
        return results

# Satellite Telemetry Simulator
class SatelliteTelemetrySimulator:
    def __init__(self):
        self.base_params = {
            'temperature': {'normal': (20, 35), 'critical': (50, 80)},
            'voltage': {'normal': (3.2, 3.8), 'critical': (2.5, 4.5)},
            'current': {'normal': (0.5, 2.0), 'critical': (3.0, 5.0)},
            'pressure': {'normal': (1010, 1025), 'critical': (980, 1050)},
            'battery_level': {'normal': (70, 100), 'critical': (10, 30)},
            'solar_panel_voltage': {'normal': (12, 15), 'critical': (8, 20)},
            'gyroscope_x': {'normal': (-0.1, 0.1), 'critical': (-1, 1)},
            'gyroscope_y': {'normal': (-0.1, 0.1), 'critical': (-1, 1)},
            'gyroscope_z': {'normal': (-0.1, 0.1), 'critical': (-1, 1)},
            'magnetometer_x': {'normal': (-50, 50), 'critical': (-200, 200)},
            'magnetometer_y': {'normal': (-50, 50), 'critical': (-200, 200)},
            'magnetometer_z': {'normal': (-50, 50), 'critical': (-200, 200)}
        }
    
    def generate_telemetry(self, hours=24, anomaly_probability=0.05):
        """Generate realistic satellite telemetry data"""
        timestamps = pd.date_range(
            start=datetime.datetime.now() - datetime.timedelta(hours=hours),
            end=datetime.datetime.now(),
            freq='1min'
        )
        
        data = {'timestamp': timestamps}
        
        for param, ranges in self.base_params.items():
            normal_data = np.random.uniform(
                ranges['normal'][0], 
                ranges['normal'][1], 
                len(timestamps)
            )
            
            # Add some realistic variations
            if param == 'temperature':
                # Add orbital thermal cycling
                orbital_period = 90  # minutes
                thermal_cycle = 5 * np.sin(2 * np.pi * np.arange(len(timestamps)) / orbital_period)
                normal_data += thermal_cycle
            
            # Inject anomalies
            anomaly_indices = np.random.choice(
                len(timestamps), 
                int(len(timestamps) * anomaly_probability), 
                replace=False
            )
            
            for idx in anomaly_indices:
                if np.random.random() > 0.5:
                    normal_data[idx] = np.random.uniform(
                        ranges['critical'][0], 
                        ranges['critical'][1]
                    )
            
            data[param] = normal_data
        
        return pd.DataFrame(data)

# Advanced Visualization Functions
def create_advanced_dashboard(df, anomalies=None):
    """Create comprehensive dashboard"""
    
    # Main telemetry parameters
    critical_params = ['temperature', 'voltage', 'current', 'pressure']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=critical_params,
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": True}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, param in enumerate(critical_params):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Normal data
        normal_mask = True
        if anomalies is not None and 'isolation_forest' in anomalies:
            normal_mask = anomalies['isolation_forest'] == 1
        
        # Plot normal data
        fig.add_trace(
            go.Scatter(
                x=df[normal_mask]['timestamp'],
                y=df[normal_mask][param],
                mode='lines',
                name=f'{param.title()} (Normal)',
                line=dict(color=colors[i], width=2),
                opacity=0.8
            ),
            row=row, col=col
        )
        
        # Plot anomalies
        if anomalies is not None and 'isolation_forest' in anomalies:
            anomaly_mask = anomalies['isolation_forest'] == -1
            if anomaly_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=df[anomaly_mask]['timestamp'],
                        y=df[anomaly_mask][param],
                        mode='markers',
                        name=f'{param.title()} (Anomaly)',
                        marker=dict(color='red', size=8, symbol='x'),
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        title="URSC Satellite Health Monitoring Dashboard",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def create_system_health_summary(df, anomalies):
    """Create system health summary"""
    
    if anomalies and 'isolation_forest' in anomalies:
        total_points = len(df)
        anomaly_count = sum(anomalies['isolation_forest'] == -1)
        health_score = ((total_points - anomaly_count) / total_points) * 100
        
        # Determine status
        if health_score >= 95:
            status = "EXCELLENT"
            color = "green"
        elif health_score >= 85:
            status = "GOOD" 
            color = "blue"
        elif health_score >= 70:
            status = "WARNING"
            color = "orange"
        else:
            status = "CRITICAL"
            color = "red"
        
        return {
            'health_score': health_score,
            'status': status,
            'color': color,
            'anomaly_count': anomaly_count,
            'total_points': total_points
        }
    
    return None

# PDF Report Generator
def generate_pdf_report(df, anomalies, health_summary):
    """Generate PDF report"""
    if not fpdf:
        return None
    
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'URSC Satellite Telemetry Anomaly Report', 0, 1, 'C')
        pdf.ln(10)
        
        # Executive Summary
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        if health_summary:
            pdf.cell(0, 8, f"System Health Score: {health_summary['health_score']:.1f}%", 0, 1)
            pdf.cell(0, 8, f"Status: {health_summary['status']}", 0, 1)
            pdf.cell(0, 8, f"Anomalies Detected: {health_summary['anomaly_count']}/{health_summary['total_points']}", 0, 1)
        
        pdf.ln(10)
        
        # Detailed Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detailed Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        critical_params = ['temperature', 'voltage', 'current', 'pressure']
        for param in critical_params:
            if param in df.columns:
                mean_val = df[param].mean()
                max_val = df[param].max()
                min_val = df[param].min()
                pdf.cell(0, 6, f"{param.title()}: Mean={mean_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f}", 0, 1)
        
        # Save to bytes
        return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Main Streamlit Application
def main():
    st.set_page_config(
        page_title="URSC Satellite Anomaly Detection",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .alert-card {
        background: #fee;
        border: 1px solid #fcc;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ URSC Satellite Health Monitoring System</h1>
        <p>AI-Powered Anomaly Detection for Satellite Telemetry</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Simulated Data", "Upload CSV"]
    )
    
    # Anomaly detection method
    detection_method = st.sidebar.selectbox(
        "Detection Method",
        ["Isolation Forest", "Statistical", "LSTM Autoencoder", "All Methods"]
    )
    
    # Initialize components
    detector = AdvancedAnomalyDetector()
    simulator = SatelliteTelemetrySimulator()
    
    # Data loading
    df = None
    
    if data_source == "Simulated Data":
        hours = st.sidebar.slider("Simulation Hours", 1, 72, 24)
        anomaly_prob = st.sidebar.slider("Anomaly Probability", 0.01, 0.2, 0.05)
        
        if st.sidebar.button("Generate Data"):
            with st.spinner("Generating satellite telemetry data..."):
                df = simulator.generate_telemetry(hours, anomaly_prob)
                st.session_state['telemetry_data'] = df
        
        if 'telemetry_data' in st.session_state:
            df = st.session_state['telemetry_data']
    
    else:  # Upload CSV
        uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Main content
    if df is not None:
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Parameters", len(df.columns) - 1)
        with col3:
            st.metric("Time Span", f"{(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f}h")
        with col4:
            st.metric("Data Quality", "100%")
        
        # Anomaly detection
        st.subheader("🔍 Anomaly Detection Results")
        
        # Prepare data for ML
        feature_columns = [col for col in df.columns if col != 'timestamp']
        X = df[feature_columns].fillna(df[feature_columns].mean())
        
        # Detect anomalies
        method_map = {
            "Isolation Forest": "isolation_forest",
            "Statistical": "statistical", 
            "LSTM Autoencoder": "lstm",
            "All Methods": "all"
        }
        
        with st.spinner("Detecting anomalies..."):
            anomalies = detector.detect_anomalies(X, method_map[detection_method])
        
        # Display results
        if anomalies:
            # Health summary
            health_summary = create_system_health_summary(df, anomalies)
            
            if health_summary:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: {health_summary['color']}">System Status</h3>
                        <h2>{health_summary['status']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Health Score</h3>
                        <h2>{health_summary['health_score']:.1f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Anomalies</h3>
                        <h2>{health_summary['anomaly_count']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📊 Telemetry Dashboard")
        
        if anomalies:
            fig = create_advanced_dashboard(df, anomalies)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        if st.expander("📋 Detailed Parameter Analysis"):
            for method, results in anomalies.items():
                st.write(f"**{method.replace('_', ' ').title()} Results:**")
                anomaly_count = sum(results == -1)
                normal_count = sum(results == 1)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Normal Points", normal_count)
                with col2:
                    st.metric("Anomalous Points", anomaly_count)
        
        # Download section
        st.subheader("📥 Download Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download CSV Report"):
                # Add anomaly columns to dataframe
                result_df = df.copy()
                for method, results in anomalies.items():
                    result_df[f'anomaly_{method}'] = results
                
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"satellite_anomaly_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Generate PDF Report"):
                health_summary = create_system_health_summary(df, anomalies)
                pdf_data = generate_pdf_report(df, anomalies, health_summary)
                
                if pdf_data:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"ursc_satellite_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
    
    else:
        st.info("👆 Please select a data source and generate/upload telemetry data to begin analysis.")
        
        # Show example data structure
        st.subheader("Expected Data Format")
        example_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [25.1, 26.3, 45.2, 24.8, 25.9],
            'voltage': [3.3, 3.4, 2.1, 3.5, 3.2],
            'current': [1.2, 1.1, 4.5, 1.0, 1.3],
            'pressure': [1013, 1015, 995, 1012, 1014]
        })
        st.dataframe(example_df)

if __name__ == "__main__":
    main()
