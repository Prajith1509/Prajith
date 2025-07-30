import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Satellite Health Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-danger {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SatelliteDataSimulator:
    """Simulates realistic satellite telemetry data"""
    
    def __init__(self):
        self.base_params = {
            'temperature': {'normal': (20, 25), 'range': (15, 30), 'critical': (35, 40)},
            'voltage': {'normal': (11.8, 12.2), 'range': (10.5, 13.0), 'critical': (9.5, 14.0)},
            'current': {'normal': (2.8, 3.2), 'range': (2.0, 4.0), 'critical': (1.0, 5.0)},
            'pressure': {'normal': (1010, 1020), 'range': (995, 1035), 'critical': (980, 1050)}
        }
    
    def generate_data(self, hours=24, freq_minutes=5, anomaly_rate=0.05):
        """Generate satellite telemetry data with controlled anomalies"""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=hours),
            end=datetime.now(),
            freq=f'{freq_minutes}min'
        )
        
        n_points = len(timestamps)
        data = {'timestamp': timestamps}
        
        # Generate normal data with orbital variations
        for param, config in self.base_params.items():
            normal_mean = np.mean(config['normal'])
            normal_std = (config['normal'][1] - config['normal'][0]) / 4
            
            # Simulate orbital effects (temperature variations, eclipse periods)
            orbital_period = n_points // 16  # ~16 orbits per day
            orbital_variation = 0.3 * normal_mean * np.sin(2 * np.pi * np.arange(n_points) / orbital_period)
            
            # Base signal with orbital variation
            base_signal = normal_mean + orbital_variation
            
            # Add random noise
            noise = np.random.normal(0, normal_std * 0.5, n_points)
            values = base_signal + noise
            
            # Inject realistic anomalies
            n_anomalies = int(n_points * anomaly_rate)
            anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
            
            for idx in anomaly_indices:
                anomaly_type = np.random.choice(['thermal_spike', 'power_drop', 'sensor_drift', 'eclipse_anomaly'])
                
                if anomaly_type == 'thermal_spike':
                    # Sudden temperature increase (solar panel heating)
                    if param == 'temperature':
                        values[idx:idx+3] += np.random.uniform(8, 15)
                elif anomaly_type == 'power_drop':
                    # Battery/power system issues
                    if param in ['voltage', 'current']:
                        drop_duration = min(5, n_points - idx)
                        values[idx:idx+drop_duration] *= np.random.uniform(0.6, 0.8)
                elif anomaly_type == 'sensor_drift':
                    # Gradual sensor degradation
                    drift_duration = min(20, n_points - idx)
                    drift_factor = np.linspace(1, np.random.uniform(1.2, 1.5), drift_duration)
                    values[idx:idx+drift_duration] *= drift_factor
                elif anomaly_type == 'eclipse_anomaly':
                    # Eclipse period effects
                    if param == 'temperature':
                        values[idx:idx+4] -= np.random.uniform(5, 10)
            
            # Clip values to realistic ranges
            values = np.clip(values, config['critical'][0], config['critical'][1])
            data[param] = values
        
        # Add system status and operational data
        data['satellite_mode'] = np.random.choice(
            ['NOMINAL', 'SAFE_MODE', 'ECLIPSE', 'MAINTENANCE'], 
            n_points, 
            p=[0.75, 0.10, 0.10, 0.05]
        )
        data['data_quality'] = np.random.uniform(0.88, 1.0, n_points)
        data['signal_strength'] = np.random.uniform(0.7, 1.0, n_points)
        
        return pd.DataFrame(data)

class SimpleAnomalyDetector:
    """Lightweight anomaly detection using only sklearn"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = ['temperature', 'voltage', 'current', 'pressure']
        self.thresholds = {}
    
    def train_isolation_forest(self, data):
        """Train Isolation Forest model"""
        X = data[self.feature_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = IsolationForest(
            contamination=0.1, 
            random_state=42,
            n_estimators=100
        )
        model.fit(X_scaled)
        
        self.models['isolation_forest'] = model
        self.scalers['isolation_forest'] = scaler
        
        return model
    
    def train_elliptic_envelope(self, data):
        """Train Elliptic Envelope model (Robust Covariance)"""
        X = data[self.feature_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = EllipticEnvelope(contamination=0.1, random_state=42)
        model.fit(X_scaled)
        
        self.models['elliptic_envelope'] = model
        self.scalers['elliptic_envelope'] = scaler
        
        return model
    
    def train_one_class_svm(self, data):
        """Train One-Class SVM model"""
        X = data[self.feature_columns]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        model.fit(X_scaled)
        
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
        
        return model
    
    def train_statistical_threshold(self, data):
        """Train statistical threshold-based detection"""
        thresholds = {}
        for col in self.feature_columns:
            mean = data[col].mean()
            std = data[col].std()
            thresholds[col] = {
                'lower': mean - 3 * std,
                'upper': mean + 3 * std,
                'mean': mean,
                'std': std
            }
        
        self.thresholds['statistical'] = thresholds
        return thresholds
    
    def detect_anomalies(self, data, model_type='isolation_forest'):
        """Detect anomalies using specified model"""
        X = data[self.feature_columns]
        
        if model_type == 'statistical':
            anomalies = np.zeros(len(data), dtype=bool)
            scores = np.zeros(len(data))
            
            for col in self.feature_columns:
                thresh = self.thresholds['statistical'][col]
                col_anomalies = (data[col] < thresh['lower']) | (data[col] > thresh['upper'])
                anomalies |= col_anomalies
                # Calculate z-scores
                z_scores = np.abs((data[col] - thresh['mean']) / thresh['std'])
                scores = np.maximum(scores, z_scores)
        
        else:
            X_scaled = self.scalers[model_type].transform(X)
            predictions = self.models[model_type].predict(X_scaled)
            anomalies = predictions == -1
            
            # Get anomaly scores
            if hasattr(self.models[model_type], 'score_samples'):
                scores = -self.models[model_type].score_samples(X_scaled)  # Higher = more anomalous
            elif hasattr(self.models[model_type], 'decision_function'):
                scores = -self.models[model_type].decision_function(X_scaled)
            else:
                scores = np.random.random(len(data))  # Fallback
        
        return anomalies, scores

def create_telemetry_plots(data, anomalies=None):
    """Create comprehensive telemetry visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature (°C)', 'Voltage (V)', 'Current (A)', 'Pressure (hPa)'),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    params = ['temperature', 'voltage', 'current', 'pressure']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (param, color) in enumerate(zip(params, colors)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Normal data line
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data[param],
                mode='lines',
                name=param.title(),
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Add markers for recent data points
        recent_data = data.tail(10)
        fig.add_trace(
            go.Scatter(
                x=recent_data['timestamp'],
                y=recent_data[param],
                mode='markers',
                marker=dict(color=color, size=6),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Highlight anomalies
        if anomalies is not None:
            anomaly_data = data[anomalies]
            if not anomaly_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=anomaly_data['timestamp'],
                        y=anomaly_data[param],
                        mode='markers',
                        name=f'{param.title()} Anomalies',
                        marker=dict(color='red', size=10, symbol='x-thin'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    fig.update_layout(
        height=600,
        title_text="🛰️ Satellite Telemetry Dashboard - Real-time Monitoring",
        title_x=0.5,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig

def create_system_health_chart(data):
    """Create system health overview chart"""
    latest_data = data.tail(20)
    
    fig = go.Figure()
    
    # Overall health score (simplified calculation)
    health_scores = []
    for _, row in latest_data.iterrows():
        # Simple health scoring based on parameter ranges
        temp_score = 1.0 if 18 <= row['temperature'] <= 28 else 0.5
        voltage_score = 1.0 if 11.5 <= row['voltage'] <= 12.5 else 0.5
        current_score = 1.0 if 2.5 <= row['current'] <= 3.5 else 0.5
        pressure_score = 1.0 if 1005 <= row['pressure'] <= 1025 else 0.5
        
        overall_health = (temp_score + voltage_score + current_score + pressure_score) / 4
        health_scores.append(overall_health * 100)
    
    fig.add_trace(go.Scatter(
        x=latest_data['timestamp'],
        y=health_scores,
        mode='lines+markers',
        name='System Health',
        line=dict(color='green', width=3),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="System Health Score (%)",
        xaxis_title="Time",
        yaxis_title="Health Score",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig

def generate_anomaly_report(data, anomalies, model_type):
    """Generate detailed anomaly report"""
    anomaly_count = anomalies.sum()
    total_points = len(data)
    anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
    
    report = f"""# 🛰️ Satellite Anomaly Detection Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Model Used:** {model_type.replace('_', ' ').title()}  
**Analysis Period:** {data['timestamp'].min()} to {data['timestamp'].max()}  

## Executive Summary
- **Total Data Points Analyzed:** {total_points:,}
- **Anomalies Detected:** {anomaly_count}
- **Anomaly Rate:** {anomaly_rate:.2f}%
- **System Status:** {"⚠️ ATTENTION REQUIRED" if anomaly_count > 0 else "✅ NOMINAL"}

## Parameter Analysis
"""
    
    # Parameter statistics
    for param in ['temperature', 'voltage', 'current', 'pressure']:
        mean_val = data[param].mean()
        std_val = data[param].std()
        min_val = data[param].min()
        max_val = data[param].max()
        
        report += f"""
### {param.title()}
- **Mean:** {mean_val:.2f}
- **Std Dev:** {std_val:.2f}
- **Range:** {min_val:.2f} to {max_val:.2f}
"""
    
    if anomaly_count > 0:
        report += f"\n## Detected Anomalies\n"
        anomaly_data = data[anomalies].copy()
        
        for i, (_, row) in enumerate(anomaly_data.iterrows()):
            if i < 10:  # Limit to first 10 anomalies
                report += f"""
**Anomaly #{i+1}** - {row['timestamp']}
- Temperature: {row['temperature']:.2f}°C
- Voltage: {row['voltage']:.2f}V  
- Current: {row['current']:.2f}A
- Pressure: {row['pressure']:.2f}hPa
- Mode: {row['satellite_mode']}
- Data Quality: {row['data_quality']:.1%}
"""
        
        if len(anomaly_data) > 10:
            report += f"\n*... and {len(anomaly_data) - 10} more anomalies*\n"
    
    else:
        report += "\n## ✅ No Anomalies Detected\nAll satellite systems operating within normal parameters during the analysis period.\n"
    
    report += f"""
## Recommendations
{'- Investigate recent anomalies immediately' if anomaly_count > 0 else '- Continue normal monitoring'}
- Monitor trending parameters closely
- Review maintenance schedules
- Update anomaly detection thresholds if needed

---
*Report generated by URSC Satellite Health Monitoring System*
"""
    
    return report

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = SimpleAnomalyDetector()
if 'simulator' not in st.session_state:
    st.session_state.simulator = SatelliteDataSimulator()

def main():
    st.title("🛰️ URSC Satellite Health Monitoring System")
    st.markdown("**AI-Based Anomaly Detection in Satellite Telemetry Data**")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration Panel")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "📊 Data Source",
        ["Simulate Real-time Data", "Upload CSV File"],
        help="Choose between simulated satellite data or upload your own telemetry CSV"
    )
    
    if data_source == "Simulate Real-time Data":
        st.sidebar.subheader("Simulation Parameters")
        hours = st.sidebar.slider("Analysis Period (hours)", 1, 72, 24)
        freq_minutes = st.sidebar.slider("Data Frequency (minutes)", 1, 30, 5)
        anomaly_rate = st.sidebar.slider("Anomaly Rate", 0.0, 0.2, 0.05, help="Percentage of data points that will be anomalous")
        
        if st.sidebar.button("🔄 Generate New Data", type="primary"):
            with st.spinner("🛰️ Generating satellite telemetry data..."):
                data = st.session_state.simulator.generate_data(hours, freq_minutes, anomaly_rate)
                st.session_state.data = data
                st.sidebar.success("✅ New data generated!")
    
    else:
        uploaded_file = st.sidebar.file_uploader(
            "📁 Upload Telemetry CSV", 
            type=['csv'],
            help="CSV should contain columns: temperature, voltage, current, pressure"
        )
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                if 'timestamp' not in data.columns:
                    data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(data), freq='5min')
                
                # Add missing columns with default values if needed
                required_cols = ['temperature', 'voltage', 'current', 'pressure']
                for col in required_cols:
                    if col not in data.columns:
                        st.sidebar.warning(f"Missing column: {col}")
                
                st.session_state.data = data
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    
    # Model selection
    st.sidebar.subheader("🤖 AI Model Selection")
    model_type = st.sidebar.selectbox(
        "Anomaly Detection Algorithm",
        ["isolation_forest", "elliptic_envelope", "one_class_svm", "statistical"],
        help="Choose the AI algorithm for anomaly detection"
    )
    
    model_descriptions = {
        "isolation_forest": "🌳 Isolation Forest - Tree-based anomaly detection",
        "elliptic_envelope": "📊 Elliptic Envelope - Robust covariance estimation", 
        "one_class_svm": "🎯 One-Class SVM - Support vector anomaly detection",
        "statistical": "📈 Statistical Threshold - Traditional 3-sigma rule"
    }
    st.sidebar.info(model_descriptions[model_type])
    
    # Main content
    if 'data' in st.session_state:
        data = st.session_state.data
        
        # Validate data
        required_columns = ['temperature', 'voltage', 'current', 'pressure']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            st.info("Please ensure your CSV contains: temperature, voltage, current, pressure")
            return
        
        # Train model and detect anomalies
        with st.spinner(f"🔍 Training {model_type.replace('_', ' ').title()} model..."):
            try:
                if model_type == 'isolation_forest':
                    st.session_state.detector.train_isolation_forest(data)
                elif model_type == 'elliptic_envelope':
                    st.session_state.detector.train_elliptic_envelope(data)
                elif model_type == 'one_class_svm':
                    st.session_state.detector.train_one_class_svm(data)
                else:  # statistical
                    st.session_state.detector.train_statistical_threshold(data)
                
                anomalies, scores = st.session_state.detector.detect_anomalies(data, model_type)
                
            except Exception as e:
                st.error(f"❌ Error in model training: {str(e)}")
                return
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Data Points", 
                f"{len(data):,}",
                help="Total number of telemetry readings analyzed"
            )
        
        with col2:
            anomaly_count = int(anomalies.sum())
            st.metric(
                "⚠️ Anomalies Detected", 
                anomaly_count,
                delta=f"{(anomaly_count/len(data)*100):.1f}% of data" if len(data) > 0 else "0%"
            )
        
        with col3:
            latest_temp = data['temperature'].iloc[-1]
            temp_status = "🔥" if latest_temp > 30 else "❄️" if latest_temp < 15 else "🌡️"
            st.metric(
                f"{temp_status} Current Temperature", 
                f"{latest_temp:.1f}°C"
            )
        
        with col4:
            latest_voltage = data['voltage'].iloc[-1]
            voltage_status = "🔋" if 11.5 <= latest_voltage <= 12.5 else "⚡"
            st.metric(
                f"{voltage_status} Current Voltage", 
                f"{latest_voltage:.2f}V"
            )
        
        # Alert system
        st.markdown("### 🚨 System Status")
        if anomalies.sum() > 0:
            recent_anomalies = anomalies[-20:].sum()  # Last 20 readings
            if recent_anomalies > 0:
                st.markdown("""
                <div class="alert-danger">
                    <strong>🚨 CRITICAL ALERT:</strong> Recent anomalies detected in satellite telemetry! 
                    <br>Immediate operator attention required. Check all subsystems.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-warning">
                    <strong>⚠️ WARNING:</strong> Historical anomalies detected in telemetry data.
                    <br>Monitor closely for recurring patterns and trending issues.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <strong>✅ NOMINAL STATUS:</strong> All satellite systems operating within normal parameters.
                <br>Continue routine monitoring operations.
            </div>
            """, unsafe_allow_html=True)
        
        # Main telemetry visualization
        st.markdown("### 📈 Real-time Telemetry Dashboard")
        fig = create_telemetry_plots(data, anomalies)
        st.plotly_chart(fig, use_container_width=True)
        
        # System health overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if anomalies.sum() > 0:
                st.markdown("### 🔍 Detected Anomalies")
                anomaly_data = data[anomalies][['timestamp'] + required_columns + ['satellite_mode']].copy()
                anomaly_data['timestamp'] = anomaly_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Style the dataframe
                st.dataframe(
                    anomaly_data, 
                    use_container_width=True,
                    height=min(300, len(anomaly_data) * 35 + 38)
                )
                
                if len(anomaly_data) > 10:
                    st.info(f"Showing recent anomalies. Total detected: {len(anomaly_data)}")
            else:
                st.markdown("### ✅ No Anomalies Detected")
                st.success("All telemetry parameters are within expected ranges.")
        
        with col2:
            st.markdown("### 🏥 System Health")
            health_fig = create_system_health_chart(data)
            st.plotly_chart(health_fig, use_container_width=True)
        
        # Current system parameters
        st.markdown("### 🔧 Current System Parameters")
        latest_data = data.iloc[-1]
        
        param_col1, param_col2 = st.columns(2)
        
        with param_col1:
            st.markdown("**Primary Systems:**")
            st.write(f"🌡️ **Temperature:** {latest_data['temperature']:.2f}°C")
            st.write(f"⚡ **Voltage:** {latest_data['voltage']:.2f}V")
            st.write(f"🔌 **Current:** {latest_data['current']:.2f}A")
            st.write(f"🌐 **Pressure:** {latest_data['pressure']:.2f}hPa")
        
        with param_col2:
            st.markdown("**Operational Status:**")
            st.write(f"🛰️ **Mode:** {latest_data['satellite_mode']}")
            st.write(f"📊 **Data Quality:** {latest_data['data_quality']:.1%}")
            if 'signal_strength' in latest_data:
                st.write(f"📡 **Signal Strength:** {latest_data['signal_strength']:.1%}")
            st.write(f"⏰ **Last Update:** {latest_data['timestamp'].strftime('%H:%M:%S')}")
        
        # Anomaly score visualization
        if len(scores) > 0 and max(scores) > 0:
            st.markdown("### 📊 Anomaly Score Analysis")
            fig_hist = px.histogram(
                x=scores,
                nbins=30,
                title="Distribution of Anomaly Scores",
                labels={'x': 'Anomaly Score', 'y': 'Frequency'},
                template="plotly_white"
            )
            fig_hist.add_vline(
                x=np.percentile(scores, 90), 
                line_dash="dash", 
                line_color="red",
                annotation_text="90th Percentile"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Report generation
        st.markdown("### 📋 Generate Mission Report")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📄 Generate Detailed Report", type="primary"):
                with st.spinner("📝 Generating comprehensive anomaly report..."):
                    report = generate_anomaly_report(data, anomalies, model_type)
                    st.session_state.report = report
                    st.success("✅ Report generated successfully!")
        
        with col2:
            if 'report' in st.session_state:
                st.download_button(
                    label="💾 Download Report (.md)",
                    data=st.session_state.report,
                    file_name=f"URSC_satellite_anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        # Show report preview
        if 'report' in st.session_state:
            st.markdown("### 📖 Report Preview")
            with st.expander("View Generated Report"):
                st.markdown(st.session_state.report)
    
    else:
        # Welcome screen
        st.markdown("### 👋 Welcome to URSC Satellite Health Monitor")
        st.info("🔧 Please configure your data source in the sidebar to begin monitoring satellite telemetry.")
        
        # Show expected data format
        st.markdown("### 📋 Expected Data Format")
        st.markdown("Your CSV file should contain the following columns:")
        
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 00:00:00', periods=5, freq='5min'),
            'temperature': [22.1, 23.5, 21.8, 24.2, 22.9],
            'voltage': [12.0, 11.9, 12.1, 11.8, 12.0],
            'current': [3.0, 2.9, 3.1, 3.2, 2.8],
            'pressure': [1015, 1016, 1014, 1017, 1015]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        **Column Descriptions:**
        - `timestamp`: Date and time of telemetry reading
        - `temperature`: Satellite temperature in Celsius  
        - `voltage`: Battery/power system voltage
        - `current`: Current draw in Amperes
        - `pressure`: Internal pressure in hPa
        """)

if __name__ == "__main__":
    main()
