import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

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
                    if param == 'temperature':
                        values[idx:idx+3] += np.random.uniform(8, 15)
                elif anomaly_type == 'power_drop':
                    if param in ['voltage', 'current']:
                        drop_duration = min(5, n_points - idx)
                        values[idx:idx+drop_duration] *= np.random.uniform(0.6, 0.8)
                elif anomaly_type == 'sensor_drift':
                    drift_duration = min(20, n_points - idx)
                    drift_factor = np.linspace(1, np.random.uniform(1.2, 1.5), drift_duration)
                    values[idx:idx+drift_duration] *= drift_factor
                elif anomaly_type == 'eclipse_anomaly':
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
    """Simple statistical anomaly detection without external ML libraries"""
    
    def __init__(self):
        self.thresholds = {}
        self.feature_columns = ['temperature', 'voltage', 'current', 'pressure']
        self.trained = False
    
    def train_statistical_model(self, data):
        """Train statistical threshold-based detection using Z-score and IQR methods"""
        self.thresholds = {}
        
        for col in self.feature_columns:
            if col in data.columns:
                values = data[col]
                
                # Calculate basic statistics
                mean_val = values.mean()
                std_val = values.std()
                median_val = values.median()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                
                # Multiple threshold methods
                self.thresholds[col] = {
                    # Z-score method (3-sigma rule)
                    'zscore_lower': mean_val - 3 * std_val,
                    'zscore_upper': mean_val + 3 * std_val,
                    
                    # IQR method
                    'iqr_lower': q1 - 1.5 * iqr,
                    'iqr_upper': q3 + 1.5 * iqr,
                    
                    # Modified Z-score using median
                    'mad': np.median(np.abs(values - median_val)),
                    'median': median_val,
                    
                    # Basic stats for scoring
                    'mean': mean_val,
                    'std': std_val,
                    'min': values.min(),
                    'max': values.max()
                }
        
        self.trained = True
        return self.thresholds
    
    def detect_anomalies(self, data, method='combined'):
        """Detect anomalies using statistical methods"""
        if not self.trained:
            st.error("Model not trained! Please train the model first.")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        anomalies = np.zeros(len(data), dtype=bool)
        scores = np.zeros(len(data))
        
        for col in self.feature_columns:
            if col in data.columns:
                values = data[col]
                thresh = self.thresholds[col]
                
                if method == 'zscore':
                    # Z-score method
                    col_anomalies = (values < thresh['zscore_lower']) | (values > thresh['zscore_upper'])
                    col_scores = np.abs((values - thresh['mean']) / thresh['std'])
                
                elif method == 'iqr':
                    # IQR method
                    col_anomalies = (values < thresh['iqr_lower']) | (values > thresh['iqr_upper'])
                    col_scores = np.maximum(
                        (thresh['iqr_lower'] - values) / (thresh['iqr_lower'] - thresh['min'] + 1e-6),
                        (values - thresh['iqr_upper']) / (thresh['max'] - thresh['iqr_upper'] + 1e-6)
                    )
                    col_scores = np.maximum(col_scores, 0)
                
                else:  # combined method
                    # Combine both methods
                    zscore_anomalies = (values < thresh['zscore_lower']) | (values > thresh['zscore_upper'])
                    iqr_anomalies = (values < thresh['iqr_lower']) | (values > thresh['iqr_upper'])
                    
                    col_anomalies = zscore_anomalies | iqr_anomalies
                    
                    # Combined scoring
                    zscore_scores = np.abs((values - thresh['mean']) / thresh['std'])
                    iqr_scores = np.maximum(
                        (thresh['iqr_lower'] - values) / (thresh['iqr_lower'] - thresh['min'] + 1e-6),
                        (values - thresh['iqr_upper']) / (thresh['max'] - thresh['iqr_upper'] + 1e-6)
                    )
                    iqr_scores = np.maximum(iqr_scores, 0)
                    col_scores = np.maximum(zscore_scores, iqr_scores)
                
                anomalies |= col_anomalies
                scores = np.maximum(scores, col_scores)
        
        return anomalies, scores

def create_telemetry_plots(data, anomalies=None):
    """Create telemetry visualization using matplotlib"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🛰️ Satellite Telemetry Dashboard - Real-time Monitoring', fontsize=16, fontweight='bold')
    
    params = ['temperature', 'voltage', 'current', 'pressure']
    units = ['°C', 'V', 'A', 'hPa']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (param, unit, color) in enumerate(zip(params, units, colors)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        if param in data.columns:
            # Plot normal data
            ax.plot(data['timestamp'], data[param], color=color, linewidth=2, alpha=0.8, label='Normal')
            
            # Highlight anomalies
            if anomalies is not None:
                anomaly_data = data[anomalies]
                if not anomaly_data.empty:
                    ax.scatter(anomaly_data['timestamp'], anomaly_data[param], 
                             color='red', s=50, marker='x', linewidth=3, label='Anomalies', zorder=5)
            
            ax.set_title(f'{param.title()} ({unit})', fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel(f'{param.title()} ({unit})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def create_anomaly_distribution_plot(scores):
    """Create anomaly score distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of anomaly scores
    ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.percentile(scores, 90), color='red', linestyle='--', 
                label=f'90th Percentile: {np.percentile(scores, 90):.2f}')
    ax1.set_title('Anomaly Score Distribution')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time series of anomaly scores
    ax2.plot(scores, color='orange', linewidth=2)
    ax2.axhline(np.percentile(scores, 90), color='red', linestyle='--', alpha=0.7)
    ax2.set_title('Anomaly Scores Over Time')
    ax2.set_xlabel('Data Point Index')
    ax2.set_ylabel('Anomaly Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_system_health_chart(data):
    """Create system health overview"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Simple health scoring
    latest_data = data.tail(50)  # Last 50 points
    health_scores = []
    
    for _, row in latest_data.iterrows():
        # Simple health scoring based on parameter ranges
        temp_score = 1.0 if 18 <= row['temperature'] <= 28 else 0.5
        voltage_score = 1.0 if 11.5 <= row['voltage'] <= 12.5 else 0.5
        current_score = 1.0 if 2.5 <= row['current'] <= 3.5 else 0.5
        pressure_score = 1.0 if 1005 <= row['pressure'] <= 1025 else 0.5
        
        overall_health = (temp_score + voltage_score + current_score + pressure_score) / 4
        health_scores.append(overall_health * 100)
    
    ax.plot(latest_data['timestamp'], health_scores, color='green', linewidth=3, marker='o', markersize=4)
    ax.fill_between(latest_data['timestamp'], health_scores, alpha=0.3, color='green')
    ax.set_title('System Health Score (%)', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Health Score (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    # Add threshold lines
    ax.axhline(90, color='green', linestyle='--', alpha=0.7, label='Excellent')
    ax.axhline(70, color='orange', linestyle='--', alpha=0.7, label='Good')
    ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='Warning')
    ax.legend()
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    return fig

def generate_anomaly_report(data, anomalies, method):
    """Generate detailed anomaly report"""
    anomaly_count = anomalies.sum()
    total_points = len(data)
    anomaly_rate = (anomaly_count / total_points * 100) if total_points > 0 else 0
    
    report = f"""# 🛰️ Satellite Anomaly Detection Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Detection Method:** {method.replace('_', ' ').title()}  
**Analysis Period:** {data['timestamp'].min()} to {data['timestamp'].max()}  

## Executive Summary
- **Total Data Points Analyzed:** {total_points:,}
- **Anomalies Detected:** {anomaly_count}
- **Anomaly Rate:** {anomaly_rate:.2f}%
- **System Status:** {"⚠️ ATTENTION REQUIRED" if anomaly_count > 0 else "✅ NOMINAL"}

## Parameter Analysis
"""
    
    # Parameter statistics
    required_columns = ['temperature', 'voltage', 'current', 'pressure']
    for param in required_columns:
        if param in data.columns:
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
- Mode: {row.get('satellite_mode', 'N/A')}
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
    st.title("🛰️ AI-Based Anomaly Detection in Satellite Health Telemetry")
    st.markdown("**URSC Satellite Health Monitoring System**")
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
                else:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                
                st.session_state.data = data
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    
    # Detection method selection
    st.sidebar.subheader("🔍 Detection Method")
    detection_method = st.sidebar.selectbox(
        "Anomaly Detection Algorithm",
        ["combined", "zscore", "iqr"],
        help="Choose the statistical method for anomaly detection"
    )
    
    method_descriptions = {
        "combined": "🔄 Combined Method - Uses both Z-score and IQR methods",
        "zscore": "📊 Z-Score Method - Traditional 3-sigma statistical rule", 
        "iqr": "📈 IQR Method - Interquartile range outlier detection"
    }
    st.sidebar.info(method_descriptions[detection_method])
    
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
        with st.spinner(f"🔍 Training {detection_method} anomaly detection model..."):
            try:
                st.session_state.detector.train_statistical_model(data)
                anomalies, scores = st.session_state.detector.detect_anomalies(data, detection_method)
                
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
        try:
            fig = create_telemetry_plots(data, anomalies)
            st.pyplot(fig)
            plt.close(fig)  # Close to prevent memory issues
        except Exception as e:
            st.error(f"Error creating telemetry plots: {str(e)}")
        
        # Two column layout for additional information
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if anomalies.sum() > 0:
                st.markdown("### 🔍 Detected Anomalies")
                anomaly_data = data[anomalies][['timestamp'] + required_columns].copy()
                if 'satellite_mode' in data.columns:
                    anomaly_data['satellite_mode'] = data[anomalies]['satellite_mode']
                
                anomaly_data['timestamp'] = anomaly_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Display the dataframe
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
            try:
                health_fig = create_system_health_chart(data)
                st.pyplot(health_fig)
                plt.close(health_fig)
            except Exception as e:
                st.error(f"Error creating health chart: {str(e)}")
        
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
            if 'satellite_mode' in latest_data:
                st.write(f"🛰️ **Mode:** {latest_data['satellite_mode']}")
            if 'data_quality' in latest_data:
                st.write(f"📊 **Data Quality:** {latest_data['data_quality']:.1%}")
            if 'signal_strength' in latest_data:
                st.write(f"📡 **Signal Strength:** {latest_data['signal_strength']:.1%}")
            st.write(f"⏰ **Last Update:** {latest_data['timestamp'].strftime('%H:%M:%S')}")
        
        # Anomaly score visualization
        if len(scores) > 0 and max(scores) > 0:
            st.markdown("### 📊 Anomaly Score Analysis")
            try:
                fig_dist = create_anomaly_distribution_plot(scores)
                st.pyplot(fig_dist)
                plt.close(fig_dist)
            except Exception as e:
                st.error(f"Error creating anomaly distribution plot: {str(e)}")
        
        # Report generation
        st.markdown("### 📋 Generate Mission Report")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📄 Generate Detailed Report", type="primary"):
                with st.spinner("📝 Generating comprehensive anomaly report..."):
                    report = generate_anomaly_report(data, anomalies, detection_method)
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
        
        # Show project features
        st.markdown("### 🚀 Key Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **🤖 AI Detection Methods:**
            - Statistical Z-Score Analysis
            - Interquartile Range (IQR) Method
            - Combined Multi-Method Approach
            - Real-time Anomaly Scoring
            """)
            
            st.markdown("""
            **📊 Monitoring Capabilities:**
            - Temperature Monitoring
            - Power System Analysis
            - Pressure Variations
            - System Health Scoring
            """)
        
        with feature_col2:
            st.markdown("""
            **⚠️ Alert System:**
            - Real-time Anomaly Detection
            - Critical/Warning/Normal Status
            - Historical Pattern Analysis
            - Automated Report Generation
            """)
            
            st.markdown("""
            **📋 Reporting Features:**
            - Detailed Anomaly Reports
            - Statistical Analysis
            - Downloadable Documentation
            - Mission-Ready Formats
            """)

if __name__ == "__main__":
    main()
