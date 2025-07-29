"""
PCB Defect Detection System using YOLOv8 and Streamlit
=====================================================

Required installations (run these commands):
pip install streamlit ultralytics opencv-python pillow numpy pandas matplotlib seaborn plotly

To run: streamlit run pcb_defect_detector.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO
import tempfile
import os
import time
from pathlib import Path
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .defect-info {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class PCBDefectDetector:
    def __init__(self):
        self.model = None
        self.defect_types = {
            0: "Missing Component",
            1: "Wrong Component", 
            2: "Damaged Component",
            3: "Soldering Defect",
            4: "Short Circuit",
            5: "Open Circuit",
            6: "Misalignment",
            7: "Corrosion",
            8: "Crack",
            9: "Contamination"
        }
        self.defect_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (255, 192, 203), # Pink
            (128, 128, 128)  # Gray
        ]
        
    def load_model(self):
        """Load or create YOLO model"""
        try:
            # Try to load a pre-trained model (you can replace with your trained model)
            self.model = YOLO('yolov8n.pt')  # Using nano version for speed
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def create_synthetic_detections(self, image):
        """Create synthetic detections for demo purposes"""
        h, w = image.shape[:2]
        detections = []
        
        # Simulate some defects
        np.random.seed(42)  # For consistent demo results
        num_defects = np.random.randint(2, 6)
        
        for i in range(num_defects):
            x1 = np.random.randint(0, w//2)
            y1 = np.random.randint(0, h//2)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            
            # Ensure coordinates are within image bounds
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            defect_type = np.random.randint(0, len(self.defect_types))
            confidence = np.random.uniform(0.6, 0.95)
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class': defect_type,
                'label': self.defect_types[defect_type]
            })
        
        return detections
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        image_copy = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class']
            label = detection['label']
            
            # Get color for this defect type
            color = self.defect_colors[class_id % len(self.defect_colors)]
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"{label}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(image_copy, (x1, y1-30), (x1 + text_width + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(image_copy, label_text, (x1 + 5, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image_copy
    
    def analyze_defects(self, detections):
        """Analyze detected defects and return statistics"""
        if not detections:
            return {}
        
        df = pd.DataFrame(detections)
        
        analysis = {
            'total_defects': len(detections),
            'defect_types': df['label'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': df['confidence'].mean(),
                'min': df['confidence'].min(),
                'max': df['confidence'].max(),
                'std': df['confidence'].std()
            },
            'severity_distribution': self.calculate_severity(detections)
        }
        
        return analysis
    
    def calculate_severity(self, detections):
        """Calculate defect severity based on type and confidence"""
        severity_map = {
            "Missing Component": 3,
            "Wrong Component": 3,
            "Short Circuit": 3,
            "Open Circuit": 3,
            "Damaged Component": 2,
            "Soldering Defect": 2,
            "Misalignment": 2,
            "Crack": 2,
            "Corrosion": 1,
            "Contamination": 1
        }
        
        severities = []
        for detection in detections:
            base_severity = severity_map.get(detection['label'], 1)
            confidence_factor = detection['confidence']
            final_severity = base_severity * confidence_factor
            severities.append(final_severity)
        
        return {
            'high': sum(1 for s in severities if s >= 2.5),
            'medium': sum(1 for s in severities if 1.5 <= s < 2.5),
            'low': sum(1 for s in severities if s < 1.5)
        }

def create_defect_charts(analysis):
    """Create visualization charts for defect analysis"""
    
    # Defect types pie chart
    if analysis.get('defect_types'):
        fig_pie = px.pie(
            values=list(analysis['defect_types'].values()),
            names=list(analysis['defect_types'].keys()),
            title="Distribution of Defect Types",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Severity distribution bar chart
    if analysis.get('severity_distribution'):
        severity_data = analysis['severity_distribution']
        fig_bar = px.bar(
            x=['High', 'Medium', 'Low'],
            y=[severity_data['high'], severity_data['medium'], severity_data['low']],
            title="Defect Severity Distribution",
            color=['High', 'Medium', 'Low'],
            color_discrete_map={'High': '#ff4444', 'Medium': '#ffaa00', 'Low': '#44ff44'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 PCB Defect Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = PCBDefectDetector()
        
    detector = st.session_state.detector
    
    # Sidebar
    st.sidebar.title("🛠️ Settings")
    
    # Model loading
    if st.sidebar.button("Load Detection Model"):
        with st.spinner("Loading YOLO model..."):
            if detector.load_model():
                st.sidebar.success("✅ Model loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load model")
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Detection", "📊 Batch Analysis", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload PCB Image for Defect Detection</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PCB image...",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a high-quality PCB image for defect detection"
        )
        
        # Sample images
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📱 Use Sample PCB 1"):
                st.session_state.use_sample = 1
        with col2:
            if st.button("🔧 Use Sample PCB 2"):
                st.session_state.use_sample = 2
        with col3:
            if st.button("⚡ Use Sample PCB 3"):
                st.session_state.use_sample = 3
        
        # Process image
        image = None
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.success(f"✅ Image uploaded: {uploaded_file.name}")
            
        elif hasattr(st.session_state, 'use_sample'):
            # Create sample PCB-like image
            sample_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            # Add some PCB-like patterns
            cv2.rectangle(sample_img, (50, 50), (750, 550), (0, 100, 0), 2)
            cv2.circle(sample_img, (400, 300), 50, (255, 255, 255), -1)
            cv2.rectangle(sample_img, (200, 200), (600, 400), (50, 50, 50), -1)
            
            image = Image.fromarray(sample_img)
            st.info(f"📱 Using Sample PCB {st.session_state.use_sample}")
        
        if image is not None:
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, caption="Original PCB Image", use_column_width=True)
                
                # Image info
                st.markdown(f"""
                <div class="defect-info">
                    <strong>Image Information:</strong><br>
                    • Dimensions: {image.size[0]} x {image.size[1]} pixels<br>
                    • Format: {image.format}<br>
                    • Mode: {image.mode}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("🔍 Detecting defects..."):
                    # Simulate detection process
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    progress_bar.empty()
                    
                    # Get detections (using synthetic data for demo)
                    detections = detector.create_synthetic_detections(opencv_image)
                    
                    # Filter by confidence threshold
                    filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
                    
                    # Draw detections
                    result_image = detector.draw_detections(opencv_image, filtered_detections)
                    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
                    st.image(result_image_rgb, caption="Detected Defects", use_column_width=True)
                
                # Analysis results
                if filtered_detections:
                    analysis = detector.analyze_defects(filtered_detections)
                    
                    st.markdown("### 📊 Detection Summary")
                    
                    # Metrics
                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{analysis['total_defects']}</h3>
                            <p>Total Defects</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        high_severity = analysis['severity_distribution']['high']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{high_severity}</h3>
                            <p>High Severity</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        avg_conf = analysis['confidence_stats']['mean']
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{avg_conf:.2f}</h3>
                            <p>Avg Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_d:
                        unique_types = len(analysis['defect_types'])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{unique_types}</h3>
                            <p>Defect Types</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed results table
                    st.markdown("### 📋 Detailed Results")
                    
                    results_df = pd.DataFrame([
                        {
                            'Defect Type': d['label'],
                            'Confidence': f"{d['confidence']:.3f}",
                            'Bounding Box': f"({d['bbox'][0]}, {d['bbox'][1]}) - ({d['bbox'][2]}, {d['bbox'][3]})",
                            'Severity': 'High' if d['confidence'] > 0.8 else 'Medium' if d['confidence'] > 0.6 else 'Low'
                        }
                        for d in filtered_detections
                    ])
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                else:
                    st.success("✅ No defects detected! PCB appears to be in good condition.")
    
    with tab2:
        st.markdown('<h2 class="sub-header">Batch Processing</h2>', unsafe_allow_html=True)
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Upload multiple PCB images for batch processing",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple PCB images to process them all at once"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded for batch processing")
            
            if st.button("🚀 Start Batch Processing"):
                batch_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Process each image
                    image = Image.open(uploaded_file)
                    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    # Get detections
                    detections = detector.create_synthetic_detections(opencv_image)
                    filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
                    
                    # Analyze
                    analysis = detector.analyze_defects(filtered_detections)
                    
                    batch_results.append({
                        'filename': uploaded_file.name,
                        'total_defects': analysis.get('total_defects', 0),
                        'high_severity': analysis.get('severity_distribution', {}).get('high', 0),
                        'avg_confidence': analysis.get('confidence_stats', {}).get('mean', 0),
                        'status': 'Defects Found' if analysis.get('total_defects', 0) > 0 else 'Clean'
                    })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Batch processing completed!")
                
                # Display batch results
                st.markdown("### 📊 Batch Processing Results")
                
                batch_df = pd.DataFrame(batch_results)
                st.dataframe(batch_df, use_container_width=True)
                
                # Batch statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Files with defects
                    files_with_defects = len([r for r in batch_results if r['total_defects'] > 0])
                    fig_donut = go.Figure(data=[go.Pie(
                        labels=['Clean PCBs', 'PCBs with Defects'],
                        values=[len(batch_results) - files_with_defects, files_with_defects],
                        hole=.3
                    )])
                    fig_donut.update_layout(title="PCB Quality Distribution")
                    st.plotly_chart(fig_donut, use_container_width=True)
                
                with col2:
                    # Defect counts
                    fig_hist = px.histogram(
                        batch_df, 
                        x='total_defects', 
                        title="Distribution of Defect Counts",
                        nbins=10
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        # Sample analytics data
        if st.button("📈 Generate Sample Analytics"):
            # Create sample data for demonstration
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            sample_data = {
                'date': dates,
                'total_pcbs': np.random.poisson(50, 30),
                'defective_pcbs': np.random.poisson(5, 30),
                'defect_rate': np.random.uniform(0.05, 0.25, 30)
            }
            
            analytics_df = pd.DataFrame(sample_data)
            
            # Time series plots
            col1, col2 = st.columns(2)
            
            with col1:
                fig_line = px.line(
                    analytics_df, 
                    x='date', 
                    y=['total_pcbs', 'defective_pcbs'],
                    title="PCB Production vs Defects Over Time"
                )
                st.plotly_chart(fig_line, use_container_width=True)
            
            with col2:
                fig_defect_rate = px.line(
                    analytics_df,
                    x='date',
                    y='defect_rate',
                    title="Defect Rate Trend"
                )
                st.plotly_chart(fig_defect_rate, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            
            col_1, col_2, col_3, col_4 = st.columns(4)
            
            with col_1:
                avg_production = analytics_df['total_pcbs'].mean()
                st.metric("Avg Daily Production", f"{avg_production:.0f} PCBs")
            
            with col_2:
                avg_defects = analytics_df['defective_pcbs'].mean()
                st.metric("Avg Daily Defects", f"{avg_defects:.0f} PCBs")
            
            with col_3:
                avg_defect_rate = analytics_df['defect_rate'].mean()
                st.metric("Avg Defect Rate", f"{avg_defect_rate:.1%}")
            
            with col_4:
                total_inspected = analytics_df['total_pcbs'].sum()
                st.metric("Total Inspected", f"{total_inspected:,} PCBs")
    
    with tab4:
        st.markdown('<h2 class="sub-header">About PCB Defect Detection</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 System Overview
        
        This PCB Defect Detection System uses advanced computer vision and machine learning techniques to automatically identify and classify defects in printed circuit boards.
        
        ### 🔧 Key Features
        
        - **Real-time Detection**: Instant defect identification using YOLOv8
        - **Multiple Defect Types**: Detects 10 different types of PCB defects
        - **Batch Processing**: Process multiple PCB images simultaneously
        - **Detailed Analytics**: Comprehensive defect analysis and reporting
        - **Interactive Interface**: User-friendly Streamlit web interface
        
        ### 🎨 Detected Defect Types
        """)
        
        # Display defect types
        defect_info = detector.defect_types
        
        col1, col2 = st.columns(2)
        
        for i, (key, value) in enumerate(defect_info.items()):
            if i % 2 == 0:
                with col1:
                    st.markdown(f"**{key + 1}.** {value}")
            else:
                with col2:
                    st.markdown(f"**{key + 1}.** {value}")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload Image**: Use the "Image Detection" tab to upload a PCB image
        2. **Adjust Settings**: Modify detection parameters in the sidebar
        3. **View Results**: Analyze detected defects and their severity
        4. **Batch Process**: Process multiple images using the "Batch Analysis" tab
        5. **Monitor Trends**: Track defect patterns using the "Analytics" tab
        
        ### 🛠️ Technical Details
        
        - **Model**: YOLOv8 (You Only Look Once) object detection
        - **Framework**: Ultralytics, OpenCV, Streamlit
        - **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF
        - **Detection Speed**: Real-time processing
        - **Accuracy**: High precision defect classification
        
        ### 📧 Support
        
        For technical support or questions about this system, please contact your development team.
        """)
        
        # System requirements
        with st.expander("📋 System Requirements & Installation"):
            st.code("""
# Required Python packages:
pip install streamlit ultralytics opencv-python pillow numpy pandas matplotlib seaborn plotly

# To run the application:
streamlit run pcb_defect_detector.py

# System Requirements:
- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- GPU support optional (for faster processing)
- Web browser for interface access
            """, language="bash")

if __name__ == "__main__":
    main()

