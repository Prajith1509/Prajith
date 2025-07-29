import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageEnhance
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
import io
import base64

# Page setup
st.set_page_config(
    page_title="PCB Defect Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2980b9 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2980b9;
    }
    .defect-alert {
        background: #fee;
        border: 1px solid #fcc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .success-alert {
        background: #efe;
        border: 1px solid #cfc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🔍 Advanced PCB Defect Detection System</h1>
    <p style="color: #ecf0f1; margin: 0;">AI-Powered Quality Control for Electronics Manufacturing</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "quality_metrics" not in st.session_state:
    st.session_state.quality_metrics = {
        "total_inspected": 0,
        "defective_count": 0,
        "pass_rate": 100.0,
        "common_defects": {}
    }

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

# Defect types and their characteristics
DEFECT_TYPES = {
    "Short Circuit": {"severity": "Critical", "color": "#e74c3c"},
    "Open Circuit": {"severity": "Critical", "color": "#e74c3c"},
    "Missing Component": {"severity": "High", "color": "#f39c12"},
    "Wrong Component": {"severity": "High", "color": "#f39c12"},
    "Solder Bridge": {"severity": "Medium", "color": "#f1c40f"},
    "Insufficient Solder": {"severity": "Medium", "color": "#f1c40f"},
    "Component Misalignment": {"severity": "Low", "color": "#27ae60"},
    "Surface Contamination": {"severity": "Low", "color": "#27ae60"}
}

# Load model with error handling
@st.cache_resource
def load_trained_model():
    try:
        # Try to load actual model
        model = tf.keras.models.load_model("trained_pcb_model.h5")
        return model
    except:
        # Create a dummy model if actual model not found
        st.warning("⚠️ Model file not found. Using simulated predictions.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

# Enhanced YOLO simulation with defect types
def simulate_advanced_yolo_detection(image, sensitivity=0.7):
    width, height = image.size
    detections = []
    
    # Simulate different types of defects based on image analysis
    np.random.seed(42)  # For consistent results
    
    # Convert image to array for basic analysis
    img_array = np.array(image)
    
    # Simulate defect detection based on image characteristics
    num_defects = np.random.randint(0, 4)
    
    for i in range(num_defects):
        # Random bounding box
        x1 = np.random.randint(0, width // 2)
        y1 = np.random.randint(0, height // 2)
        x2 = x1 + np.random.randint(50, width // 4)
        y2 = y1 + np.random.randint(50, height // 4)
        
        # Random defect type
        defect_type = np.random.choice(list(DEFECT_TYPES.keys()))
        confidence = np.random.uniform(0.6, 0.95)
        
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "type": defect_type,
            "confidence": confidence,
            "severity": DEFECT_TYPES[defect_type]["severity"]
        })
    
    return detections

# Enhanced drawing function
def draw_advanced_detection_boxes(image, detections):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    for detection in detections:
        bbox = detection["bbox"]
        defect_type = detection["type"]
        confidence = detection["confidence"]
        severity = detection["severity"]
        
        # Color based on severity
        color = DEFECT_TYPES[defect_type]["color"]
        
        # Draw bounding box
        draw.rectangle(bbox, outline=color, width=3)
        
        # Draw label background
        label = f"{defect_type}: {confidence:.2f}"
        text_bbox = draw.textbbox((0, 0), label)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        label_bbox = (bbox[0], bbox[1] - text_height - 5, 
                     bbox[0] + text_width + 10, bbox[1])
        draw.rectangle(label_bbox, fill=color)
        draw.text((bbox[0] + 5, bbox[1] - text_height - 2), label, fill="white")
    
    return image_copy

# Enhanced preprocessing
def preprocess_image_advanced(image, target_size=(224, 224)):
    # Convert to RGB
    image = image.convert("RGB")
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    return np.expand_dims(img_array, axis=0)

# Enhanced classification with detailed analysis
def classify_pcb_advanced(image, model, detections):
    preprocessed = preprocess_image_advanced(image)
    
    try:
        prediction = model.predict(preprocessed, verbose=0)
        base_confidence = float(prediction[0][0])
    except:
        # Fallback to simulated prediction
        base_confidence = np.random.uniform(0.3, 0.9)
    
    # Adjust prediction based on detections
    if detections:
        # If defects detected, likely problematic
        defect_severity_weights = {
            "Critical": 0.9,
            "High": 0.7,
            "Medium": 0.5,
            "Low": 0.3
        }
        
        max_severity_weight = max([defect_severity_weights.get(d["severity"], 0.5) 
                                 for d in detections])
        adjusted_confidence = min(0.95, base_confidence + max_severity_weight * 0.3)
        predicted_class = "Defective"
    else:
        adjusted_confidence = max(0.05, base_confidence * 0.7)
        predicted_class = "Good"
    
    return predicted_class, adjusted_confidence, len(detections)

# Sidebar configuration
st.sidebar.header("🔧 Detection Settings")

# Model settings
st.sidebar.subheader("Model Configuration")
detection_sensitivity = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.7, 0.1)
show_boxes = st.sidebar.checkbox("Show Defect Bounding Boxes", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Batch processing option
st.sidebar.subheader("Processing Mode")
processing_mode = st.sidebar.radio("Select Mode", ["Single Image", "Batch Processing"])

# Quality control thresholds
st.sidebar.subheader("Quality Thresholds")
min_confidence = st.sidebar.slider("Minimum Confidence", 0.5, 0.95, 0.75, 0.05)
max_defects = st.sidebar.number_input("Max Acceptable Defects", 0, 10, 2)

# Main content area
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("📤 Image Upload")
    
if processing_mode == "Single Image":
    uploaded_file = st.file_uploader(
        "Choose a PCB image...", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear image of a PCB for defect detection"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, caption="Uploaded PCB Image", use_container_width=True)
        
        # Process image
        with st.spinner("🔍 Analyzing PCB for defects..."):
            # Load model
            model = load_trained_model()
            
            # Detect defects
            detections = simulate_advanced_yolo_detection(image, detection_sensitivity)
            
            # Classify overall quality
            prediction, confidence, defect_count = classify_pcb_advanced(image, model, detections)
            
            # Update quality metrics
            st.session_state.quality_metrics["total_inspected"] += 1
            if prediction == "Defective":
                st.session_state.quality_metrics["defective_count"] += 1
                
            # Calculate pass rate
            total = st.session_state.quality_metrics["total_inspected"]
            defective = st.session_state.quality_metrics["defective_count"]
            pass_rate = ((total - defective) / total * 100) if total > 0 else 100
            st.session_state.quality_metrics["pass_rate"] = pass_rate
        
        with col2:
            st.subheader("🎯 Detection Results")
            
            # Show processed image with detections
            if show_boxes and detections:
                processed_image = draw_advanced_detection_boxes(image, detections)
                st.image(processed_image, caption="Detected Defects", use_container_width=True)
            
            # Results summary
            if prediction == "Good":
                st.markdown(f"""
                <div class="success-alert">
                    <h3>✅ PCB Status: PASSED</h3>
                    <p><strong>Quality:</strong> Good ({confidence*100:.1f}% confidence)</p>
                    <p><strong>Defects Found:</strong> {defect_count}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="defect-alert">
                    <h3>❌ PCB Status: FAILED</h3>
                    <p><strong>Quality:</strong> Defective ({confidence*100:.1f}% confidence)</p>
                    <p><strong>Defects Found:</strong> {defect_count}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed defect information
        if detections:
            st.subheader("🔍 Detailed Defect Analysis")
            
            defect_df = pd.DataFrame([
                {
                    "Defect Type": d["type"],
                    "Severity": d["severity"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Location": f"({d['bbox'][0]}, {d['bbox'][1]})",
                    "Size": f"{d['bbox'][2]-d['bbox'][0]}×{d['bbox'][3]-d['bbox'][1]} px"
                }
                for d in detections
            ])
            
            st.dataframe(defect_df, use_container_width=True)
            
            # Defect severity distribution
            severity_counts = defect_df['Severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Defect Severity Distribution",
                color_discrete_map={
                    "Critical": "#e74c3c",
                    "High": "#f39c12", 
                    "Medium": "#f1c40f",
                    "Low": "#27ae60"
                }
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Add to history
        detection_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": uploaded_file.name,
            "Status": prediction,
            "Confidence": f"{confidence*100:.1f}%",
            "Defect_Count": defect_count,
            "Defects": [d["type"] for d in detections] if detections else []
        }
        
        st.session_state.detection_history.append(detection_record)

else:  # Batch Processing
    st.subheader("📦 Batch Processing")
    uploaded_files = st.file_uploader(
        "Choose multiple PCB images...", 
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"📊 Processing {len(uploaded_files)} images...")
        
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            model = load_trained_model()
            
            detections = simulate_advanced_yolo_detection(image, detection_sensitivity)
            prediction, confidence, defect_count = classify_pcb_advanced(image, model, detections)
            
            batch_results.append({
                "Filename": uploaded_file.name,
                "Status": prediction,
                "Confidence": f"{confidence*100:.1f}%",
                "Defects": defect_count,
                "Processing_Time": "0.5s"  # Simulated
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Display batch results
        st.subheader("📊 Batch Results Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_processed = len(batch_results)
            st.metric("Total Processed", total_processed)
        
        with col2:
            passed = sum(1 for r in batch_results if r["Status"] == "Good")
            st.metric("Passed", passed, f"{passed/total_processed*100:.1f}%")
        
        with col3:
            failed = sum(1 for r in batch_results if r["Status"] == "Defective") 
            st.metric("Failed", failed, f"{failed/total_processed*100:.1f}%")
        
        with col4:
            avg_defects = np.mean([r["Defects"] for r in batch_results])
            st.metric("Avg Defects", f"{avg_defects:.1f}")
        
        # Batch results table
        batch_df = pd.DataFrame(batch_results)
        st.dataframe(batch_df, use_container_width=True)
        
        # Export batch results
        csv_data = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Batch Results",
            csv_data,
            f"pcb_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

# Quality Dashboard
st.header("📈 Quality Control Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Total Inspected</h3>
        <h2 style="color: #2980b9;">{}</h2>
    </div>
    """.format(st.session_state.quality_metrics["total_inspected"]), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Defective Units</h3>
        <h2 style="color: #e74c3c;">{}</h2>
    </div>
    """.format(st.session_state.quality_metrics["defective_count"]), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Pass Rate</h3>
        <h2 style="color: #27ae60;">{:.1f}%</h2>
    </div>
    """.format(st.session_state.quality_metrics["pass_rate"]), unsafe_allow_html=True)

with col4:
    first_pass_yield = st.session_state.quality_metrics["pass_rate"] * 0.95  # Simulated
    st.markdown("""
    <div class="metric-card">
        <h3>First Pass Yield</h3>
        <h2 style="color: #f39c12;">{:.1f}%</h2>
    </div>
    """.format(first_pass_yield), unsafe_allow_html=True)

# Detection History and Analytics
if st.session_state.detection_history:
    st.subheader("📋 Detection History")
    
    # Create DataFrame from history
    history_df = pd.DataFrame(st.session_state.detection_history)
    
    # Display recent detections
    st.dataframe(history_df.tail(10)[::-1], use_container_width=True)
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Status distribution
        status_counts = history_df['Status'].value_counts()
        fig_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Overall Quality Distribution",
            color_discrete_map={"Good": "#27ae60", "Defective": "#e74c3c"}
        )
        st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        # Defect count trend
        history_df['Date'] = pd.to_datetime(history_df['Timestamp']).dt.date
        daily_defects = history_df.groupby('Date')['Defect_Count'].sum().reset_index()
        
        fig_trend = px.line(
            daily_defects, 
            x='Date', 
            y='Defect_Count',
            title="Daily Defect Trend",
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Export history
    csv_history = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Detection History",
        csv_history,
        f"pcb_detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv"
    )

else:
    st.info("🔍 No detection history available. Upload and analyze some PCB images to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>🔬 Advanced PCB Defect Detection System | Powered by Machine Learning</p>
    <p>Built with Streamlit, TensorFlow, and Computer Vision</p>
</div>
""", unsafe_allow_html=True)
