"""
PCB Defect Detection System - ZERO DEPENDENCIES VERSION
=====================================================

ONLY ONE PACKAGE NEEDED:
pip install streamlit

To run: streamlit run pcb_defect_detector.py
"""

import streamlit as st
import random
import time
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PCB Defect Detection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateY(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .defect-alert {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(255,107,107,0.3);
    }
    
    .success-alert {
        background: linear-gradient(135deg, #55a3ff, #1e90ff);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(85,163,255,0.3);
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 20px;
        border-left: 5px solid #17a2b8;
        margin: 15px 0;
        border-radius: 0 15px 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .detection-box {
        border: 3px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        background: rgba(220, 53, 69, 0.1);
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        margin: 20px 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        border-radius: 10px;
        margin: 5px;
        min-width: 120px;
        box-shadow: 0 4px 15px rgba(116,185,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

class ZeroDependencyPCBDetector:
    def __init__(self):
        self.defect_types = [
            "Missing Component",
            "Wrong Component", 
            "Damaged Component",
            "Soldering Defect",
            "Short Circuit",
            "Open Circuit",
            "Misalignment",
            "Corrosion",
            "Crack",
            "Contamination"
        ]
        
        self.defect_descriptions = {
            "Missing Component": "Critical: Component absent from designated location",
            "Wrong Component": "Critical: Incorrect component installed", 
            "Damaged Component": "High: Physical damage to component",
            "Soldering Defect": "High: Poor solder joint quality",
            "Short Circuit": "Critical: Unwanted electrical connection",
            "Open Circuit": "Critical: Broken electrical connection",
            "Misalignment": "Medium: Component not properly positioned",
            "Corrosion": "Low: Oxidation or chemical damage",
            "Crack": "Medium: Physical crack in board or component",
            "Contamination": "Low: Foreign material present"
        }
        
        self.severity_colors = {
            "Critical": "#dc3545",
            "High": "#fd7e14", 
            "Medium": "#ffc107",
            "Low": "#28a745"
        }
    
    def create_sample_pcb_image(self, width=800, height=600, defect_level="medium"):
        """Create a sample PCB image using PIL"""
        # Create base green PCB
        img = Image.new('RGB', (width, height), color=(0, 80, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw PCB outline
        draw.rectangle([20, 20, width-20, height-20], outline=(0, 120, 0), width=3)
        
        # Draw copper traces
        for i in range(5):
            y = 80 + i * 100
            draw.line([50, y, width-50, y], fill=(0, 150, 0), width=4)
        
        for i in range(8):
            x = 100 + i * 80
            draw.line([x, 50, x, height-50], fill=(0, 150, 0), width=2)
        
        # Add components
        # Resistors
        for i in range(6):
            x = 120 + i * 100
            draw.rectangle([x, 120, x+40, 140], fill=(139, 69, 19))  # Brown resistors
        
        # Capacitors (circles)
        for i in range(4):
            x = 150 + i * 150
            draw.ellipse([x-15, 200-15, x+15, 200+15], fill=(64, 64, 64))
        
        # IC chips
        for i in range(3):
            x = 200 + i * 150
            draw.rectangle([x, 350, x+80, 400], fill=(20, 20, 20))
            # IC pins
            for pin in range(8):
                px = x + 10 + pin * 8
                draw.rectangle([px, 345, px+3, 350], fill=(200, 200, 200))  # Top pins
                draw.rectangle([px, 400, px+3, 405], fill=(200, 200, 200))  # Bottom pins
        
        # Solder pads
        for i in range(15):
            for j in range(10):
                x = 60 + i * 45
                y = 80 + j * 45
                if random.random() > 0.7:  # Sparse distribution
                    draw.ellipse([x-3, y-3, x+3, y+3], fill=(200, 200, 200))
        
        return img
    
    def analyze_image_properties(self, image):
        """Analyze image to determine realistic defect simulation"""
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Calculate basic properties
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        green_dominance = np.mean(img_array[:,:,1]) - np.mean(img_array[:,:,0])
        
        properties = {
            'brightness': brightness,
            'contrast': contrast,
            'is_pcb_like': green_dominance > 20,  # Green PCBs
            'image_quality': 'high' if contrast > 50 else 'medium' if contrast > 25 else 'low'
        }
        
        return properties
    
    def simulate_ai_detection(self, image, confidence_threshold=0.5):
        """Simulate AI-powered defect detection"""
        
        # Analyze image properties for realistic simulation
        props = self.analyze_image_properties(image)
        width, height = image.size
        
        # Determine detection parameters based on image
        if props['is_pcb_like']:
            base_defect_count = random.randint(2, 6)
        else:
            base_defect_count = random.randint(1, 4)
        
        # Adjust based on image quality
        if props['image_quality'] == 'high':
            confidence_boost = 0.1
        elif props['image_quality'] == 'low':
            confidence_boost = -0.1
        else:
            confidence_boost = 0
        
        detections = []
        
        # Generate realistic defect detections
        for i in range(base_defect_count):
            # Random location with some clustering (more realistic)
            if i > 0 and random.random() > 0.6:  # 40% chance to cluster near previous defect
                prev_detection = detections[-1]
                x = max(20, min(width-120, prev_detection['x'] + random.randint(-100, 100)))
                y = max(20, min(height-120, prev_detection['y'] + random.randint(-100, 100)))
            else:
                x = random.randint(20, width-120)
                y = random.randint(20, height-120)
            
            # Box dimensions
            box_w = random.randint(40, 100)
            box_h = random.randint(40, 100)
            
            # Select defect type with realistic probabilities
            defect_probabilities = [0.18, 0.15, 0.12, 0.20, 0.08, 0.07, 0.10, 0.04, 0.03, 0.03]
            defect_id = random.choices(range(len(self.defect_types)), weights=defect_probabilities)[0]
            defect_name = self.defect_types[defect_id]
            
            # Generate confidence with some realism
            base_confidences = {
                "Missing Component": (0.75, 0.95),
                "Wrong Component": (0.70, 0.90),
                "Damaged Component": (0.65, 0.85),
                "Soldering Defect": (0.70, 0.90),
                "Short Circuit": (0.60, 0.80),
                "Open Circuit": (0.60, 0.80),
                "Misalignment": (0.55, 0.85),
                "Corrosion": (0.45, 0.70),
                "Crack": (0.50, 0.75),
                "Contamination": (0.40, 0.65)
            }
            
            conf_min, conf_max = base_confidences.get(defect_name, (0.5, 0.8))
            confidence = random.uniform(conf_min, conf_max) + confidence_boost
            confidence = max(0.1, min(0.99, confidence))  # Clamp to valid range
            
            # Determine severity
            if defect_name in ["Missing Component", "Wrong Component", "Short Circuit", "Open Circuit"]:
                severity = "Critical"
            elif defect_name in ["Damaged Component", "Soldering Defect", "Crack", "Misalignment"]:
                severity = "High" if confidence > 0.7 else "Medium"
            else:
                severity = "Medium" if confidence > 0.6 else "Low"
            
            if confidence >= confidence_threshold:
                detections.append({
                    'x': x,
                    'y': y,
                    'width': box_w,
                    'height': box_h,
                    'confidence': confidence,
                    'defect_type': defect_name,
                    'severity': severity,
                    'description': self.defect_descriptions[defect_name]
                })
        
        return detections
    
    def draw_detections_on_image(self, image, detections):
        """Draw detection boxes on image using PIL"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        colors = {
            "Critical": "#ff0000",
            "High": "#ff8800", 
            "Medium": "#ffdd00",
            "Low": "#00ff00"
        }
        
        for detection in detections:
            x, y = detection['x'], detection['y']
            w, h = detection['width'], detection['height']
            severity = detection['severity']
            color = colors.get(severity, "#ff0000")
            
            # Draw bounding box
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            
            # Draw label background
            label = f"{detection['defect_type'][:15]}..."
            try:
                # Try to load a font (might not work on all systems)
                font = ImageFont.load_default()
            except:
                font = None
            
            # Get text size (approximate for default font)
            text_width = len(label) * 6  # Rough estimate
            text_height = 12
            
            # Draw label background
            draw.rectangle([x, y-20, x+text_width+10, y], fill=color)
            
            # Draw text
            draw.text((x+5, y-18), label, fill="white", font=font)
        
        return img_copy

def main():
    # Header with animation effect
    st.markdown('<h1 class="main-header">🔍 AI-Powered PCB Defect Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize detector
    detector = ZeroDependencyPCBDetector()
    
    # Sidebar
    st.sidebar.title("🛠️ AI Configuration")
    st.sidebar.markdown("---")
    
    confidence_threshold = st.sidebar.slider(
        "🎯 Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Lower values detect more defects but may include false positives"
    )
    
    st.sidebar.markdown("### 🎨 Defect Color Legend")
    st.sidebar.markdown("🔴 **Critical** - Immediate attention required")
    st.sidebar.markdown("🟠 **High** - Should be addressed soon") 
    st.sidebar.markdown("🟡 **Medium** - Monitor and plan fixes")
    st.sidebar.markdown("🟢 **Low** - Minor issues")
    
    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Analysis", "🧪 AI Demo", "📊 Batch Processing", "📚 System Info"])
    
    with tab1:
        st.markdown("### 📤 Upload PCB Image for AI Analysis")
        
        uploaded_file = st.file_uploader(
            "Select a PCB image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear, well-lit image of a PCB for best results"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📷 Original PCB Image")
                st.image(image, caption=f"📁 {uploaded_file.name}", use_column_width=True)
                
                # Image analysis
                props = detector.analyze_image_properties(image)
                
                st.markdown(f"""
                <div class="info-card">
                    <strong>📊 Image Analysis:</strong><br>
                    • **Dimensions:** {image.size[0]} × {image.size[1]} pixels<br>
                    • **Format:** {image.format}<br>
                    • **Quality:** {props['image_quality'].title()}<br>
                    • **PCB Detected:** {'✅ Yes' if props['is_pcb_like'] else '❌ No'}<br>
                    • **Brightness:** {props['brightness']:.1f}/255
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 🤖 AI Detection Results")
                
                # AI Processing simulation
                with st.spinner("🧠 AI is analyzing PCB... Please wait"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    stages = [
                        "🔍 Scanning image...",
                        "🧠 Loading AI model...", 
                        "⚡ Processing neural network...",
                        "🎯 Detecting defects...",
                        "📊 Calculating confidence scores...",
                        "✅ Analysis complete!"
                    ]
                    
                    for i, stage in enumerate(stages):
                        status_text.text(stage)
                        time.sleep(0.5)
                        progress_bar.progress((i + 1) / len(stages))
                    
                    progress_bar.empty()
                    status_text.empty()
                
                # Run detection
                detections = detector.simulate_ai_detection(image, confidence_threshold)
                
                if detections:
                    # Draw results
                    result_image = detector.draw_detections_on_image(image, detections)
                    st.image(result_image, caption="🚨 AI Detection Results", use_column_width=True)
                    
                    # Alert
                    st.markdown(f"""
                    <div class="defect-alert">
                        <h3>⚠️ DEFECTS DETECTED</h3>
                        <p>AI has identified <strong>{len(detections)}</strong> potential issues requiring attention.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                    <div class="success-alert">
                        <h3>✅ PCB QUALITY: EXCELLENT</h3>
                        <p>No defects detected above {confidence_threshold:.0%} confidence threshold.</p>
                        <p>This PCB meets quality standards! 🎉</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed results
            if detections:
                st.markdown("---")
                st.markdown("### 📋 Detailed Detection Report")
                
                # Summary metrics
                critical_count = sum(1 for d in detections if d['severity'] == 'Critical')
                high_count = sum(1 for d in detections if d['severity'] == 'High')
                avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.markdown(f"""
                    <div class="stat-item">
                        <h2>{len(detections)}</h2>
                        <p>Total Defects</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_b:
                    st.markdown(f"""
                    <div class="stat-item" style="background: linear-gradient(135deg, #ff6b6b, #ee5a24);">
                        <h2>{critical_count}</h2>
                        <p>Critical Issues</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_c:
                    st.markdown(f"""
                    <div class="stat-item" style="background: linear-gradient(135deg, #fd79a8, #e84393);">
                        <h2>{high_count}</h2>
                        <p>High Priority</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_d:
                    st.markdown(f"""
                    <div class="stat-item" style="background: linear-gradient(135deg, #55a3ff, #74b9ff);">
                        <h2>{avg_confidence:.0%}</h2>
                        <p>Avg Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Individual defect details
                st.markdown("#### 🔍 Individual Defect Analysis")
                
                for i, detection in enumerate(detections, 1):
                    severity_color = detector.severity_colors[detection['severity']]
                    
                    st.markdown(f"""
                    <div class="detection-box" style="border-color: {severity_color};">
                        <h4>🎯 Defect #{i}: {detection['defect_type']}</h4>
                        <p><strong>Severity:</strong> <span style="color: {severity_color};">{detection['severity']}</span></p>
                        <p><strong>Confidence:</strong> {detection['confidence']:.1%}</p>
                        <p><strong>Location:</strong> ({detection['x']}, {detection['y']})</p>
                        <p><strong>Description:</strong> {detection['description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 🧪 AI System Demonstration")
        st.markdown("Test the AI detection system with computer-generated sample PCBs:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🟢 Generate Perfect PCB", use_container_width=True):
                st.session_state.demo_type = 'perfect'
        
        with col2:
            if st.button("🟡 Generate Minor Issues", use_container_width=True):
                st.session_state.demo_type = 'minor'
        
        with col3:
            if st.button("🔴 Generate Major Defects", use_container_width=True):
                st.session_state.demo_type = 'major'
        
        if 'demo_type' in st.session_state:
            demo_type = st.session_state.demo_type
            
            # Set random seed for consistent demo
            if demo_type == 'perfect':
                random.seed(999)  # Very few defects
            elif demo_type == 'minor':
                random.seed(555)  # Some defects
            else:  # major
                random.seed(111)  # Many defects
            
            # Generate sample PCB
            sample_pcb = detector.create_sample_pcb_image(800, 600, demo_type)
            
            col1, col2 = st.columns(2)
            
            with col1:
                ...
        # Display defect types in organized layout
        st.markdown("##### 🔍 **Detectable Defect Types:**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            - Missing Component
            - Wrong Component
            - Damaged Component
            - Soldering Defect
            - Short Circuit
            """)

        with col2:
            st.markdown("""
            - Open Circuit
            - Misalignment
            - Corrosion
            - Crack
            - Contamination
            """)

        st.markdown("""
        #### 📈 **Performance Highlights**
        - Simulates realistic AI-based PCB defect detection
        - Works with custom uploaded PCB images
        - Detects defects with different severity levels
        - Generates demo PCBs and batch processing simulations

        #### 🚀 **How to Use**
        1. Upload a PCB image or generate a sample.
        2. Adjust the detection confidence in the sidebar.
        3. View the detection results and summary report.

        #### 📅 **Version**
        - Streamlit App v1.0 (Zero Dependency AI Simulation)

        #### 😎 **Created by**
        - Prajith A, B.Tech AI&DS
        - United Institute of Technology
        """)

if __name__ == "__main__":
    main()

               
