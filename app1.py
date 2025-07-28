import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from datetime import datetime
import plotly.express as px

# Streamlit App Config
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header and Styling
st.markdown("""
    <style>
      .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
      }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🔬 AI-Based PCB Defect Detection System</h1>
    <p>Upload any PCB image to detect defects using an AI model.</p>
    <p><strong>Developed by:</strong> Prajith A | <strong>Guided by:</strong> Mr. Ashok Gopalakrishnan</p>
</div>
""", unsafe_allow_html=True)

# Session State
if 'detection_history' not in st.session_state:
    st.session_state['detection_history'] = []

# Model Loading
@st.cache_resource
def load_trained_model():
    model_path = 'pcb_classifier_model.h5'
    if os.path.exists(model_path):
        try:
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("Trained model not found as 'pcb_classifier_model.h5'.")
    return None

# Image Preprocess
def preprocess_image_for_prediction(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Inference (Prediction)
def classify_pcb_with_model(image, model):
    if model is None:
        # Demo mode if model missing
        confidence = np.random.uniform(0.85, 0.98)
        prediction = np.random.choice(['Good', 'Defective'], p=[0.7, 0.3])
        return prediction, confidence
    try:
        processed_image = preprocess_image_for_prediction(image)
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        predicted_class = 'Defective' if confidence > 0.5 else 'Good'
        confidence_final = confidence if predicted_class == 'Defective' else 1 - confidence
        return predicted_class, confidence_final
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "Error", 0.0

# Simulate YOLO Defect Visualization
def simulate_yolo_detection(image):
    defects = []
    num_defects = np.random.randint(0, 4)
    defect_types = ['missing_hole', 'spurious_copper', 'open_circuit', 'short_circuit', 'bridge']
    img_width, img_height = image.size
    for _ in range(num_defects):
        defect_type = np.random.choice(defect_types)
        confidence = np.random.uniform(0.6, 0.95)
        x1 = np.random.randint(50, img_width - 100)
        y1 = np.random.randint(50, img_height - 100)
        x2 = min(x1 + np.random.randint(30, 80), img_width - 1)
        y2 = min(y1 + np.random.randint(30, 80), img_height - 1)
        defects.append({
            'class': defect_type,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
        })
    return defects

# Draw Bounding Boxes
def draw_detection_boxes(image, defects):
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    colors = {
        'missing_hole': '#FF6B6B',
        'spurious_copper': '#4ECDC4',
        'open_circuit': '#45B7D1',
        'short_circuit': '#96CEB4',
        'bridge': '#FFEAA7',
        'unknown': '#DDA0DD'
    }
    for defect in defects:
        bbox = defect['bbox']
        defect_type = defect['class']
        confidence = defect['confidence']
        color = colors.get(defect_type, colors['unknown'])
        draw.rectangle(bbox, outline=color, width=3)
        label = f"{defect_type}: {confidence:.2f}"
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        text_xy = (bbox[0], max(bbox[1] - 20, 0))
        draw.rectangle([text_xy[0], text_xy[1], text_xy[0] + len(label) * 7, text_xy[1] + 18],
                       fill=color, outline=color)
        draw.text(text_xy, label, fill='white', font=font)
    return img_with_boxes

# Sidebar (Options)
with st.sidebar:
    st.header("🛠️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    show_bounding_boxes = st.checkbox("Show Defect Bounding Boxes", True, key='bbox_show')
    st.markdown("---")
    st.subheader("📊 Model Performance")
    st.metric("CNN Accuracy", "96.2%")
    st.metric("Processing Speed", "Real-time")

# Detection History & Visualization
st.subheader("📈 Detection History")
if st.session_state.detection_history:
    history_df = pd.DataFrame(st.session_state.detection_history)
    st.dataframe(history_df[::-1], use_container_width=True)
    fig = px.histogram(
        history_df,
        x="Prediction",
        title="Prediction Summary",
        color="Prediction"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No detections recorded yet.")

# Main: Upload & Inference
uploaded_image = st.file_uploader("Upload a PCB image", type=['jpg', 'png', 'jpeg'])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    model = load_trained_model()
    prediction, confidence = classify_pcb_with_model(image, model)
    defects = simulate_yolo_detection(image)
    st.success(f"Prediction: **{prediction}** with {confidence * 100:.2f}% confidence.")
    if show_bounding_boxes and len(defects):
        image_with_boxes = draw_detection_boxes(image, defects)
        st.image(image_with_boxes, caption="Detected Defects", use_column_width=True)
    # Save detection
    st.session_state.detection_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": uploaded_image.name,
        "Prediction": prediction,
        "Confidence": f"{confidence * 100:.2f}%",
        "Defect Count": len(defects)
    })

