import streamlit as st

# Install dependencies inside the script if running on new environments
try:
    import ultralytics
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', 'opencv-python', 'Pillow'])

from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("PCB Defect Detection with YOLOv8")
st.write("Upload a PCB image (JPG, PNG). The app detects and highlights defects using a YOLOv8 model.")

# Load PCB-specific YOLOv8 model (or use 'yolov8n.pt' / 'yolov8s.pt' by default)
@st.cache_resource
def load_model():
    # For demo: download YOLOv8s; replace with PCB-trained .pt file for best results
    return YOLO('yolov8s.pt')  # Use custom PCB model if available

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded PCB.', use_column_width=True)
    st.write("Detecting defects...")

    # Convert PIL to numpy
    img_np = np.array(image)
    # Run detection
    results = model(img_np)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    confs = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    # Define class names (default/coco or your PCB model classes)
    class_names = model.model.names if hasattr(model.model, "names") else ["defect"]

    # Draw boxes
    img_annot = img_np.copy()
    for box, conf, label in zip(boxes, confs, labels):
        x1, y1, x2, y2 = map(int, box)
        cls_name = class_names[int(label)]
        cv2.rectangle(img_annot, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_annot, f"{cls_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    st.image(img_annot, caption="Detection Results", use_column_width=True)
    st.success(f"Found {len(boxes)} objects (see image).")

    # Show details
    for i, (box, conf, label) in enumerate(zip(boxes, confs, labels), 1):
        st.write(f"{i}. Class: **{class_names[int(label)]}**, Confidence: {conf:.2f}, Box: {box.astype(int)}")
else:
    st.info("Upload an image to start.")

st.markdown("---")
st.caption("Demo: For best results, use a PCB-trained YOLOv8 model. This sample uses YOLOv8s pretrained weights as a placeholder.")

# To run: save as app.py and use: streamlit run app.py
