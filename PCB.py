# app.py

# ─── Auto‑install dependencies ─────────────────────────────────────────────────
import subprocess, sys

def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)

# Install before any other imports
for package in ("streamlit", "ultralytics", "torch", "opencv-python"):
    install(package)

# ─── Now safe to import everything ───────────────────────────────────────────────
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ─── Load your model ─────────────────────────────────────────────────────────────
try:
    model = YOLO("pcb_yolov8.pt")
except Exception:
    st.warning("Falling back to the demo model.")
    model = YOLO("yolov8n.pt")

# ─── Streamlit UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🛠️ PCB Defect Detection")

uploaded = st.file_uploader("Upload PCB image", type=["jpg", "png", "jpeg"])
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    st.image(img, caption="Input", use_column_width=True)
    results = model.predict(img, imgsz=640)[0]
    annotated = img.copy()
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    st.image(annotated, caption="Detected Defects", use_column_width=True)
else:
    st.info("Please upload a PCB image to start defect detection.")

