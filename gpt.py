# app.py

# ─── Auto‑install dependencies ─────────────────────────────────────────────────
import subprocess, sys
def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg])
for package in ("streamlit", "ultralytics", "opencv-python", "torch"):
    install(package)

# ─── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ─── Load or define your model ─────────────────────────────────────────────────
# Place your trained PCB‐defect weights in the project folder as 'pcb_yolov8.pt'
# Train your own on your dataset and save to that filename. Otherwise it'll
# default to the small COCO model (good for test/demo, but not PCB specific).
WEIGHTS = "pcb_yolov8.pt"
try:
    model = YOLO(WEIGHTS)
except Exception:
    st.warning(f"Could not load '{WEIGHTS}', falling back to yolov8n.pt demo model.")
    model = YOLO("yolov8n.pt")

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🛠️ PCB Defect Detection")
st.markdown(
    """
    Upload an image of a PCB and the model will highlight defect regions with bounding boxes.
    - **Note:** For best results, replace `pcb_yolov8.pt` with your own trained model.
    """
)

uploaded = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png"])
if uploaded:
    # Read image
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Input Image", use_column_width=True)

    # Run inference
    results = model.predict(img_np, imgsz=640)[0]

    # Draw boxes
    annotated = img_np.copy()
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = f"{model.names[int(cls)]} {conf:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    st.image(annotated, caption="Detected Defects", use_column_width=True)
else:
    st.info("Please upload a PCB image to start defect detection.")
