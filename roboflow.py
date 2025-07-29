
# pcb_defect_inference.py

import subprocess, sys

# ─── Auto‑install dependencies ─────────────────────────────────────────────────
def install(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", pkg], check=True)

for pkg in ("streamlit", "roboflow", "opencv-python", "Pillow", "numpy"):
    install(pkg)

# ─── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
from roboflow import Roboflow
import numpy as np
import cv2
from PIL import Image
import io

# ─── CONFIGURE your Roboflow access here ───────────────────────────────────────
RF_API_KEY = "rf_vyfNn4lyvxbQRGovmULZC9c439t2"  # ← your API key
WORKSPACE  = "YOUR_WORKSPACE_SLUG"              # ← e.g. "my-pcb-lab"
PROJECT    = "pcb-defect-aqobk"                 # ← your project slug
VERSION    = 1                                  # ← your version number

# ─── Initialize the model ───────────────────────────────────────────────────────
rf      = Roboflow(api_key=RF_API_KEY)
proj    = rf.workspace(WORKSPACE).project(PROJECT)
model   = proj.version(VERSION).model

# ─── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="PCB Defect Detector", layout="wide")
st.title("🔍 PCB Defect Detection")

st.markdown("""
Upload a PCB image and the pre‑trained model (`pcb-defect-aqobk/1`) will draw bounding boxes around detected defects.
""")

uploaded = st.file_uploader("Upload PCB image", type=["jpg","jpeg","png"])
if not uploaded:
    st.info("Please upload a PCB image to proceed.")
    st.stop()

# ─── Run inference ───────────────────────────────────────────────────────────────
img_bytes = uploaded.read()
response  = model.predict(img_bytes, confidence=40, overlap=20).json()

# ─── Draw boxes on the image ───────────────────────────────────────────────────
np_img    = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
annotated = np_img.copy()

for obj in response["predictions"]:
    x1 = int(obj["x"] - obj["width"]/2)
    y1 = int(obj["y"] - obj["height"]/2)
    x2 = x1 + int(obj["width"])
    y2 = y1 + int(obj["height"])
    label = f"{obj['class']} {obj['confidence']:.2f}"
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(annotated, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# ─── Display results ────────────────────────────────────────────────────────────
st.subheader("Input PCB")
st.image(np_img, use_column_width=True)

st.subheader("Detected Defects")
st.image(annotated, use_column_width=True)
st.markdown(f"**Defects found:** {len(response['predictions'])}")
