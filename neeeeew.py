import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests



st.write("Streamlit is also great for more traditional ML use cases like computer vision or NLP. Here's an example of edge detection using OpenCV. 👁️") 

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open(requests.get("https://picsum.photos/200/120", stream=True).raw)

edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2 = st.tabs(["Detected edges", "Original"])
tab1.image(edges, use_column_width=True)
tab2.image(image, use_column_width=True)




import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# App title
st.set_page_config(page_title="PCB Defect Detection", layout="centered")
st.title("🔍 PCB Defect Detection using AI")
st.write("Upload a PCB image to detect potential defects using a deep learning model.")

# Define the class labels
class_names = ['No_Defect', 'Scratch', 'Crack', 'Hole',
               'Short_Circuit', 'Open_Circuit', 'Solder_Bridge', 'Missing_Component']

# Load model if available
MODEL_PATH = "trained_pcb_model.h5"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    model_loaded = True
    st.success("✅ Model loaded successfully!")
else:
    model_loaded = False
    st.warning("⚠️ Trained model not found. Using dummy predictions.")

# File uploader
uploaded_file = st.file_uploader("Upload PCB Image", type=["jpg", "jpeg", "png"])

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    if image.shape[-1] == 4:  # RGBA to RGB
        image = image[:, :, :3]
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Run prediction
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded PCB Image", use_column_width=True)

    img = Image.open(uploaded_file).convert("RGB")
    img_tensor = preprocess_image(img)

    if model_loaded:
        prediction = model.predict(img_tensor)[0]
    else:
        prediction = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]  # Dummy confidence

    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader("🔍 Detection Result")
    st.success(f"Predicted Defect: **{pred_class}**")
    st.info(f"Confidence: {confidence:.2f}")

    st.subheader("📊 Confidence per Class")
    for i, prob in enumerate(prediction):
        st.write(f"{class_names[i]}: {prob:.2f}")

