import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_pcb_model.h5')
import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)
import streamlit as st

st.title("PCB Defect Detection")
uploaded_file = st.file_uploader("Upload a PCB image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    if st.button("Detect Defect"):
        img = preprocess_image(uploaded_file)
        prediction = model.predict(img)[0][0]

        if prediction >= 0.5:
            st.error(f"❌ Defective PCB Detected! (Confidence: {prediction:.2f})")
        else:
            st.success(f"✅ Good PCB Detected! (Confidence: {1 - prediction:.2f})")


# Page setup
st.set_page_config(page_title="PCB Defect Detection", layout="centered")
st.title("🔍 PCB Defect Detection using Machine Learning")

# Initialize detection history
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# Load the trained ML model
@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("trained_pcb_model.h5")
    return model

# Simulate YOLO defect detection (dummy boxes)
def simulate_yolo_detection(image):
    width, height = image.size
    return [
        (int(0.1 * width), int(0.1 * height), int(0.3 * width), int(0.3 * height)),
        (int(0.5 * width), int(0.5 * height), int(0.7 * width), int(0.7 * height))
    ]

# Draw red boxes on image
def draw_detection_boxes(image, boxes):
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    for box in boxes:
        draw.rectangle(box, outline="red", width=3)
    return image_copy

# Preprocess image for classification
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict using ML model
def classify_pcb_with_model(image, model):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    predicted_class = "Problematic" if prediction[0][0] > 0.5 else "Good"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return predicted_class, confidence

# Sidebar toggle
st.sidebar.header("⚙️ Options")
show_boxes = st.sidebar.checkbox("Show simulated bounding boxes", value=True)

# Upload image
uploaded_image = st.file_uploader("Upload a PCB image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_trained_model()
    prediction, confidence = classify_pcb_with_model(image, model)

    defects = simulate_yolo_detection(image) if show_boxes else []

    st.success(f"Prediction: **{prediction}** ({confidence*100:.2f}% confidence)")

    if show_boxes and defects:
        boxed_image = draw_detection_boxes(image, defects)
        st.image(boxed_image, caption="Detected Defects (Simulated)", use_column_width=True)

    # Save to history
    st.session_state.detection_history.append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": uploaded_image.name,
        "Prediction": prediction,
        "Confidence": f"{confidence*100:.2f}%",
        "Defect Count": len(defects)
    })

# Detection history table + chart
st.subheader("📈 Detection History")

if st.session_state.detection_history:
    df = pd.DataFrame(st.session_state.detection_history)
    st.dataframe(df[::-1], use_container_width=True)

    fig = px.histogram(df, x="Prediction", color="Prediction", title="Prediction Summary")
    st.plotly_chart(fig, use_container_width=True)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Detection History", csv, "pcb_detection_history.csv", "text/csv")
else:
    st.info("No detection yet. Upload an image to start.")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import os
from tqdm import tqdm

# Set paths and parameters
IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 32
MODEL_NAME = "trained_pcb_model.h5"

# Simulate dataset for PCB (Good vs Defective)
def create_dummy_pcb_dataset(n_samples=500):
    images = []
    labels = []
    for i in range(n_samples):
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        label = np.random.choice([0, 1])  # 0: Good, 1: Defective
        if label == 1:
            cv2.line(img, (10, 10), (100, 100), (255, 255, 255), 2)
            cv2.rectangle(img, (60, 60), (100, 100), (255, 255, 255), -1)
        else:
            cv2.circle(img, (64, 64), 20, (255, 255, 255), -1)
        images.append(img)
        labels.append(label)
    images, labels = shuffle(images, labels)
    return np.array(images), np.array(labels)

# Load dataset
X, y = create_dummy_pcb_dataset(1000)
X = X / 255.0  # Normalize

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build CNN model
def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss'),
    ModelCheckpoint(MODEL_NAME, save_best_only=True)
]

# Train model
model = build_model()
model.fit(X_train, y_train, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

# Save final model
model.save(MODEL_NAME)
print(f"Model saved as {MODEL_NAME}")
