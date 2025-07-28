import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PCBDatasetGenerator:
    """Generate synthetic PCB dataset for training"""

    def __init__(self, num_samples=2000, img_size=(224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size

    def create_good_pcb(self):
        """Create a synthetic good PCB image"""
        img = np.zeros((*self.img_size, 3), dtype=np.uint8)
        img.fill(34)  # Dark background

        # Add copper traces (good patterns)
        for _ in range(np.random.randint(5, 12)):
            # Horizontal traces
            y = np.random.randint(20, self.img_size[0] - 20)
            thickness = np.random.randint(3, 8)
            cv2.line(img, (10, y), (self.img_size[1] - 10, y), (139, 69, 19), thickness)

            # Vertical traces
            x = np.random.randint(20, self.img_size[1] - 20)
            cv2.line(img, (x, 10), (x, self.img_size[0] - 10), (139, 69, 19), thickness)

        # Add components (rectangles)
        for _ in range(np.random.randint(3, 8)):
            x1 = np.random.randint(30, self.img_size[1] - 60)
            y1 = np.random.randint(30, self.img_size[0] - 60)
            x2 = x1 + np.random.randint(20, 40)
            y2 = y1 + np.random.randint(15, 30)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), -1)

        # Add via holes
        for _ in range(np.random.randint(8, 15)):
            center = (np.random.randint(30, self.img_size[1] - 30),
                     np.random.randint(30, self.img_size[0] - 30))
            cv2.circle(img, center, np.random.randint(2, 5), (200, 200, 200), -1)

        return img

    def create_defective_pcb(self):
        """Create a synthetic defective PCB image"""
        img = self.create_good_pcb()

        # Add random defects
        defect_type = np.random.choice(['spurious_copper', 'missing_hole', 'open_circuit', 'short'])

        if defect_type == 'spurious_copper':
            # Add unwanted copper patches
            for _ in range(np.random.randint(1, 4)):
                center = (np.random.randint(50, self.img_size[1] - 50),
                         np.random.randint(50, self.img_size[0] - 50))
                cv2.circle(img, center, np.random.randint(8, 20), (139, 69, 19), -1)

        elif defect_type == 'missing_hole':
            # Cover some holes with copper
            for _ in range(np.random.randint(2, 5)):
                center = (np.random.randint(30, self.img_size[1] - 30),
                         np.random.randint(30, self.img_size[0] - 30))
                cv2.circle(img, center, 8, (139, 69, 19), -1)

        elif defect_type == 'open_circuit':
            # Break some traces
            for _ in range(np.random.randint(1, 3)):
                x = np.random.randint(50, self.img_size[1] - 50)
                y = np.random.randint(50, self.img_size[0] - 50)
                cv2.rectangle(img, (x-5, y-5), (x+5, y+5), (34, 34, 34), -1)

        elif defect_type == 'short':
            # Add unwanted connections
            for _ in range(np.random.randint(1, 3)):
                pt1 = (np.random.randint(30, self.img_size[1] - 30),
                       np.random.randint(30, self.img_size[0] - 30))
                pt2 = (pt1[0] + np.random.randint(-30, 30),
                       pt1[1] + np.random.randint(-30, 30))
                cv2.line(img, pt1, pt2, (139, 69, 19), np.random.randint(2, 6))

        return img

    def generate_dataset(self):
        """Generate complete dataset"""
        images = []
        labels = []

        print("Generating synthetic PCB dataset...")

        # Generate good PCBs
        for i in range(self.num_samples // 2):
            if i % 100 == 0:
                print(f"Generated {i} good PCBs...")
            img = self.create_good_pcb()
            images.append(img)
            labels.append(0)  # 0 = Good

        # Generate defective PCBs
        for i in range(self.num_samples // 2):
            if i % 100 == 0:
                print(f"Generated {i} defective PCBs...")
            img = self.create_defective_pcb()
            images.append(img)
            labels.append(1)  # 1 = Defective

        return np.array(images), np.array(labels)

class PCBCNNModel:
    """CNN Model for PCB Classification"""

    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        """Build CNN architecture"""
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classifier
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint('best_pcb_model.h5', save_best_only=True)
        ]

        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        return history

def train_pcb_cnn_model():
    """Complete training pipeline for CNN model"""
    print("=== PCB CNN MODEL TRAINING ===")

    # Generate dataset
    generator = PCBDatasetGenerator(num_samples=4000)
    X, y = generator.generate_dataset()

    # Normalize images
    X = X.astype('float32') / 255.0

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # Build and train model
    pcb_model = PCBCNNModel()
    model = pcb_model.build_model()

    print("\nModel Architecture:")
    model.summary()

    print("\nTraining model...")
    history = pcb_model.train(X_train, y_train, X_val, y_val, epochs=30)

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    # Predictions and classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Good', 'Defective']))

    # Save final model
    model.save('pcb_classifier_model.h5')
    print("Model saved as 'pcb_classifier_model.h5'")

    return model, history

# ============================================================================
# 2. YOLO TRAINING PREPARATION
# ============================================================================

class YOLODatasetGenerator:
    """Generate YOLO format dataset for defect detection"""

    def __init__(self, num_samples=1000, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size
        self.defect_classes = {
            0: 'missing_hole',
            1: 'spurious_copper',
            2: 'open_circuit',
            3: 'short_circuit',
            4: 'bridge'
        }

    def create_pcb_with_annotations(self):
        """Create PCB image with bounding box annotations"""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img.fill(34)  # Dark background

        annotations = []

        # Add base PCB pattern
        for _ in range(np.random.randint(8, 15)):
            y = np.random.randint(50, self.img_size - 50)
            thickness = np.random.randint(3, 8)
            cv2.line(img, (20, y), (self.img_size - 20, y), (139, 69, 19), thickness)

        # Add components
        for _ in range(np.random.randint(5, 10)):
            x1 = np.random.randint(50, self.img_size - 100)
            y1 = np.random.randint(50, self.img_size - 100)
            x2 = x1 + np.random.randint(30, 50)
            y2 = y1 + np.random.randint(20, 40)
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), -1)

        # Add normal holes
        for _ in range(np.random.randint(10, 20)):
            center = (np.random.randint(50, self.img_size - 50),
                     np.random.randint(50, self.img_size - 50))
            cv2.circle(img, center, np.random.randint(3, 6), (200, 200, 200), -1)

        # Add defects with annotations
        num_defects = np.random.randint(1, 4)

        for _ in range(num_defects):
            defect_class = np.random.randint(0, 5)

            if defect_class == 0:  # missing_hole
                x = np.random.randint(100, self.img_size - 100)
                y = np.random.randint(100, self.img_size - 100)
                cv2.circle(img, (x, y), 15, (139, 69, 19), -1)  # Cover hole with copper

                # Annotation: center_x, center_y, width, height (normalized)
                center_x = x / self.img_size
                center_y = y / self.img_size
                width = 30 / self.img_size
                height = 30 / self.img_size
                annotations.append([defect_class, center_x, center_y, width, height])

            elif defect_class == 1:  # spurious_copper
                x = np.random.randint(100, self.img_size - 100)
                y = np.random.randint(100, self.img_size - 100)
                size = np.random.randint(15, 30)
                cv2.circle(img, (x, y), size, (139, 69, 19), -1)

                center_x = x / self.img_size
                center_y = y / self.img_size
                width = (size * 2) / self.img_size
                height = (size * 2) / self.img_size
                annotations.append([defect_class, center_x, center_y, width, height])

            elif defect_class == 2:  # open_circuit
                x = np.random.randint(100, self.img_size - 100)
                y = np.random.randint(100, self.img_size - 100)
                cv2.rectangle(img, (x-10, y-10), (x+10, y+10), (34, 34, 34), -1)

                center_x = x / self.img_size
                center_y = y / self.img_size
                width = 20 / self.img_size
                height = 20 / self.img_size
                annotations.append([defect_class, center_x, center_y, width, height])

            elif defect_class == 3:  # short_circuit
                x1 = np.random.randint(100, self.img_size - 200)
                y1 = np.random.randint(100, self.img_size - 100)
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(-30, 30)
                cv2.line(img, (x1, y1), (x2, y2), (139, 69, 19), 5)

                center_x = (x1 + x2) / 2 / self.img_size
                center_y = (y1 + y2) / 2 / self.img_size
                width = abs(x2 - x1) / self.img_size
                height = max(abs(y2 - y1), 10) / self.img_size
                annotations.append([defect_class, center_x, center_y, width, height])

            elif defect_class == 4:  # bridge
                x = np.random.randint(100, self.img_size - 100)
                y = np.random.randint(100, self.img_size - 100)
                cv2.rectangle(img, (x-5, y-20), (x+5, y+20), (139, 69, 19), -1)

                center_x = x / self.img_size
                center_y = y / self.img_size
                width = 10 / self.img_size
                height = 40 / self.img_size
                annotations.append([defect_class, center_x, center_y, width, height])

        return img, annotations

    def generate_yolo_dataset(self, output_dir='yolo_dataset'):
        """Generate YOLO format dataset"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/images', exist_ok=True)
        os.makedirs(f'{output_dir}/labels', exist_ok=True)

        print("Generating YOLO dataset...")

        for i in range(self.num_samples):
            if i % 100 == 0:
                print(f"Generated {i}/{self.num_samples} samples...")

            img, annotations = self.create_pcb_with_annotations()

            # Save image
            img_path = f'{output_dir}/images/pcb_{i:06d}.jpg'
            cv2.imwrite(img_path, img)

            # Save annotations
            label_path = f'{output_dir}/labels/pcb_{i:06d}.txt'
            with open(label_path, 'w') as f:
                for ann in annotations:
                    f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

        # Create data.yaml for YOLO
        data_yaml = {
            'train': f'{output_dir}/images',
            'val': f'{output_dir}/images',
            'nc': 5,
            'names': list(self.defect_classes.values())
        }

        with open(f'{output_dir}/data.yaml', 'w') as f:
            for key, value in data_yaml.items():
                if isinstance(value, list):
                    f.write(f"{key}: {value}\n")
                else:
                    f.write(f"{key}: {value}\n")

        print(f"YOLO dataset generated in '{output_dir}'")
        return output_dir

def train_yolo_model():
    """Train YOLO model for defect detection"""
    print("=== YOLO MODEL TRAINING ===")

    # Generate YOLO dataset
    yolo_gen = YOLODatasetGenerator(num_samples=2000)
    dataset_dir = yolo_gen.generate_yolo_dataset()

    # Note: For actual YOLO training, you would use:
    # from ultralytics import YOLO
    # model = YOLO('yolov8n.pt')
    # model.train(data=f'{dataset_dir}/data.yaml', epochs=100, imgsz=640)

    print("YOLO dataset prepared. To train actual YOLO model, run:")
    print("pip install ultralytics")
    print("from ultralytics import YOLO")
    print("model = YOLO('yolov8n.pt')")
    print(f"model.train(data='{dataset_dir}/data.yaml', epochs=100, imgsz=640)")

    return dataset_dir
