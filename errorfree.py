import os
import shutil
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


class PCBDatasetPreparator:
    def __init__(self):
        self.class_names = [
            'No_Defect', 'Scratch', 'Crack', 'Hole',
            'Short_Circuit', 'Open_Circuit', 'Solder_Bridge', 'Missing_Component'
        ]
        self.target_size = (224, 224)

    def create_dataset_structure(self, base_dir='pcb_dataset'):
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'val')
        test_dir = os.path.join(base_dir, 'test')

        for main_dir in [train_dir, val_dir, test_dir]:
            for class_name in self.class_names:
                os.makedirs(os.path.join(main_dir, class_name), exist_ok=True)

        info = {
            'dataset_name': 'PCB Defect Detection Dataset',
            'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'classes': self.class_names,
            'num_classes': len(self.class_names),
            'image_size': self.target_size,
            'splits': {'train': '70%', 'validation': '15%', 'test': '15%'}
        }
        with open(os.path.join(base_dir, 'dataset_info.json'), 'w') as f:
            json.dump(info, f, indent=2)

        with open(os.path.join(base_dir, 'README.md'), 'w') as f:
            f.write("# PCB Defect Detection Dataset\n\nDirectory structure and guidelines...")

        return base_dir

    def validate_images(self, directory):
        issues_found = []
        fixed_count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(root, file)
                    try:
                        img = cv2.imread(file_path)
                        if img is None:
                            issues_found.append(f"Cannot read: {file_path}")
                            continue
                        h, w = img.shape[:2]
                        if h < 50 or w < 50:
                            issues_found.append(f"Too small: {file_path}")
                        if np.sum(img) == 0:
                            issues_found.append(f"Black/corrupted: {file_path}")
                        if h != self.target_size[0] or w != self.target_size[1]:
                            img_resized = cv2.resize(img, self.target_size)
                            cv2.imwrite(file_path, img_resized)
                            fixed_count += 1
                    except Exception as e:
                        issues_found.append(f"Error: {file_path}: {str(e)}")

        return fixed_count, issues_found

    def analyze_dataset(self, dataset_dir):
        analysis = {
            'total_images': 0,
            'class_distribution': {},
            'split_distribution': {'train': 0, 'val': 0, 'test': 0}
        }

        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(dataset_dir, split)
            if os.path.exists(split_dir):
                for class_name in self.class_names:
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.exists(class_dir):
                        count = len([f for f in os.listdir(class_dir)
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
                        if class_name not in analysis['class_distribution']:
                            analysis['class_distribution'][class_name] = {'train': 0, 'val': 0, 'test': 0}
                        analysis['class_distribution'][class_name][split] = count
                        analysis['split_distribution'][split] += count
                        analysis['total_images'] += count

        self._create_analysis_plots(analysis, dataset_dir)
        with open(os.path.join(dataset_dir, 'dataset_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)

        return analysis

    def _create_analysis_plots(self, analysis, output_dir):
        plt.figure(figsize=(15, 10))

        # Subplot 1
        plt.subplot(2, 2, 1)
        class_totals = {name: sum(data.values()) for name, data in analysis['class_distribution'].items()}
        plt.bar(class_totals.keys(), class_totals.values())
        plt.xticks(rotation=45)
        plt.title("Total Images per Class")
        plt.ylabel("Count")

        # Subplot 2
        plt.subplot(2, 2, 2)
        plt.pie(analysis['split_distribution'].values(), labels=analysis['split_distribution'].keys(), autopct='%1.1f%%')
        plt.title("Dataset Split")

        # Subplot 3
        plt.subplot(2, 2, 3)
        classes = list(analysis['class_distribution'].keys())
        train = [analysis['class_distribution'][c]['train'] for c in classes]
        val = [analysis['class_distribution'][c]['val'] for c in classes]
        test = [analysis['class_distribution'][c]['test'] for c in classes]
        plt.bar(classes, train, label='Train')
        plt.bar(classes, val, bottom=train, label='Val')
        plt.bar(classes, test, bottom=[i+j for i,j in zip(train,val)], label='Test')
        plt.xticks(rotation=45)
        plt.legend()
        plt.title("Images per Class by Split")

        # Subplot 4
        plt.subplot(2, 2, 4)
        total_counts = list(class_totals.values())
        plt.boxplot(total_counts)
        plt.title('Class Balance')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dataset_analysis.png'))
        plt.close()

    def balance_dataset(self, dataset_dir, target_samples_per_class=300):
        for split in ['train']:
            split_dir = os.path.join(dataset_dir, split)
            for class_name in self.class_names:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    images = [f for f in os.listdir(class_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    current_count = len(images)
                    if current_count < target_samples_per_class:
                        needed = target_samples_per_class - current_count
                        for i in range(needed):
                            src = images[i % current_count]
                            src_path = os.path.join(class_dir, src)
                            name, ext = os.path.splitext(src)
                            new_name = f"{name}_aug_{i:03d}{ext}"
                            new_path = os.path.join(class_dir, new_name)
                            img = cv2.imread(src_path)
                            if i % 4 == 0:
                                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                            elif i % 4 == 1:
                                img = cv2.flip(img, 1)
                            elif i % 4 == 2:
                                img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
                            cv2.imwrite(new_path, img)

    def create_data_splits(self, source_dir, target_dir, train_ratio=0.7, val_ratio=0.15):
        random.seed(42)
        self.create_dataset_structure(target_dir)
        for class_name in self.class_names:
            src_dir = os.path.join(source_dir, class_name)
            if os.path.exists(src_dir):
                images = [f for f in os.listdir(src_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                random.shuffle(images)
                total = len(images)
                train = images[:int(total * train_ratio)]
                val = images[int(total * train_ratio):int(total * (train_ratio + val_ratio))]
                test = images[int(total * (train_ratio + val_ratio)):]
                for img_set, split in [(train, 'train'), (val, 'val'), (test, 'test')]:
                    for img in img_set:
                        shutil.copy2(os.path.join(src_dir, img),
                                     os.path.join(target_dir, split, class_name, img))


# === STREAMLIT FRONTEND ===
st.title("🔧 PCB Dataset Preparation Tool")

preparator = PCBDatasetPreparator()

option = st.sidebar.selectbox("Select Operation", [
    "1. Create Dataset Structure",
    "2. Validate Images",
    "3. Analyze Dataset",
    "4. Balance Dataset",
    "5. Create Data Splits"
])

if option == "1. Create Dataset Structure":
    path = st.text_input("Enter target base directory", "pcb_dataset")
    if st.button("Create"):
        output = preparator.create_dataset_structure(path)
        st.success(f"Structure created at: {output}")

elif option == "2. Validate Images":
    folder = st.text_input("Enter dataset directory to validate", "pcb_dataset")
    if st.button("Validate"):
        fixed, issues = preparator.validate_images(folder)
        st.write(f"✅ Fixed: {fixed}")
        st.write(f"⚠️ Issues Found: {len(issues)}")
        if issues:
            st.write("Sample Issues:")
            st.write(issues[:5])

elif option == "3. Analyze Dataset":
    folder = st.text_input("Enter dataset directory to analyze", "pcb_dataset")
    if st.button("Analyze"):
        stats = preparator.analyze_dataset(folder)
        st.json(stats)
        img_path = os.path.join(folder, "dataset_analysis.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Dataset Analysis")

elif option == "4. Balance Dataset":
    folder = st.text_input("Enter dataset directory", "pcb_dataset")
    samples = st.slider("Target samples per class", 100, 1000, 300)
    if st.button("Balance"):
        preparator.balance_dataset(folder, samples)
        st.success("Dataset balanced successfully!")

elif option == "5. Create Data Splits":
    source = st.text_input("Enter source folder (all images sorted by class)", "unsplit_images")
    target = st.text_input("Enter target dataset folder", "pcb_dataset")
    if st.button("Split"):
        preparator.create_data_splits(source, target)
        st.success("Data split into train/val/test successfully!")
