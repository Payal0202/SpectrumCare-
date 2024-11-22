# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:04:43 2024

@author: KA
"""

# -*- coding: utf-8 -*-
"""
Testing Script for ResNet Models
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Define constants
IMG_SIZES = [124, 248]
BATCH_SIZE = 32
DATASET_PATH = r'C:\Users\kagra\Downloads\archive\AutismDataset\split'
OUTPUT_PATH = r'C:\Users\kagra\Downloads\archive\AutismDataset\models'

# Function to calculate evaluation metrics
def evaluate_model(model, test_generator):
    y_true = test_generator.classes
    y_pred = np.argmax(model.predict(test_generator), axis=-1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Metrics calculations
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Error Rate: {error_rate:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1_score:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# Testing ResNet models
for size in IMG_SIZES:
    print(f"\nTesting ResNet model for image size {size} × {size}")

    # Load model
    model_path = os.path.join(OUTPUT_PATH, f"resnet_{size}x{size}.h5")
    if not os.path.exists(model_path):
        print(f"Model file not found for image size {size} × {size}. Skipping...")
        continue

    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Prepare test data generator
    test_dir = os.path.join(DATASET_PATH, f"{size}x{size}", "test")
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(size, size), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )

    # Evaluate model
    evaluate_model(model, test_generator)
