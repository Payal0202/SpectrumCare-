# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:50:48 2024

@author: KA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:00:00 2024

@author: KA
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# Define constants
IMG_SIZES = [124, 248]
BATCH_SIZE = 32
EPOCHS_STAGE_1 = 5
EPOCHS_STAGE_2 = 7
EPOCHS_STAGE_3 = 8
LEARNING_RATE_STAGE_1 = 1e-3
LEARNING_RATE_STAGE_2 = 1e-4
LEARNING_RATE_STAGE_3 = 1e-5

# Paths to data directories
DATASET_PATH = r'C:\Users\kagra\Downloads\archive\AutismDataset\split'
OUTPUT_PATH = r'C:\Users\kagra\Downloads\archive\AutismDataset\models'

# Function to create ResNet model
def create_resnet_model(input_shape, num_classes):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

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

# Function to train and evaluate the ResNet model
def train_resnet(image_size):
    print(f"Training ResNet with image size {image_size} × {image_size}")
    
    input_shape = (image_size, image_size, 3)
    train_dir = os.path.join(DATASET_PATH, f"{image_size}x{image_size}", "train")
    val_dir = os.path.join(DATASET_PATH, f"{image_size}x{image_size}", "val")
    test_dir = os.path.join(DATASET_PATH, f"{image_size}x{image_size}", "test")
    
    # Prepare data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(image_size, image_size), batch_size=BATCH_SIZE, class_mode="categorical"
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(image_size, image_size), batch_size=BATCH_SIZE, class_mode="categorical"
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(image_size, image_size), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
    )

    # Get number of classes
    num_classes = train_generator.num_classes

    # Create model
    model, base_model = create_resnet_model(input_shape, num_classes)

    # Stage 1: Initial training
    print("Stage 1: Initial training")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_STAGE_1), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_STAGE_1)

    # Stage 2: Freeze all layers except last two
    print("Stage 2: Selective training")
    for layer in base_model.layers:
        layer.trainable = False
    model.layers[-2].trainable = True
    model.layers[-1].trainable = True
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_STAGE_2), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_STAGE_2)

    # Stage 3: Unfreeze all layers for fine-tuning
    print("Stage 3: Fine-tuning")
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_STAGE_3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS_STAGE_3)

    # Save the trained model
    model.save(os.path.join(OUTPUT_PATH, f"resnet_{image_size}x{image_size}.h5"))
    print(f"Model saved for image size {image_size} × {image_size}")

    # Evaluate the model
    evaluate_model(model, test_generator)

# Train and evaluate models for each image size
for size in IMG_SIZES:
    train_resnet(size)
