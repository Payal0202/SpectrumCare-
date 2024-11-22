# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:37:44 2024

@author: KA
"""

# Import necessary libraries
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define constants
IMG_SIZE = 124  # or 248 based on your model
MODEL_PATH = r'C:/Users/kagra/Downloads/archive/AutismDataset/models/resnet_124x124.h5'  # Path to your trained model
IMAGE_PATH = r'C:/Users/kagra/Downloads/archive/AutismDataset/split/124x124/test/Non_Autistic/0480.jpg'  # Path to the image to classify
CLASS_NAMES = ['Non_Autistic', 'Autistic']  # Update based on your class order in training

# Load the trained model
model = load_model(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")

# Load and preprocess the image
img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize the image (rescale)

# Perform prediction
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])  # Index of the highest probability
predicted_class = CLASS_NAMES[predicted_class_index]
confidence = predictions[0][predicted_class_index] * 100  # Confidence percentage

# Display the result
print(f"\nClassification Result:")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
