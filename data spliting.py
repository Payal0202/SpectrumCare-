# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:35:07 2024

@author: KA
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Function to split dataset into train, validate, and test
def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Splits the dataset into training, validation, and testing sets.

    Parameters:
    - input_dir: Directory containing the input images.
    - output_dir: Directory to store the split datasets.
    - train_ratio: Proportion of data to use for training.
    - val_ratio: Proportion of data to use for validation.

    Output:
    - Three subdirectories (train, val, test) with the split datasets.
    """
    # Ensure output directories exist
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of all image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Split into train, validation, and test sets
    train_files, temp_files = train_test_split(image_files, test_size=(1 - train_ratio), random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=(1 - val_ratio / (1 - train_ratio)), random_state=42)

    # Copy files to respective directories
    for filename in train_files:
        shutil.copy(os.path.join(input_dir, filename), os.path.join(train_dir, filename))
    for filename in val_files:
        shutil.copy(os.path.join(input_dir, filename), os.path.join(val_dir, filename))
    for filename in test_files:
        shutil.copy(os.path.join(input_dir, filename), os.path.join(test_dir, filename))

    print(f"Dataset split complete for {input_dir}:")
    print(f"  Training: {len(train_files)} images")
    print(f"  Validation: {len(val_files)} images")
    print(f"  Testing: {len(test_files)} images")

# Paths to the flipped folders
flipped_folder1 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\124x124\Autistic'
flipped_folder2 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\124x124\Non_Autistic'
flipped_folder3 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\248x248\Autistic'
flipped_folder4 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\248x248\Non_Autistic'

# Corresponding output folders for split datasets
split_folder1 = r'C:\Users\kagra\Downloads\archive\AutismDataset\split\124x124\Autistic'
split_folder2 = r'C:\Users\kagra\Downloads\archive\AutismDataset\split\124x124\Non_Autistic'
split_folder3 = r'C:\Users\kagra\Downloads\archive\AutismDataset\split\248x248\Autistic'
split_folder4 = r'C:\Users\kagra\Downloads\archive\AutismDataset\split\248x248\Non_Autistic'

# Apply splitting for all flipped folders
split_dataset(flipped_folder1, split_folder1, train_ratio=0.8, val_ratio=0.1)
split_dataset(flipped_folder2, split_folder2, train_ratio=0.8, val_ratio=0.1)
split_dataset(flipped_folder3, split_folder3, train_ratio=0.8, val_ratio=0.1)
split_dataset(flipped_folder4, split_folder4, train_ratio=0.8, val_ratio=0.1)
