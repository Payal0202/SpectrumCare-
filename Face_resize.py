# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:11:10 2024

@author: KA
"""

import cv2
import os

# Define paths to the input directories for acoustic and non-acoustic classes
acoustic_dir = r'C:\Users\kagra\Downloads\archive\AutismDataset\consolidated\Autistic'  # Acoustic class images directory
non_acoustic_dir = r'C:\Users\kagra\Downloads\archive\AutismDataset\consolidated\Non_Autistic'  # Non-acoustic class images directory

# Define paths to output directories for each size and class
output_dir_124_acoustic = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\124x124\Autistic'  # Acoustic resized to 124x124
output_dir_248_acoustic = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\248x248\Autistic'  # Acoustic resized to 248x248

output_dir_124_non_acoustic = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\124x124\Non_Autistic'  # Non-acoustic resized to 124x124
output_dir_248_non_acoustic = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\248x248\Non_Autistic'  # Non-acoustic resized to 248x248

# Create output directories for both classes if they don't exist
os.makedirs(output_dir_124_acoustic, exist_ok=True)
os.makedirs(output_dir_248_acoustic, exist_ok=True)
os.makedirs(output_dir_124_non_acoustic, exist_ok=True)
os.makedirs(output_dir_248_non_acoustic, exist_ok=True)

# Define the two resize dimensions
resize_dims_124 = (124, 124)
resize_dims_248 = (248, 248)

# Function to resize images in a given directory
def resize_images(input_dir, output_dir_124, output_dir_248):
    # Loop through all the images in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Specify file formats
            # Load the image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            if img is not None:
                # Resize to 124x124
                resized_img_124 = cv2.resize(img, resize_dims_124, interpolation=cv2.INTER_AREA)
                # Save the resized image to the respective output folder
                cv2.imwrite(os.path.join(output_dir_124, filename), resized_img_124)

                # Resize to 248x248
                resized_img_248 = cv2.resize(img, resize_dims_248, interpolation=cv2.INTER_AREA)
                # Save the resized image to the respective output folder
                cv2.imwrite(os.path.join(output_dir_248, filename), resized_img_248)

            else:
                print(f"Failed to load {filename}")
    print(f"Resizing complete for images in {input_dir}")

# Resize images for both acoustic and non-acoustic classes
resize_images(acoustic_dir, output_dir_124_acoustic, output_dir_248_acoustic)
resize_images(non_acoustic_dir, output_dir_124_non_acoustic, output_dir_248_non_acoustic)