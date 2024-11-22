# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:44:01 2024

@author: KA
"""

import os
import random
from PIL import Image

# Function to check if a file is a valid image
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verifies the image file format
        return True
    except (IOError, SyntaxError):
        return False

# Function to flip some images in a given folder to add diversity while keeping the count same
def random_flip_images(input_dir, output_dir, flip_ratio=0.5, horizontal=True, vertical=False):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    # Randomly select a subset of the images to flip
    num_to_flip = int(len(image_files) * flip_ratio)
    images_to_flip = random.sample(image_files, num_to_flip)

    processed_count = 0
    flipped_count = 0

    # Loop through all images in the input directory
    for filename in image_files:
        img_path = os.path.join(input_dir, filename)

        # Check if the file is a valid image
        if is_valid_image(img_path):
            try:
                # Load image
                img = Image.open(img_path)
                processed_count += 1

                # If the image is selected for flipping, apply flips
                if filename in images_to_flip:
                    if horizontal:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    if vertical:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    flipped_count += 1

                # Save the image (flipped or not) with the same name
                output_path = os.path.join(output_dir, filename)
                img.save(output_path)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue  # Skip to the next image if there is an error
        else:
            print(f"Invalid image: {filename}")

    print(f"Flipping complete for images in {input_dir}")
    print(f"Total images processed: {processed_count}")
    print(f"Total images flipped: {flipped_count}")

# Paths to the four folders (update paths as per your setup)
folder1 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\124x124\Autistic'
folder2 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\124x124\Non_Autistic'
folder3 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\248x248\Autistic'
folder4 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output\248x248\Non_Autistic'

# Corresponding output folders for flipped images
output_folder1 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\124x124\Autistic'
output_folder2 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\124x124\Non_Autistic'
output_folder3 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\248x248\Autistic'
output_folder4 = r'C:\Users\kagra\Downloads\archive\AutismDataset\output_flip2\248x248\Non_Autistic'

# Apply flipping for all folders with 50% of the images flipped
random_flip_images(folder1, output_folder1, flip_ratio=0.5, horizontal=True, vertical=False)
random_flip_images(folder2, output_folder2, flip_ratio=0.5, horizontal=True, vertical=False)
random_flip_images(folder3, output_folder3, flip_ratio=0.5, horizontal=True, vertical=False)
random_flip_images(folder4, output_folder4, flip_ratio=0.5, horizontal=True, vertical=False)
