import os
import random
import shutil
import numpy as np
from PIL import Image
import albumentations as A


def augment_images(input_directory, output_directory, num_augmentations=5):
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory created: {output_directory}")

    # Define the augmentation pipeline
    transform = A.Compose([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None, p=0.5),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3)
    ])

    labels = {}
    max_images = 0

    # Collect image paths and determine the maximum number of images in any class
    for class_name in os.listdir(input_directory):
        class_path = os.path.join(input_directory, class_name)
        if os.path.isdir(class_path):
            image_files = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                labels[class_name] = [os.path.join(class_path, img) for img in image_files]
                max_images = max(max_images, len(labels[class_name]))
                print(f"Found {len(image_files)} images for class '{class_name}'")
            else:
                print(f"Warning: No images found for class '{class_name}'. Skipping.")
        else:
            print(f"Warning: '{class_name}' is not a directory. Skipping.")

    # Create a directory for each class in the balanced dataset
    for class_name in labels.keys():
        os.makedirs(os.path.join(output_directory, class_name), exist_ok=True)

    # Balance the dataset by augmenting images for underrepresented classes
    for class_name, image_paths in labels.items():
        class_dir = os.path.join(output_directory, class_name)

        # Augment images if the class has fewer images
        while len(image_paths) < max_images:
            image_path = random.choice(image_paths)
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            augmented = transform(image=image)['image']
            augmented_image = Image.fromarray(augmented)
            augmented_image_path = os.path.join(class_dir,
                                                f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{len(image_paths)}.png")
            augmented_image.save(augmented_image_path)
            image_paths.append(augmented_image_path)
            print(f"Augmented image saved to: {augmented_image_path}")

        # Copy original images to the new directory
        for img_path in image_paths:
            if not img_path.endswith('.png') or '_aug_' not in os.path.basename(img_path):
                destination_path = os.path.join(class_dir, os.path.basename(img_path))
                if not os.path.exists(destination_path):
                    shutil.copy(img_path, destination_path)
                    print(f"Original image copied to: {destination_path}")


# Usage
input_directory = r'D:\Directory\Med Scan\Segmented Medicinal Leaf Images'  # Use raw string literals for paths
output_directory = r'D:\Directory\Med Scan\BalancedData'  # Use raw string literals for paths
augment_images(input_directory, output_directory, num_augmentations=5)

