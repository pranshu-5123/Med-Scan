import os
import shutil
import random

def split_data(input_dir, output_dir, val_ratio=0.15, test_ratio=0.15):
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'validation')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through all classes
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class directories in train, validation, and test
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all image files
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        num_images = len(images)
        num_test = int(num_images * test_ratio)
        num_val = int(num_images * val_ratio)

        # Randomly select test and validation images
        test_images = random.sample(images, num_test)
        remaining_images = list(set(images) - set(test_images))
        val_images = random.sample(remaining_images, num_val)
        train_images = list(set(remaining_images) - set(val_images))

        # Copy images to respective directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(test_class_dir, img))

    print(f"Data split completed. Validation ratio: {val_ratio}, Test ratio: {test_ratio}")

# Usage
input_directory = "D:\Directory\Med Scan\BalancedData"
output_directory = "D:\Directory\Med Scan\split"
split_data(input_directory, output_directory, val_ratio=0.15, test_ratio=0.15)