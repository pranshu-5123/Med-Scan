import os
import csv
from PIL import Image

# Define the path to the dataset and output file
dataset_path = 'D:\Directory\Med Scan\Segmented Medicinal Leaf Images'
output_file = 'labels.csv'

# Initialize a list to hold the labels
labels = []

# Loop through each class directory
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_path):
        # Loop through each image in the class directory
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                labels.append([class_name, img_path])

# Write the labels to a CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Image Path'])
    writer.writerows(labels)

print(f"Labels have been saved to {output_file}")
