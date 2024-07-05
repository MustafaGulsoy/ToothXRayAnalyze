import os
import cv2
import numpy as np

input_dir = './data/LabelData'
output_dir = './data/LabelTxt'


# Function to get unique colors in an image
def get_unique_colors(image):
    colors = image.reshape(-1, image.shape[-1])
    unique_colors = np.unique(colors, axis=0)
    return unique_colors


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    # Load the image in color mode
    mask = cv2.imread(image_path)
    H, W, _ = mask.shape

    # Get unique colors (classes)
    unique_colors = get_unique_colors(mask)

    # Process each color separately
    for class_id, color in enumerate(unique_colors):
        if np.all(color == [0, 0, 0]):
            continue  # Skip the background

        # Create a binary mask for the current color
        color_mask = cv2.inRange(mask, color, color)

        # Find contours
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:  # Filter small contours
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # Write the polygons to a file
        output_file = os.path.join(output_dir, '{}.txt'.format(j[:-4]))
        with open(output_file, 'a') as f:
            for polygon in polygons:
                f.write('{} '.format(class_id))
                f.write(' '.join(map(str, polygon)))
                f.write('\n')
