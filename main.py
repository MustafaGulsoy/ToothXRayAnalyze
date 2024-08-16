# # Load model directly
# from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
#
# processor = AutoImageProcessor.from_pretrained("Sankpan/vit-teeth-segment")
# model = SegformerForSemanticSegmentation.from_pretrained("Sankpan/vit-teeth-segment")


import os
import pydicom
from PIL import Image
import numpy as np

def convert_dcm_to_png(dcm_folder, png_folder):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    for filename in os.listdir(dcm_folder):
        if filename.endswith(".dcm"):
            filepath = os.path.join(dcm_folder, filename)
            ds = pydicom.dcmread(filepath)
            pixel_array = ds.pixel_array

            # Normalize the pixel array to 0-255
            image = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

            image = Image.fromarray(image)
            png_filename = f"{os.path.splitext(filename)[0]}.png"
            image.save(os.path.join(png_folder, png_filename))
            print(f"Converted {filename} to {png_filename}")


convert_dcm_to_png(dcm_folder, png_folder)



import os
import pydicom
from PIL import Image
import numpy as np

def convert_dcm_to_png(dcm_folder, png_folder):
    if not os.path.exists(png_folder):
        os.makedirs(png_folder)

    for filename in os.listdir(dcm_folder):
        if filename.endswith(".dcm"):
            filepath = os.path.join(dcm_folder, filename)
            try:
                ds = pydicom.dcmread(filepath, force=True)
                pixel_array = ds.pixel_array

                # Normalize the pixel array to 0-255
                image = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

                image = Image.fromarray(image)
                png_filename = f"{os.path.splitext(filename)[0]}.png"
                image.save(os.path.join(png_folder, png_filename))
                print(f"Converted {filename} to {png_filename}")
            except Exception as e:
                print(f"Could not convert {filename}: {e}")


convert_dcm_to_png(dcm_folder, png_folder)

