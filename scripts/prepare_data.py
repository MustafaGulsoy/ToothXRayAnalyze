import os
from src.data_preprocessing import preprocess_data

data_dir = 'data/raw/'
mask_dir = 'data/masks/'
output_dir = 'data/processed/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

preprocess_data(data_dir, mask_dir, output_dir)
