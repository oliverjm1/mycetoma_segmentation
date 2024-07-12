# Utility functions such as for normalising and transforming

import numpy as np
import os
from PIL import Image

# Function that takes an image and a given upper bound
# The image is clipped to this upper bound and then normalized between 0 and 1
def clip_and_norm(image, upper_bound):
    # Clip intensity values
    image = np.clip(image, 0, upper_bound)

    # Normalize the image to the range [0, 1]
    norm = (image - 0) / (upper_bound - 0)

    return norm

# Function that takes in the data directory and relative path,
# and returns a numpy array of the image and corresponding mask
def return_image_and_mask(data_dir, path):
    image_path = os.path.join(data_dir, path + '.jpg')
    mask_path = os.path.join(data_dir, path + '_mask.tif')

    # Load the image and mask
    image = np.asarray(Image.open(image_path))
    mask = np.asarray(Image.open(mask_path))

    return image, mask