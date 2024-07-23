# Utility functions such as for normalising and transforming

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

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
def return_image_and_mask(data_dir, path, as_numpy=True):
    image_path = os.path.join(data_dir, path + '.jpg')
    mask_path = os.path.join(data_dir, path + '_mask.tif')

    # Load the image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    if as_numpy:
        image = np.asarray(image)
        mask = np.asarray(mask)

    return image, mask

# Function that takes in the data directory and relative path,
# and returns a numpy array of the image
def return_image(data_dir, path):
    image_path = os.path.join(data_dir, path + '.jpg')

    # Load the image
    image = np.asarray(Image.open(image_path))

    return image

# Function that takes in the data directory and relative path,
# and returns a numpy array of the mask
def return_mask(data_dir, path):
    mask_path = os.path.join(data_dir, path + '_mask.tif')

    # Load the mask
    mask = np.asarray(Image.open(mask_path))

    return mask

# Define function for plotting image and ground truth side by side when given a path
def plot_image_and_mask(path, DATA_DIR = '../data/'):

    # Load image and mask
    im, mask = return_image_and_mask(DATA_DIR, path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(im)
    ax[0].set_title('Image')
    ax[0].axis('off')

    ax[1].imshow(mask)
    ax[1].set_title('Mask')
    ax[1].axis('off')

    plt.show()