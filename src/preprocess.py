# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import hashlib
from PIL import Image
from collections import defaultdict

sys.path.append('../src')
from utils import return_image_and_mask, return_image, return_mask, plot_image_and_mask

# Set data directory
DATA_DIR = '../data'

# Get full image path by adding filename to base path

# Get the paths
test_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/test_dataset/**/*.jpg')])

print(f"Test length: {len(test_paths)}")

# Sort paths to make sorting by patient easier
test_paths.sort()

# Check specific index
idx = 10
print(test_paths[idx:idx+10])



# Create image hash
def compute_image_hash(data_dir, path):

    image = return_image(data_dir, path)
    return hashlib.md5(image.tobytes()).hexdigest()


# Create dictionary for image hashes
def image_hash_dict(data_dir, paths):

    image_hashes = {}

    for path in paths:
        image_hashes[path] = compute_image_hash(data_dir, path)

    return image_hashes


# Use default dict to see which image appears in multiple paths
# Takes dictionary with key: value pairs of image path: image hash
# Returns dictionary of hash values (that appear more than once), with list of paths as values
def get_image_duplicates(hashes):
    hash_path_dict = defaultdict(list)
        
    for path, image_hash in hashes.items():
        hash_path_dict[image_hash].append(path)

    duplicates = {k: v for k, v in hash_path_dict.items() if len(v) > 1}
    return duplicates