"""
File which preforms all preprocessing of the data.
All functions used are found in src/preprocess.py.
Give the data subdirectory as a parser argument (e.g. 'training_dataset')
WARNING:    Running this script will overwrite the original dataset.
            If you want to keep the original dataset, create a copy.
"""

from src.preprocess import *
import argparse

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training UNet Multitask, saving metrics")
    parser.add_argument('--directory', type=str, required=True, help="Name of data directory to preprocess")
    return parser.parse_args()

# Parse arguments
args = parse_args()

###########################################################################
# Initialization
###########################################################################
# Set data directory
DATA_DIR = './data'

# Get the paths
paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/{args.directory}/**/*.jpg')])

# Sort paths to make sorting by patient easier
paths.sort()

# Correct mask channel issues
check_mask_channels(DATA_DIR, paths)

###########################################################################
# Delete and combining duplicate masks
###########################################################################
# Combine duplicate masks 
deal_with_duplicates(data_dir=DATA_DIR, paths=paths)

###########################################################################
# Combine overlapping masks (by Image Similarity Prediction)
###########################################################################
# Recreate list of paths now duplicates have been removed
paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/{args.directory}/**/*.jpg')])
paths.sort()

patient_ids = list(set([get_patient_id(path) for path in paths]))

overlap_adjustments(data_dir=DATA_DIR, paths=paths, patient_ids=patient_ids)