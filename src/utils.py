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


def visualize_segmented_image(post_proc_mask, im, pred, imagename_output):
    # Plot prediction before and after processing
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(im)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(pred)
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    ax[2].imshow(post_proc_mask)
    ax[2].set_title('Postprocessed Predicted Mask')
    ax[2].axis('off')

    plt.savefig(imagename_output)
    #plt.show()



def format_file_paths(paths, img_paths):
    file_paths = []
    patient = []
    for file in img_paths:
        patient.append(file)
        matched = False
        #file_start = '.'.join(file.split('.')[:-1])
        file_end = file.split("\\")[-1].split('.')[0]
        for file2 in paths:
            if 'mask' in file2:
                #file_start2 = '.'.join(file2.split('.')[:-1])[:-5]
                file_end2 = '_'.join(file2.split("\\")[-1].split('.')[0].split('_')[0:2])
                if file_end == file_end2:
                    patient.append(file2)
                    file_paths.append(patient)
                    patient = []
                    matched = True
        if matched == False:
            print(f"Could not find mask for {file_end}")
    return file_paths


def custom_dirname_creator(trial) -> str:
    # Create a shorter directory name by using only the trial_id
    trial_id = trial.trial_id
    return f"trial_{trial_id}"


def plot_calibration_curve(prob_pred, prob_true, path) -> None:
    plt.figure(figsize=(10, 5))

    # Plot calibration curve
    plt.plot(prob_pred, prob_true, marker='o', label='DenseNet121')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Predicted probability')
    plt.ylabel('True probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(path + "/calibration_curve.png", dpi=300)

# Plot distribution of predicted probabilities
def plot_pred_prob_dist(prob_pred, path) -> None:
    plt.figure(figsize=(10, 5))

    plt.hist(prob_pred, bins=10, range=(0, 1), edgecolor='k', alpha=0.7)
    plt.xlabel('Predicted probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')

    plt.tight_layout()
    plt.savefig(path + "/pred_prob_dist.png", dpi=300)

# Plot the roc curve
def plot_roc_curve(fpr, tpr, roc_auc, path) -> None:
    plt.figure(figsize=(10, 5))

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.savefig(path + "/AUROC.png", dpi=300)
