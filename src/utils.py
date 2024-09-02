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

def visualize_image_classified_segmented(image_orig_input, mask_seg_predicted, predicted_class, imagename_output):
    # Plot prediction before and after processing
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_orig_input)
    ax[0].set_title('Input Image')
    ax[0].axis('off')

    ax[1].imshow(mask_seg_predicted)
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')

    #adding text inside the plot
    plt.text(-450, 800, 'Class Predicted: '+str(predicted_class), fontsize = 22)

    plt.savefig(imagename_output)
    #plt.show()

def format_file_paths_simplified(paths, img_paths): #adapted by Asra because we already ahve masks and images in different folders, no need to search exhaustively
    file_paths = []
    for file in img_paths:
        patient = []
        patient.append(file)
        matched = False
        file1_end = file.split("\\")[-1].split('.')[0]
        print(file1_end)
        print(file)
        for file2 in paths:
            file2_end = file2.split("\\")[-1].split('.')[0].split('_')[0]
            if file1_end == file2_end:
                print(file2_end)
                print(file2)
                patient.append(file2)
                print(patient)
                file_paths.append(patient)
                patient = []
                matched = True
                break
        if matched == False:
            print(f"Could not find mask for {file1_end}")
            

    return file_paths

# def format_file_paths(paths, img_paths):
#     file_paths = []
#     patient = []
#     for file in img_paths:
#         patient.append(file)
#         matched = False
#         #file_start = '.'.join(file.split('.')[:-1])
#         file_end = file.split("\\")[-1].split('.')[0]
#         for file2 in paths:
#             if 'mask' in file2:
#                 #file_start2 = '.'.join(file2.split('.')[:-1])[:-5]
#                 file_end2 = '_'.join(file2.split("\\")[-1].split('.')[0].split('_')[0:2])
#                 if file_end == file_end2:
#                     patient.append(file2)
#                     file_paths.append(patient)
#                     patient = []
#                     matched = True
#         if matched == False:
#             print(f"Could not find mask for {file_end}")
#     return file_paths