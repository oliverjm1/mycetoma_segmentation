# imports
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
import hashlib
from PIL import Image
from collections import defaultdict
import cv2
from tqdm import tqdm
from itertools import combinations

sys.path.append('../src')
from src.utils import return_image_and_mask, return_image, return_mask, plot_image_and_mask


# check size of every mask - if 3 channels then overwrite the file with one channel
def check_mask_channels(data_dir, paths):
    for i, path in enumerate(paths):
        _, mask = return_image_and_mask(data_dir, path)
        if mask.shape != (600, 800):
            print(f'problem with {path}: shape is {mask.shape}')
            single_channel_mask = mask[:, :, 0]
                
            # Convert the single-channel array back to an image
            single_channel_img = Image.fromarray(single_channel_mask)
            
            # Save the single-channel image, overwriting the original file
            single_channel_img.save(data_dir + '/' + path + '_mask.tif')

            print('overwritten')


###########################################################################
# Functions to combine duplicate masks
###########################################################################

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


# Function that takes list of path names, and combines the masks of these paths. Clips to 1 to keep mask binary.
def combine_duplicate_masks(data_dir, duplicate_paths):
    # all images and masks are (600,800)
    combined_duplicate_mask = np.zeros(shape=(600,800))

    # for each path, return the mask and add to combined mask
    for path in duplicate_paths:
        combined_duplicate_mask += return_mask(data_dir, path)

    # clip the mask
    combined_duplicate_mask = np.clip(combined_duplicate_mask, 0, 1)
    
    return combined_duplicate_mask



# FUNCTION TO COMBINE DUPLICATE MASKS, SAVE NEW ONE AND DELETE DUPLICATES
def deal_with_duplicates(data_dir, paths):

    # Get path to hash dictionary
    print('getting hash dict...')
    hash_dict = image_hash_dict(data_dir, paths)

    # Get duplicates
    duplicates = get_image_duplicates(hash_dict)
    
    # For each set of duplicates,
    print('looping through duplicates...')
    for duplicate_paths in duplicates.values():
        
        print(f"data_dir = {data_dir}")

        # Combine masks into one
        combined_duplicate_masks = combine_duplicate_masks(data_dir, duplicate_paths)

        # Overwrite first mask
        first_path = duplicate_paths[0]
        combined_mask_img = Image.fromarray(combined_duplicate_masks)
        
        # Save the single-channel image, overwriting the original file
        combined_mask_img.save(data_dir + '/' + first_path + '_mask.tif')

        # Delete duplicate images and masks
        for path in duplicate_paths[1:]:
            os.remove(os.path.join(data_dir, path) + '.jpg')
            os.remove(os.path.join(data_dir, path) + '_mask.tif')
            print(f"Deleted {path} from the dataset.")
        
    print('done.')




###########################################################################
# Functions to combine overlapping masks
###########################################################################

# Function to split path and return only the patient code (e.g. BM10)
def get_patient_id(path):
    
    # First take final part of the path, then take everything before the underscore
    patient_id = path.split('\\')[2].split('_')[0]
    return patient_id


# For a given patient, get all image paths
def get_patient_paths(paths, patient_id):
    patient_paths = [path for path in paths if patient_id in path]
    return patient_paths


# Function to match features of two images
# To be used as part of process to check if two images share an overlapping region
def detect_and_match_features(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return keypoints1, keypoints2, matches




# After keypoints and matches of two images have been calculated,
# this function takes the mean of the top 10 match translations 
def estimate_translation(keypoints1, keypoints2, matches):
    
    # Calculate the translation vector
    translations = []

    for match in matches[:10]:
            pt1 = keypoints1[match.queryIdx].pt
            pt2 = keypoints2[match.trainIdx].pt
            translations.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))

    translations = np.array(translations)
    median_translation = np.rint(np.median(translations, axis=0))

    return median_translation

# Function to apply x-y translation to an image
def translate_image(image, translation):

    rows, cols = image.shape[:2]
    translation = (-translation[0], -translation[1])
    translation_matrix = np.float32([[1, 0, translation[0]], [0, 1, translation[1]]])
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows))

    return translated_image



# Once an image has been translated, find the span of the nonzero part (the overlap part)
def find_overlap_bbox(image):

    # Find coordinates of all non-zero pixels
    coords = cv2.findNonZero(image[...,0])
    
    # Calculate the bounding box of the non-zero region
    x, y, w, h = cv2.boundingRect(coords)
    min_x, min_y, max_x, max_y = x, y, x + w, y + h

    return min_x, min_y, max_x, max_y



# OVERLAP FUNCTION
# After matching features and getting a translation, the second image is translated to line up with the first.
# A bounding box of the matching region is then taken of both images, and the difference is taken.
# If the images truly do share a region, this difference is essentially zero (do blur of both before to account for small mis-alignment).

def check_overlap(image1, image2, visualize=True):
    # Detect and match features
    keypoints1, keypoints2, matches = detect_and_match_features(image1, image2)
    
    # Estimate translation
    translation = estimate_translation(keypoints1, keypoints2, matches)
    
    # Translate the second image
    translated_image2 = translate_image(image2, translation)
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(translated_image2, cv2.COLOR_BGR2GRAY)

    # Crop both to matching region
    xl,yl,xh,yh = find_overlap_bbox(translated_image2)
    crop1 = gray1[yl:yh,xl:xh]
    crop2 = gray2[yl:yh,xl:xh]

    # Gaussian Blur before measuring difference to reduce alignment error
    blurred_crop1 = cv2.GaussianBlur(crop1, (11, 11), 0)
    blurred_crop2 = cv2.GaussianBlur(crop2, (11, 11), 0)
    
    # Compute absolute difference between images
    diff = cv2.absdiff(blurred_crop1, blurred_crop2)

    # Threshold the difference image
    threshold_val = 40
    _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    # Calculate percentage of overlap
    overlap = np.count_nonzero(thresh == 0) / np.size(thresh)

    return overlap, translation


# If the overlap is greater than a certain threshold, shift the masks in the same way and add to each other.

def overlap_adjustments(data_dir, paths, patient_ids):
    # cycle through patients
    for i, patient in enumerate(patient_ids):
        print(f'{patient} ({i+1}/23)')
        patient_paths = get_patient_paths(paths, patient)

        # cycle through path combinations
        for path1, path2 in combinations(patient_paths, 2):

            # Load images and masks
            image1, mask1 = return_image_and_mask(data_dir, path1)
            image2, mask2 = return_image_and_mask(data_dir, path2)
            
            # Check overlap
            overlap, translation = check_overlap(image1, image2, visualize=False)

            # If the overlap is sufficient
            if overlap > 0.9:

                # Translate mask2 in same way
                translated_mask2 = translate_image(mask2, translation)

                # Translate mask1 in opposite way
                translated_mask1 = translate_image(mask1, (-translation[0],-translation[1]))

                # Add to mask1
                try:
                    updated_mask1 = mask1 + translated_mask2
                    updated_mask1 = np.clip(updated_mask1, 0, 1)

                    # Add to mask2
                    updated_mask2 = mask2 + translated_mask1
                    updated_mask2 = np.clip(updated_mask2, 0, 1)

                    # Overwrite mask1 and mask2 files 
                    updated_mask1_img = Image.fromarray(updated_mask1)
                    updated_mask2_img = Image.fromarray(updated_mask2)
            
                    # Save the single-channel image, overwriting the original file
                    updated_mask1_img.save(data_dir + '/' + path1 + '_mask.tif')
                    updated_mask2_img.save(data_dir + '/' + path2 + '_mask.tif')

                    print(f'Mask at {path1} updated')
                
                except:
                    print("Failed to combine translated mask - too many channels?")
