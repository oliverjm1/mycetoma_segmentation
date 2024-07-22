# Postprocessing of mask outputs

import cv2
import numpy as np

# Apply threshold to get a binary mask
def threshold_mask(mask, threshold=0.5):
    binary_mask = (mask > threshold).astype(np.uint8)
    return binary_mask

# Fill contours
def fill_holes_contours(binary_mask):
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an empty mask to draw filled contours
    filled_mask = np.zeros_like(binary_mask)
    
    # Draw filled contours
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled_mask

# Function that uses kernel to fill small holes/remove small noise
# Only the largest n components are kept
def post_process_binary_mask(binary_mask, threshold_fraction=0.1, kernel_size=5):   

    # Define a kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Remove small noise
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
    
    # Connected Component Analysis
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # Find the largest component's area
    areas = [stat[cv2.CC_STAT_AREA] for i, stat in enumerate(stats) if i != 0]
    largest_area = max(areas) if areas else 0
    size_threshold = largest_area * threshold_fraction
    
    # Create a mask to keep only significant components
    filtered_mask = np.zeros_like(cleaned_mask)
    
    for i in range(1, num_labels):  # Start from 1 to skip the background component
        if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
            filtered_mask[labels == i] = 1
    
    # Fill contours
    filtered_mask = fill_holes_contours(filtered_mask)

    filtered_mask = np.clip(filtered_mask, 0, 1)

    return filtered_mask