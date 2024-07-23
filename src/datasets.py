# File containing definition of dataset class used to load in the mycetoma images and masks

from PIL import Image
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from utils import clip_and_norm, return_image_and_mask

# Define the 2D Dataset class
# Image and mask both need same transforms to be applied, so DO NOT USE RANDOM TRANSFORMS
# - use e.g. transforms.functional.hflip which has no randomness.
class MycetomaDataset(Dataset):
    def __init__(self, paths, data_dir, transform=None):
        self.paths = paths
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # transforms?
        if self.transform != None:
            image, mask = return_image_and_mask(self.data_dir, path, as_numpy=False)
            image, mask = self.transform(image, mask)

            # Turn from PIL to numpy array
            image = np.asarray(image)
            mask = np.asarray(mask)
        else:
            image, mask = return_image_and_mask(self.data_dir, path)

        # Check image and mask size
        assert image.shape == (600, 800, 3), f"Image shape must be (600, 800, 3), got shape {image.shape}"
        assert mask.shape == (600, 800), f"Mask shape must be (600, 800), got shape {mask.shape}"

        # normalise image
        image = clip_and_norm(image, 255)

        # clip mask
        mask = np.clip(mask, 0, 1)

        # turn to torch, permute image to move channel to front, and return
        image = torch.from_numpy(image).float().permute(2,0,1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask