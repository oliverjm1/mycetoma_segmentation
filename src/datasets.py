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
    def __init__(self, paths, data_dir, transform=None, transform_chance=0.5):
        self.paths = paths
        self.data_dir = data_dir
        self.transform = transform
        self.transform_chance = transform_chance

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        image, mask = return_image_and_mask(self.data_dir, path)

        # Check image and mask size
        assert image.shape == (600, 800, 3), "Image size must be (600, 800, 3)"

        # if mask more than one channel, turn to 2d by taking first channel
        if len(mask.shape) > 2:
            mask = mask[..., 0]

        # normalise image
        image = clip_and_norm(image, 255)

        # clip mask
        assert mask.max() == 1, "Mask must be binary"

        # turn to torch, permute image to move channel to front, and return
        image = torch.from_numpy(image).float().permute(2,0,1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        # transforms?
        if self.transform != None:

            # Need to take care of randomness myself because need same transform applied to image and gt
            if random.random() < self.transform_chance:
                image = self.transform(image)
                mask = self.transform(mask)

        return image, mask