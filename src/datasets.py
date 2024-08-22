# File containing definition of dataset class used to load in the mycetoma images and masks

import torch
from torch.utils.data import Dataset
import numpy as np
from src.utils import clip_and_norm, return_image_and_mask, return_image

from PIL import Image
import torchvision.transforms.v2 as transforms



# Define the 2D Dataset class
# Image and mask both need same transforms to be applied, so DO NOT USE RANDOM TRANSFORMS
# - use e.g. transforms.functional.hflip which has no randomness.
class MycetomaDataset(Dataset):
    def __init__(self, paths, data_dir, transform=None, test_flag=False):
        self.paths = paths
        self.data_dir = data_dir
        self.transform = transform
        self.test_flag = test_flag

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        if self.test_flag:
            image = return_image(self.data_dir, path)
            # Turn from PIL to numpy array
            image = np.asarray(image)
            
        else:
            # Determine class based on the path
            if 'BM' in path:
                label = 1  # BM is the positive class
            elif 'FM' in path:
                label = 0  # FM is the negative class
            else:
                raise ValueError(f"Path {path} does not contain 'BM' or 'FM'.")

            # transforms?
            if self.transform != None:
                image, mask = return_image_and_mask(self.data_dir, path, as_numpy=False)
                image, mask = self.transform(image, mask)

                # Turn from PIL to numpy array
                image = np.asarray(image)
                mask = np.asarray(mask)
            else:
                image, mask = return_image_and_mask(self.data_dir, path)

            # if 3 channel mask, take first
            if len(mask.shape) > 2:
                mask = mask[...,0]
            assert mask.shape == (600, 800), f"Mask shape must be (600, 800), got shape {mask.shape}"

            # clip mask
            mask = np.clip(mask, 0, 1)

            # turn to torch, permute image to move channel to front, and return
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        # Check image and mask size
        assert image.shape == (600, 800, 3), f"Image shape must be (600, 800, 3), got shape {image.shape}"
        # normalise image
        image = clip_and_norm(image, 255)

        
        # turn to torch, permute image to move channel to front, and return
        image = torch.from_numpy(image).float().permute(2,0,1)

        if self.test_flag:
            return image
        else:
            return image, mask, label
        




# Define dataset used for classifier
class MycetomaDatasetClassifier(Dataset):
    def __init__(self, paths, data_dir, mask_channel=False, transform=None, transform_chance=0.5):
        self.paths = paths
        self.data_dir = data_dir
        self.transform = transform
        self.transform_chance = transform_chance
        self.mask_channel = mask_channel

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index][0]
        mask_path = self.paths[index][1]

        # Load the image and mask
        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))
        if 'BM' in image_path:
            label = 0
        if 'FM' in image_path:
            label = 1

        # Check image and mask size
        assert image.shape == (600, 800, 3), "Image size must be (600, 800, 3)"

        # if mask more than one channel, turn to 2d by taking first channel
        if len(mask.shape) > 2:
            mask = mask[..., 0]

        # normalise image
        image = clip_and_norm(image, 255)
        

        # clip mask
        # assert mask.max() == 1, "Mask must be binary"

        # turn to torch, permute image to move channel to front, and return
        image = torch.from_numpy(image).float().permute(2,0,1)
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        if self.transform != None:
            transform1 = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
            transform2 = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)
            ])
            
            image, mask = transform1(image, mask)
            image = transform2(image)
                
        if self.mask_channel:
            image = torch.cat((image, mask), dim=0)

        return image, label