# Script to train a 2D UNet to segment grains from mycetoma histopathology images
# Currently is performing binary semantic segmentation - can look at two classes after

## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm

sys.path.append('../src')
from UNet2D import UNet2D
from datasets import MycetomaDataset
from metrics import batch_dice_coeff, bce_dice_loss, dice_coefficient

DATA_DIR = '../data'

# Get full image path by adding filename to base path
# Get the paths
train_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/training_dataset/**/*.jpg')])
val_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/validation_dataset/**/*.jpg')])

print(f"Train length: {len(train_paths)}")
print(f"Val length: {len(val_paths)}")

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Make Datasets
train_dataset = MycetomaDataset(train_paths, DATA_DIR)
val_dataset = MycetomaDataset(val_paths, DATA_DIR)

train_dataset.__len__()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# test batch
batch = next(iter(train_loader))
im, mask = batch
print(f'img shape: {im.shape}; mask shape: {mask.shape}')

# Make model
model = UNet2D(3, 1, 8)
model = model.to(device)

import gc
torch.cuda.empty_cache()
gc.collect()

train_dice_scores = []
val_dice_scores = []
train_losses = []
val_losses = []

# TRAIN
num_epochs = 20
threshold = 0.5
best_val_loss = np.inf

# Specify optimiser and criterion
criterion = bce_dice_loss
l_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=l_rate)

for epoch in range(num_epochs):

    # train mode
    model.train()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0    # counter for num of batches

    # Loop through train loader
    for idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)

        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward, backward, and update params
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().cpu().numpy()
        dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
        n += 1

    # Get train metrics, averaged over number of images in batch
    train_loss = running_loss/n
    train_dice_av = dice_coeff/n

    # After each batch, loop through validation loader and get metrics
    # set model to eval mode and reset metrics
    model.eval()
    running_loss = 0.0
    dice_coeff = 0.0
    n = 0

    print("------------ VALIDATION -------------")

    # Perform loop without computing gradients
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.detach().cpu().numpy()
            dice_coeff += batch_dice_coeff(outputs>threshold, targets).detach().cpu().numpy()
            n += 1

    # Val metrics
    val_loss = running_loss/n
    val_dice_av = dice_coeff/n

    # print stats
    print(f"--------- EPOCH {epoch} ---------")
    print(f"Train Loss: {train_loss}, Train Dice Score: {train_dice_av}")
    print(f"Val Loss: {val_loss}, Val Dice Score: {val_dice_av}")

    # save stats
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dice_scores.append(train_dice_av)
    val_dice_scores.append(val_dice_av)