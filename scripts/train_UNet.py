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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import argparse

sys.path.append('../src')
from UNet2D import UNet2D
from datasets import MycetomaDataset
from metrics import batch_dice_coeff, bce_dice_loss, dice_coefficient

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training UNet, saving metrics")
    parser.add_argument('--run_name', type=str, required=True, help="Name of run, which will govern output file names")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--use_corrected_dataset', action='store_true', help="Flag to use corrected dataset")
    parser.add_argument('--with_augmentation', action='store_true', help="Perform data augmentation on training data")
    parser.add_argument('--leave_out_bad_cases', action='store_true', help="Don't train on bad cases")
    return parser.parse_args()

# Parse arguments
args = parse_args()

DATA_DIR = '../data'

# Directory to save model and metrics
output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get full image path by adding filename to base path
# Get the paths
if args.use_corrected_dataset:
    train_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/corrected_training_dataset/**/*.jpg')])

else:
    train_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/training_dataset/**/*.jpg')])

# Always use corrected for validation(?) to be consistent
val_paths = np.array([os.path.relpath(i, DATA_DIR).split('.')[0] for i in glob.glob(f'{DATA_DIR}/corrected_validation_dataset/**/*.jpg')])

# If specified, remove bad masks (and by bad I mean a big grain not labelled at all in mask)
# Problematic train masks:
problem_train_paths = np.array([ 
    'training_dataset/BM/BM3_4',
    'training_dataset/BM/BM10_2',
    'training_dataset/BM/BM13_6',
    'training_dataset/BM/BM30_7',
    'training_dataset/BM/BM31_7',
    'training_dataset/BM/BM45_9',
    'training_dataset/FM/FM41_1',
    'training_dataset/FM/FM41_2',
    'training_dataset/FM/FM45_6',
    'training_dataset/FM/FM45_8',
    'training_dataset/BM/BM6_11',
    'training_dataset/FM/FM50_6'
])

if args.leave_out_bad_cases:
    train_paths = np.setdiff1d(train_paths, problem_train_paths)

problem_val_paths = np.array(['validation_dataset/FM/FM10_1'])

val_paths = np.setdiff1d(val_paths, problem_val_paths)

print(f"Train length: {len(train_paths)}")
print(f"Val length: {len(val_paths)}")

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the transform pipeline
if args.with_augmentation:
    print('Using Augmentation')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(600, 800), scale=(0.9, 1.0)),
    ])
else:
    transform=None

# Make Datasets
train_dataset = MycetomaDataset(train_paths, DATA_DIR, transform=transform)
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

# use multiple gpu in parallel if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

import gc
torch.cuda.empty_cache()
gc.collect()

train_dice_scores = []
val_dice_scores = []
train_losses = []
val_losses = []

# TRAIN
num_epochs = args.epochs
threshold = 0.5
best_val_loss = np.inf

# Specify optimiser and criterion
criterion = bce_dice_loss
l_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=l_rate)

best_model_path = os.path.join(output_dir, args.run_name + '_best_model.pth')

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

    # Update best model if lowest loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(model.state_dict(), best_model_path)
        print('New best model saved')

print("Training completed")

# Save scores
# Create a dictionary to hold the arrays
metrics_dict = {
    'train_losses': np.array(train_losses),
    'val_losses': np.array(val_losses),
    'train_dice_scores': np.array(train_dice_scores),
    'val_dice_scores': np.array(val_dice_scores)
}

# Save the dictionary to a .npy file
np.save(os.path.join(output_dir, args.run_name + '.npy'), metrics_dict)

print("Results Saved")