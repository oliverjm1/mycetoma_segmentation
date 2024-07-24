# Script to train a 2D UNet to segment grains from mycetoma histopathology images
# Currently is performing binary semantic segmentation - can look at two classes after

## IMPORTS
import numpy as np
import glob
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from tqdm import tqdm
import argparse

sys.path.append('../src')
from UNetMultiTask import UNetMultiTask
from datasets import MycetomaDataset
from metrics import accuracy, batch_dice_coeff, bce_dice_loss, dice_coefficient

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training UNet Multitask, saving metrics")
    parser.add_argument('--run_name', type=str, required=True, help="Name of run, which will govern output file names")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--use_corrected_dataset', action='store_true', help="Flag to use corrected dataset")
    parser.add_argument('--with_augmentation', action='store_true', help="Perform data augmentation on training data")
    parser.add_argument('--leave_out_bad_cases', action='store_true', help="Don't train on bad cases")
    parser.add_argument('--class_loss_weight', type=int, default=1, help='Weight for the classification loss')
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
    'corrected_training_dataset/BM/BM3_4',
    'corrected_training_dataset/BM/BM10_2',
    'corrected_training_dataset/BM/BM13_6',
    'corrected_training_dataset/BM/BM30_7',
    'corrected_training_dataset/BM/BM31_7',
    'corrected_training_dataset/BM/BM45_9',
    'corrected_training_dataset/FM/FM41_1',
    'corrected_training_dataset/FM/FM41_2',
    'corrected_training_dataset/FM/FM45_6',
    'corrected_training_dataset/FM/FM45_8',
    'corrected_training_dataset/BM/BM6_11',
    'corrected_training_dataset/FM/FM50_6'
])

if args.leave_out_bad_cases:
    train_paths = np.setdiff1d(train_paths, problem_train_paths)

problem_val_paths = np.array(['corrected_validation_dataset/FM/FM10_1'])

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
im, mask, label = batch
print(f'img shape: {im.shape}; mask shape: {mask.shape}; label shape: {label.shape}')

# Make model
model = UNetMultiTask(3, 1, 8)
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
train_accuracies = []
val_accuracies = []

# TRAIN
num_epochs = args.epochs
threshold = 0.5
best_val_loss = np.inf

# Specify optimiser and criterion
seg_criterion = bce_dice_loss
class_criterion = nn.BCELoss()
l_rate = 5e-5
optimizer = optim.Adam(model.parameters(), lr=l_rate)

best_model_path = os.path.join(output_dir, args.run_name + '_best_model.pth')

for epoch in range(num_epochs):

    # train mode
    model.train()
    running_loss = 0.0
    dice_coeff = 0.0
    total_accuracy = 0.0
    n = 0    # counter for num of batches

    # Loop through train loader
    for idx, (inputs, targets, labels) in enumerate(tqdm(train_loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        labels = labels.to(device) 

        optimizer.zero_grad()

        # Forward, backward, and update params
        seg_outputs, class_outputs = model(inputs)

        seg_loss = seg_criterion(seg_outputs, targets)
        class_loss = class_criterion(class_outputs.squeeze(), labels.float())

        total_loss = seg_loss + (args.loss_weight * class_loss)

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.detach().cpu().numpy()
        dice_coeff += batch_dice_coeff(seg_outputs>threshold, targets).detach().cpu().numpy()
        total_accuracy += accuracy(class_outputs.squeeze().detach().cpu(), labels.detach().cpu())
        n += 1

        # print classification outputs
        """print(f"Out: {class_outputs.squeeze().detach().cpu()}")
        print(f"Label: {labels.detach().cpu()}")

        print(f"LOSSES: seg = {seg_loss.detach().cpu().numpy()}, class = {class_loss.detach().cpu().numpy()}, total = {total_loss.detach().cpu().numpy()}")
        print(f"METRICS: dice = {batch_dice_coeff(seg_outputs>threshold, targets).detach().cpu().numpy()}, acc = {accuracy(class_outputs.squeeze().detach().cpu(), labels.detach().cpu())}")"""

    # Get train metrics, averaged over number of images in batch
    train_loss = running_loss/n
    train_dice_av = dice_coeff/n
    train_accuracy = total_accuracy/n

    # After each batch, loop through validation loader and get metrics
    # set model to eval mode and reset metrics
    model.eval()
    running_loss = 0.0
    dice_coeff = 0.0
    total_accuracy = 0.0
    n = 0

    print("------------ VALIDATION -------------")

    # Perform loop without computing gradients
    with torch.no_grad():
        for idx, (inputs, targets, labels) in enumerate(tqdm(val_loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            labels = labels.to(device) 

            seg_outputs, class_outputs = model(inputs)

            seg_loss = seg_criterion(seg_outputs, targets)
            class_loss = class_criterion(class_outputs.squeeze(), labels.float())

            total_loss = seg_loss + (args.loss_weight * class_loss)

            running_loss += total_loss.detach().cpu().numpy()
            dice_coeff += batch_dice_coeff(seg_outputs>threshold, targets).detach().cpu().numpy()
            total_accuracy += accuracy(class_outputs.squeeze().detach().cpu(), labels.detach().cpu())
            n += 1

    # Val metrics
    val_loss = running_loss/n
    val_dice_av = dice_coeff/n
    val_accuracy = total_accuracy/n

    # print stats
    print(f"--------- EPOCH {epoch} ---------")
    print(f"Train Loss: {train_loss}, Train Dice Score: {train_dice_av}, Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Val Loss: {val_loss}, Val Dice Score: {val_dice_av}, Val Accuracy: {val_accuracy * 100:.2f}%")

    # save stats
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dice_scores.append(train_dice_av)
    val_dice_scores.append(val_dice_av)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

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
    'val_dice_scores': np.array(val_dice_scores),
    'train_accuracies': np.array(train_accuracies),
    'val_accuracies': np.array(val_accuracies)
}

# Save the dictionary to a .npy file
np.save(os.path.join(output_dir, args.run_name + '.npy'), metrics_dict)

print("Results Saved")