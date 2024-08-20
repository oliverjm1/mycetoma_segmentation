import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# import matplotlib.pyplot as plt

import pandas as pd
import random
import glob
from collections import Counter
from PIL import Image

# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
# from sklearn.calibration import calibration_curve

# from monai.networks.nets import DenseNet121 #, HighResNet, EfficientNet, ResNet

from ray import tune
from ray.tune.schedulers import ASHAScheduler # taken from link https://www.geeksforgeeks.org/hyperparameter-tuning-with-ray-tune-in-pytorch/
from ray.tune.search.hyperopt import HyperOptSearch


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



###############################################################################################
# Set up train images and masks 
###############################################################################################

# Set data directory
DATA_DIR = ".\\data"
TRAIN_DATA_DIR = ".\\data\\training_dataset"

# Get the training paths
train_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}\\training_dataset\\**\\*')])
val_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}\\validation_dataset\\**\\*')])

# Post-processing binary 
train_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/binary postprocessed/corrected_masks_and_augmented_postproc_training/**/*')])
val_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/binary postprocessed/corrected_masks_and_augmented_postproc_validation/**/*')])

# Post-processing logit
train_seg_paths_log = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/logit output/corrected_masks_and_augmented_training/**/*')])
val_seg_paths_log = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/logit output/corrected_masks_and_augmented_validation/**/*')])

# multitask binary
train_seg_paths_multibin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/binary postprocessed/multitask_postproc_training/**/*')])
val_seg_paths_multibin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/binary postprocessed/multitask_postproc_validation/**/*')])

# multitask logit
train_seg_paths_multilog = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/logit output/multitask_training/**/*')])
val_seg_paths_multilog = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/logit output/multitask_validation/**/*')])

print(glob.glob(f'{DATA_DIR}\\training_dataset\\**\\*'))
      
print(train_paths)