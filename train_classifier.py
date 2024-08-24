# TODO: add code for hyperparamter sweep (ray-tune or wandb?) - added ray-tune for now
# TODO: add model weights savepoint
# TODO: decide whether to save plots on ARC4
# TODO: create virtual environment on ARC4 
# TODO: push relevant changes to from dev-james to GitHub so theyre acessible on ARC4


###############################################################################################
# Library imports
###############################################################################################
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import matplotlib.pyplot as plt

import glob
from collections import Counter
from PIL import Image

# Import evaluation metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

# Import DenseNet model
from monai.networks.nets import DenseNet121 #, HighResNet, EfficientNet, ResNet

# Import ray for hyperparamter tuning
from ray import tune
from ray.tune.schedulers import ASHAScheduler # taken from link https://www.geeksforgeeks.org/hyperparameter-tuning-with-ray-tune-in-pytorch/
from ray.tune.search.hyperopt import HyperOptSearch

from src.utils import format_file_paths, custom_dirname_creator, plot_calibration_curve, plot_pred_prob_dist, plot_roc_curve, plot_losses
from src.datasets import MycetomaDatasetClassifier

# Set running environment (True for HPC, False for local)
HPC_FLAG = sys.argv[1]

# Set debugging
DEBUG = True
DEBUG_PRINT_INTERVAL = 10

def debug_print(debug_statement):
    if DEBUG:
        print(debug_statement)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###############################################################################################
# Set up paths for images and masks 
###############################################################################################
debug_print("\n\n############ Setting up paths to data #############")

# Set data directory
def define_dataset(hpc=0):
    
    debug_print(f"Current directory: {os.getcwd()}")

    # Set data, plots save and model checkpoint paths
    if hpc:
        debug_print("Setting data paths for ARC4...")
        data_dir = "/nobackup/scjb/mycetoma/data/"
        plot_save_path = "/home/home02/scjb/mycetoma_segmentation-dev-james/train_stats"
        model_checkpoints_path = "/home/home02/scjb/mycetoma_segmentation-dev-james/model_saves"
    
    else:
        data_dir = "C:/Users/james/Documents/projects/mycetoma_segmentation/data"
        plot_save_path = "C:/Users/james/Documents/projects/mycetoma_segmentation/train_stats"
        model_checkpoints_path = "C:/Users/james/Documents/projects/mycetoma_segmentation/model_saves"
          
    # Get the training paths
    train_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{data_dir}/training_dataset/**/*')])
    val_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{data_dir}/validation_dataset/**/*')])

    # Post-processing binary 
    train_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{data_dir}/binary postprocessed/corrected_masks_and_augmented_postproc_training/**/*')])
    val_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{data_dir}/binary postprocessed/corrected_masks_and_augmented_postproc_validation/**/*')])

    # Extract just the image paths
    train_img_paths = train_paths[[not 'mask' in i for i in train_paths]]
    val_img_paths = val_paths[[not 'mask' in i for i in val_paths]]
    img_paths = [train_img_paths, val_img_paths]

    # Combine image and segmentation map paths for each patient
    train_paths = format_file_paths(train_seg_paths_bin, train_img_paths)
    val_paths = format_file_paths(val_seg_paths_bin, val_img_paths)

    debug_print(f"Train length: {len(train_paths)}")
    debug_print(f"Val length: {len(val_paths)}")

    debug_print(f"train_paths first example = {np.array(train_paths).shape} = {train_paths[0]}")
    debug_print(f"val_path first example = {np.array(val_paths).shape} = {train_paths[0]}")

    return data_dir, train_paths, val_paths, plot_save_path, model_checkpoints_path 


# ARC4
# DATA_DIR = "/nobackup/scjb/mycetoma/data/"
# # TRAIN_DATA_DIR = "/nobackup/scjb/mycetoma/data/training_dataset"
# PLOT_SAVE_DIR = "./train_stats"

# Post-processing logit
# train_seg_paths_log = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/logit output/corrected_masks_and_augmented_training/**/*')])
# val_seg_paths_log = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/logit output/corrected_masks_and_augmented_validation/**/*')])

# # # multitask binary
# train_seg_paths_multibin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/binary postprocessed/multitask_postproc_training/**/*')])
# val_seg_paths_multibin = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/binary postprocessed/multitask_postproc_validation/**/*')])

# # # multitask logit
# train_seg_paths_multilog = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/logit output/multitask_training/**/*')])
# val_seg_paths_multilog = np.array(['.'.join(i.split('.')) for i in glob.glob(f'{DATA_DIR}/multitask/logit output/multitask_validation/**/*')])


# Define segmentation mask types
seg_paths = ["binary postprocessed/corrected_masks_and_augmented_postproc",
            "logit output/corrected_masks_and_augmented",
            "multitask/binary postprocessed/multitask_postproc",
            "multitask/logit output/multitask"]


###############################################################################################
# Define hyperparameters
###############################################################################################
debug_print("\n\n############ Defining hyperparameters #############")


# Master list of hyperparameters - if we have time
# # Define DenseNet model
# model_dense121 = DenseNet121(spatial_dims=2, in_channels=num_channels, out_channels=1, pretrained=True)
# model_dense169 = DenseNet121(spatial_dims=2, in_channels=num_channels, out_channels=1, pretrained=True)
# model_dense264 = DenseNet121(spatial_dims=2, in_channels=num_channels, out_channels=1, pretrained=True)

# hyperparams_full_sweep = {
#     "lr": tune.loguniform(1e-5, 1e-3),
#     "batch_size": tune.choice([5, 10, 12, 16]),
#     "weight_decay": tune.loguniform(1e-5, 1e-2),
#     "mask_channel": tune.choice([True, False]),
#     "threshold": tune.uniform(0.4, 0.6),
#     "num_epochs": tune.choice([50]),
#     "seg_path": tune.choice([0,1,2,3]), # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
#     "model_type": model_dense121, model_dense169, model_dense264
#     "model_size": ,
#     "pre_train": tune.choice([True, False]),
#     "accumulation": ,
# }

hyperparams = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([5, 10, 12, 16]),
    "weight_decay": tune.loguniform(1e-4,1e-2), 
    "mask_channel": tune.choice([True, False]),
    "threshold": tune.uniform(0.4, 0.6),
    "num_epochs": tune.choice([30]),
    "seg_path": 0 # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
    }


# hyperparams_local_test = {
#     "lr": tune.loguniform(1e-3, 1e-2),
#     "batch_size": tune.choice([2, 4]),
#     "weight_decay": tune.choice([1e-2]),
#     "mask_channel": tune.choice([True]),
#     "threshold": tune.choice([0.4]),
#     "num_epochs": tune.choice([2]),
#     "seg_path": 0 # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
#     }

# hyperparams_single = {
#     "lr": 1e-5,
#     "batch_size": 5,
#     "weight_decay": 1e-5,
#     "mask_channel": True,
#     "threshold": 0.5,
#     "num_epochs": 50,
#     "seg_path": 0 # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
#     }

debug_print(f"hyperparams = {hyperparams}")
            
# Define segmentation mask type from hyperparameters
seg_path = hyperparams["seg_path"]
print(f"sep_path = {seg_path}\nsep_path type = {type(seg_path)}")
seg_path_end = seg_paths[seg_path]



###############################################################################################
# Model training
###############################################################################################

def train_model(hyperparams):

    data_dir, train_paths, val_paths, plot_save_dir, model_checkpoints_path = define_dataset(HPC_FLAG)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug_print(f"device = {device}")    

    # Set hyperparameter values
    num_epochs = hyperparams["num_epochs"] #config["num_epochs"]
    lr = hyperparams["lr"]
    batch_size = hyperparams["batch_size"]
    weight_decay = hyperparams["weight_decay"]
    mask_channel = hyperparams["mask_channel"]
    threshold = hyperparams["threshold"]

    if mask_channel:
        num_channels = 4
    else:
        num_channels = 3
    

    # Create datasets
    train_dataset = MycetomaDatasetClassifier(train_paths, data_dir, mask_channel=mask_channel, transform=True)
    val_dataset = MycetomaDatasetClassifier(val_paths, data_dir, mask_channel=mask_channel)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)
    
    # Define DenseNet model
    model = DenseNet121(
        spatial_dims=2,
        in_channels=num_channels,
        out_channels=1,
        pretrained=True
    )
    
    debug_print("Loading model to device...") 
    model = model.to(device)    

    criterion = nn.BCEWithLogitsLoss()
    debug_print(f"Criterion = {criterion}")

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    debug_print(f"Optimiser = {optimiser}")

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=10, verbose=True,
                                                              threshold=0.01, threshold_mode='abs')
    debug_print(f"Learning rate scheduler = {lr_scheduler}")
    

    train_losses, val_losses = [], []
    accumulation_steps = 3
    
    # Initialise minimum validation loss as infinity
    min_val_loss = float('inf')

    debug_print(f"\nTraining model for {num_epochs} epochs...\n")
    for epoch in range(num_epochs):

        debug_print(f"\n\n######### TRAINING EPOCH {epoch + 1} #########\n")
        debug_print("\n-------- Training phase --------")

        # Training phase
        model.train()
        train_loss = 0
        
        # Initialise train labels and predictions lists
        all_train_labels = []
        all_train_preds = []
        
        steps = 0
        optimiser.zero_grad()

        len_train_loader = len(train_loader)

        # Loop through train loader
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            steps += 1

            # Forward pass
            if steps % DEBUG_PRINT_INTERVAL == 0: 
                debug_print(f"Computing model outputs... [{steps} / {len_train_loader} batches]")

            output = model(features)  # Remove batch dimension
            output = output.squeeze(1)

            # Calculate loss and add to cumulative batch loss
            if steps % DEBUG_PRINT_INTERVAL == 0: 
                debug_print(f"Computing loss from outputs... [{steps} / {len_train_loader} batches]")

            loss = criterion(output, labels.float())
            train_loss += loss.item()

            if steps % DEBUG_PRINT_INTERVAL == 0: 
                debug_print(f"Computing backpropagation... [{steps} / {len_train_loader} batches]")
            # Backward pass and optimization
            loss.backward()

            if steps % accumulation_steps == 1:
                optimiser.step()
                optimiser.zero_grad()
    
            # Apply threshold to determine predicted class
            # predicted_probs = F.softmax(output, dim=1)[:, 1]  # Probability of class 1 (positive)
            if steps % DEBUG_PRINT_INTERVAL == 0: 
                debug_print(f"Computing predicted classes from model outputs... [{steps} / {len_train_loader} batches]")

            predicted_probs = torch.sigmoid(output)
            predicted_class = (predicted_probs >= threshold).long()
    
            # Store predictions and labels
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted_class.cpu().numpy())
    

        optimiser.step()
        optimiser.zero_grad()
        lr_scheduler.step(train_loss/len(train_loader))

        print('Learning rate:', optimiser.param_groups[0]['lr'])
        
        # Calculate evaluation metrics: 
        # Accuracy, AUC, confusion matrix
        debug_print("Computing training evaluation metrics...")
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_auc = roc_auc_score(all_train_labels, all_train_preds)
        train_confusion_matrix = confusion_matrix(all_train_labels, all_train_preds)
        tn, fp, fn, tp = confusion_matrix(all_train_labels, all_train_preds).ravel()

        # Compute sensitivity (recall) and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        if epoch % 1 == 0: # set to 1 for debugging 
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}', f'MCC: {mcc:.4f}')
            print(f'Train Confusion Matrix:')
            print(train_confusion_matrix)
        train_losses.append(train_loss/len(train_loader))
    

        # Evaluation phase
        debug_print("\n----------- Evaluation phase -----------")
        debug_print("Computing validation evaluation metrics...")
        model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_preds = []
        all_val_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                output = model(features)  # Remove batch dimension
                output = output.squeeze(1)
                loss = criterion(output, labels.float())
                val_loss += loss.item()
    
                # Store predictions and labels
                #predicted_probs = F.softmax(output, dim=1)[:, 1]  # Probability of class 1 (positive)
                predicted_probs = torch.sigmoid(output)
                predicted_class = (predicted_probs >= threshold).type(torch.long)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted_class.cpu().numpy())
                all_val_probs.extend(predicted_probs.cpu().numpy())

        #tune.report({"loss": val_loss / len(val_loader)})
        
        # Calculate validation evaluation metrics
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_preds)
        val_confusion_matrix = confusion_matrix(all_val_labels, all_val_preds)
        tn, fp, fn, tp = confusion_matrix(all_val_labels, all_val_preds).ravel()

        # Compute sensitivity (recall) and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        
        avg_val_loss = val_loss/len(val_loader)

        # Print val results every 10th epoch
        if epoch % 1 == 0: # Set to 1 for debugging
            print(f'val Loss: {avg_val_loss:.4f}, val Accuracy: {val_accuracy:.4f}, val AUC: {val_auc:.4f}', f'Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}', f'MCC: {mcc:.4f}')
            print('val Confusion Matrix:')
            print(val_confusion_matrix)
        
        # Append val loss to val losses list
        val_losses.append(avg_val_loss)


        # If validation loss is lowest so far save the model weights and corresponding hyperparameters
        if avg_val_loss < min_val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{avg_val_loss:.6f}) \t Saving the model...')
            model_path = f"{model_checkpoints_path}/classifier_model_weights_best_E.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved! Best epoch yet: {epoch + 1}")
            print(f"Current hyperparameters:\n{hyperparams} \t Writing hyperparameter values to file...")

            # Save current hyperparameter values 
            with open(f"{model_checkpoints_path}/classifier_hyperparams_best_E.txt", "w") as file: 
                file.write(json.dumps(hyperparams))
            
            # Reset min validation loss as current validation loss
            min_val_loss = avg_val_loss

            # Plot evaluation metrics
            print("Plotting evaluation metrics...")
            prob_true, prob_pred = calibration_curve(all_val_labels, all_val_probs, n_bins=10)
            fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
            roc_auc = auc(fpr, tpr)

            plot_calibration_curve(prob_pred, prob_true, plot_save_dir)
            plot_pred_prob_dist(prob_pred, plot_save_dir)
            plot_roc_curve(fpr, tpr, roc_auc, plot_save_dir)
            plot_losses(train_losses, val_losses, plot_save_dir)
            
            print(f"Plots saved in folder {plot_save_dir}!")


        # # Plots of final evaluation metrics
        # if epoch == num_epochs-1: #or epoch % 25 == 0 
        #     prob_true, prob_pred = calibration_curve(all_val_labels, all_val_probs, n_bins=10)
        #     fpr, tpr, _ = roc_curve(all_val_labels, all_val_probs)
        #     roc_auc = auc(fpr, tpr)

        #     plot_calibration_curve(prob_pred, prob_true, plot_save_dir)
        #     plot_pred_prob_dist(prob_pred, plot_save_dir)
        #     plot_roc_curve(fpr, tpr, roc_auc, plot_save_dir)
        #     plot_losses(train_losses, val_losses, plot_save_dir)
            

    return {"loss": avg_val_loss/len(val_loader)}





debug_print("\n\n############ Training model #############")

# Single set of hyperparameter values
# train_model(DATA_DIR, train_paths, val_paths, hyperparams)

# Hyperparameter sweep
algo = HyperOptSearch()

tuner = tune.Tuner(  # ③
    train_model,
    tune_config=tune.TuneConfig(
        metric="loss",
        mode="min",
        search_alg=algo,
        num_samples=30,  # Number of trials to run
        #num_samples=20,  # Number of trials to run
        trial_dirname_creator=custom_dirname_creator,
        max_concurrent_trials=1
    ),
    param_space=hyperparams,
)

results = tuner.fit()