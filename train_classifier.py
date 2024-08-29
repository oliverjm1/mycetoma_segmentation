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
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as F2
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Rmeoved as plotting done in WandB now
# import matplotlib.pyplot as plt

import glob
from collections import Counter
from PIL import Image

# Import evaluation metrics
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve

# Import DenseNet model
from monai.networks.nets import DenseNet121 #, HighResNet, EfficientNet, ResNet

# # Import ray for hyperparamter tuning
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler # taken from link https://www.geeksforgeeks.org/hyperparameter-tuning-with-ray-tune-in-pytorch/
# from ray.tune.search.hyperopt import HyperOptSearch

import wandb

from src.utils import format_file_paths #, custom_dirname_creator, plot_calibration_curve, plot_pred_prob_dist, plot_roc_curve, plot_losses
from src.datasets import MycetomaDatasetClassifier


os.environ["WANDB__SERVICE_WAIT"] = "300"

# Set debugging
DEBUG = True
DEBUG_PRINT_INTERVAL = 10

def debug_print(debug_statement):
    if DEBUG:
        print(debug_statement)

# Set running environment (True for HPC, False for local)
HPC_FLAG = sys.argv[1]
debug_print(f"HPC_FLAG = {HPC_FLAG}")


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device = {device}")

###############################################################################################
# Set up paths for images and masks 
###############################################################################################

# Set data directory
def define_dataset(hpc=0):
    
    debug_print("\n\n############ Setting up paths to data #############")
    
    debug_print(f"Current directory: {os.getcwd()}")

    # Set data, plots save and model checkpoint paths
    if hpc=="1":
        debug_print("Setting data paths for ARC4...")
        data_dir = "/nobackup/scjb/mycetoma/data"
        plot_save_path = "/home/home02/scjb/mycetoma_segmentation-dev-james/train_stats"
        model_checkpoints_path = "/home/home02/scjb/mycetoma_segmentation/model_saves"
        
    else:
        debug_print("Setting data paths for local machine...")
        data_dir = "C:\\Users\\james\\Documents\\projects\\mycetoma_segmentation\\data"
        plot_save_path = "C:\\Users\\james\\Documents\\projects\\mycetoma_segmentation\\train_stats"
        model_checkpoints_path = "C:\\Users\\james\\Documents\\projects\\mycetoma_segmentation\\model_saves"
          
    # Get the training paths
    train_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(os.path.join(data_dir, "training_dataset", "**", "*"))])
    val_paths = np.array(['.'.join(i.split('.')) for i in glob.glob(os.path.join(data_dir, "validation_dataset", "**", "*"))])

    debug_print(f"Train paths: {train_paths}")

    # Post-processing binary 
    train_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(os.path.join(data_dir, "binary postprocessed", "corrected_masks_and_augmented_postproc_training", "**", "*"))])
    val_seg_paths_bin = np.array(['.'.join(i.split('.')) for i in glob.glob(os.path.join(data_dir, "binary postprocessed", "corrected_masks_and_augmented_postproc_validation", "**", "*"))])

    # Extract just the image paths
    train_img_paths = train_paths[[not 'mask' in i for i in train_paths]]
    val_img_paths = val_paths[[not 'mask' in i for i in val_paths]]
    img_paths = [train_img_paths, val_img_paths]

    # Combine image and segmentation map paths for each patient
    train_paths = format_file_paths(train_seg_paths_bin, train_img_paths, HPC_FLAG)
    val_paths = format_file_paths(val_seg_paths_bin, val_img_paths, HPC_FLAG)

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


# Ray Tune hyperparameter settings
# hyperparams = {
#     "lr": tune.loguniform(1e-5, 1e-3),
#     "batch_size": tune.choice([5, 10, 12, 16]),
#     "weight_decay": tune.loguniform(1e-4,1e-2), 
#     "mask_channel": tune.choice([True, False]),
#     "threshold": tune.uniform(0.4, 0.6),
#     "num_epochs": tune.choice([30]),
#     "seg_path": 0 # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
#     }


# WandB hyperparams settings
sweep_configuration = {
    "method": "bayes",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "Val Loss"},
    "parameters": {
        "lr": {"max": 0.01, "min": 0.0001},
        "batch_size": {"values": [5,10,12,16]},
        "weight_decay": {"values": [0.0001, 0.001, 0.01]},
        "mask_channel": {"values": [True, False]},
        "threshold": {"max": 0.6, "min": 0.4},
        "num_epochs": {"values": [20]},
        "seg_path": {"values": [0]},
    },
}


# Initialize sweep by passing in config.
# Provide a name of the project.

sweep_id = wandb.sweep(sweep=sweep_configuration, project="mycetoma-classifier-bayesian-sweep")


#     "lr": tune.loguniform(1e-5, 1e-3),
#     "batch_size": tune.choice([5, 10, 12, 16]),
#     "weight_decay": tune.loguniform(1e-4,1e-2), 
#     "mask_channel": tune.choice([True, False]),
#     "threshold": tune.uniform(0.4, 0.6),
#     "num_epochs": tune.choice([30]),
#     "seg_path": 0 # 1 = binary postprocessed, 2 = logit output, 3 = multitask binary, 4 = multitask logit
#     }



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

debug_print(f"hyperparams = {sweep_configuration}")
            


###############################################################################################
# Model training
###############################################################################################

def main():

    run_start_time = time.strftime("%Y%m%d_%H%M%S")
    print(f"Run start time = {run_start_time}")

    run = wandb.init()

    data_dir, train_paths, val_paths, plot_save_dir, model_checkpoints_path = define_dataset(hpc=HPC_FLAG)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    debug_print(f"device = {device}")    

    # Set hyperparameter values
    
    # num_epochs = hyperparams["num_epochs"] #config["num_epochs"]
    # lr = hyperparams["lr"]
    # batch_size = hyperparams["batch_size"]
    # weight_decay = hyperparams["weight_decay"]
    # mask_channel = hyperparams["mask_channel"]
    # threshold = hyperparams["threshold"]
    
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    weight_decay = wandb.config.weight_decay
    mask_channel = wandb.config.mask_channel
    threshold = wandb.config.threshold
    num_epochs = wandb.config.num_epochs #config["num_epochs"]
    seg_path = wandb.config.seg_path

    print(f"Current hyperparameter values:\n {wandb.config}")

    print(f"sep_path = {seg_path}\nsep_path type = {type(seg_path)}")
    # seg_path_end = seg_paths[seg_path]


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
    
    debug_print(f"Loading model to device: {device}...") 
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

        debug_print(f"Loading features and labels to device: {device}")

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
        try:
            train_sensitivity = tp / (tp + fn)
        except ZeroDivisionError:
            print("Train senstivity divide by zero error! Setting train sensitivty to zero.")
            train_sensitivity = 0
        
        try:
            train_specificity = tn / (tn + fp)
        except ZeroDivisionError:
            print("Train specifity divide by zero error! Setting train specificity to zero.")
            train_specificity = 0

        try:
            train_mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        except ZeroDivisionError:
            print("Train mcc divide by zero error! Setting train mcc to zero.")
            train_mcc = 0

        if epoch % 1 == 0: # set to 1 for debugging 
            print(f"""
                Epoch [{epoch+1}/{num_epochs}]
                Train Loss: {train_loss/len(train_loader):.4f}
                Train Accuracy: {train_accuracy:.4f}
                Train AUC: {train_auc:.4f}
                Train Sensitivity: {train_sensitivity:.4f}
                Train Specificity: {train_specificity:.4f}
                Train MCC: {train_mcc:.4f}'
            """)
            print(f'Train Confusion Matrix:')
            print(train_confusion_matrix)
        
        avg_train_loss = train_loss/len(train_loader)
        train_losses.append(avg_train_loss)
        
        
    

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

        # # Compute sensitivity (recall) and specificity - now using try/except blocks to handle divide by zero error
        # val_sensitivity = tp / (tp + fn)
        # val_specificity = tn / (tn + fp)
        # val_mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        # Compute sensitivity (recall) and specificity
        try:
            val_sensitivity = tp / (tp + fn)
        except ZeroDivisionError:
            print("Val senstivity divide by zero error! Setting val sensitivty to zero.")
            train_sensitivity = 0
        
        try:
            val_specificity = tn / (tn + fp)
        except ZeroDivisionError:
            print("Val specifity divide by zero error! Setting val specificity to zero.")
            val_specificity = 0

        try:
            val_mcc = ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        except ZeroDivisionError:
            print("Val mcc divide by zero error! Setting val mcc to zero.")
            val_mcc = 0
        
        avg_val_loss = val_loss/len(val_loader)

        # Print val results every 10th epoch
        if epoch % 1 == 0: # Set to 1 for debugging
            print(f'val Loss: {avg_val_loss:.4f}, val Accuracy: {val_accuracy:.4f}, val AUC: {val_auc:.4f}', f'Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}', f'MCC: {val_mcc:.4f}')
            print('val Confusion Matrix:')
            print(val_confusion_matrix)
        
        # Append val loss to val losses list
        val_losses.append(avg_val_loss)


        # If validation loss is lowest so far save the model weights and corresponding hyperparameters
        if avg_val_loss < min_val_loss:
            print(f'Validation Loss Decreased({min_val_loss:.6f}--->{avg_val_loss:.6f}) \t Saving the model...')
            model_path = os.path.join(model_checkpoints_path, f"classifier_model_weights_best_E_{run_start_time}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved! Best epoch yet: {epoch + 1}")
            print(f"Current hyperparameters:\n{wandb.config} \t Writing hyperparameter values to file...")

            # # Save current hyperparameter values 
            # with open(os.path.join(model_checkpoints_path, f"classifier_hyperparams_best_E_{run_start_time}.txt"), "w") as file: 
            #     file.write(wandb.config)
            
            # Reset min validation loss as current validation loss
            min_val_loss = avg_val_loss

            # Calculate evaluation metrics
            print("Calculating evaluation metrics...")
            prob_true, prob_pred = calibration_curve(all_val_labels, all_val_probs, n_bins=10)
            val_fpr, val_tpr, _ = roc_curve(all_val_labels, all_val_probs)
            val_roc_auc = auc(val_fpr, val_tpr)

            # Removed plotting as plotting performed in WandB
            # plot_calibration_curve(prob_pred, prob_true, plot_save_dir)
            # plot_pred_prob_dist(prob_pred, plot_save_dir)
            # plot_roc_curve(val_fpr, val_tpr, val_roc_auc, plot_save_dir)
            # plot_losses(train_losses, val_losses, plot_save_dir)   
            # print(f"Plots saved in folder {plot_save_dir}!")

        # log to wandb
        wandb.log(
            {
                "Train Loss": avg_train_loss,
                "Train Accuracy": train_accuracy,
                "Train AUC": train_auc,
                "Train Sensitivity": train_sensitivity,
                "Train Specificity": train_specificity,
                "Train MCC": train_mcc,
                "Val Loss": avg_val_loss,
                "Val Accuracy": val_accuracy,
                "Val AUC": val_auc,
                "Val Sensitivity": val_sensitivity,
                "Val Specificity": val_specificity,
                "Val MCC": val_mcc
            }
        )
        
        
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


# RAY TUNE - COMMENTED OUT IN FAVOUR OF WANDB
# Set ray tune to not change the working directory

# os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
# print(f"ray tune set to not change working directory: RAY_CHDIR_TO_TRIAL_DIR = {os.environ["RAY_CHDIR_TO_TRIAL_DIR"]}")

# # Hyperparameter sweep
# algo = HyperOptSearch()

# tuner = tune.Tuner(  # ③
#     train_model,
#     tune_config=tune.TuneConfig(
#         metric="loss",
#         mode="min",
#         search_alg=algo,
#         num_samples=30,  # Number of trials to run
#         #num_samples=20,  # Number of trials to run
#         trial_dirname_creator=custom_dirname_creator,
#         max_concurrent_trials=1
#     ),
#     param_space=hyperparams,
# )

# results = tuner.fit()

# Start sweep job.
wandb.agent(sweep_id, function=main, count=20)