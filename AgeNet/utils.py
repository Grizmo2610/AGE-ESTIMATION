import os
import json
import random
import re
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def get_latest_epoch(
    root: str = 'models', 
    prefix="model_epoch_", 
    suffix=".pth"
) -> int:
    max_epoch = 0  # Initialize max epoch to 0

    # Iterate through all files in the specified folder
    for filename in os.listdir(root):
        # Check if the filename matches the pattern "model_epoch_{number}.pth"
        match = re.match(fr"{prefix}(\d+){suffix}", filename)
        if match:
            # Extract the epoch number from the filename
            epoch_num = int(match.group(1))
            # Update max_epoch if this epoch number is greater
            if epoch_num > max_epoch:
                max_epoch = epoch_num

    return max_epoch  # Return the highest epoch number found

def save_best_models(
    model: nn.Module,
    val_result: dict[str, float],
    epoch: int,
    save_paths: dict[str, str],
    root: str = "models",
    best_metrics: dict[str, float] | None = None,
) -> bool:

    os.makedirs(root, exist_ok=True)

    if best_metrics is None:
        best_metrics = {}

    updated = False

    for metric, value in val_result.items():

        if metric not in save_paths:
            continue

        save_path = os.path.join(root, save_paths[metric])

        best_val = best_metrics.get(metric, float("inf"))

        if value < best_val:
            best_metrics[metric] = value
            torch.save(model.state_dict(), save_path)
            print(f"Best {metric} model saved at epoch {epoch+1:02d}")
            updated = True

    return updated

def save_epoch_model(
    model: nn.Module,
    epoch: int,
    root: str="models"
):
    # Save the model for the current epoch (regardless of performance)
    path = os.path.join(root, f"model_epoch_{epoch + 1:02d}.pth")
    torch.save(model.state_dict(), path)
    print(f"Model for epoch {epoch + 1:02d} saved.")

def plot_history(
    history: dict[str, dict[str, list]],
    paths: dict = {},
    save: bool = True,
    root: str = "sample"
):
    # Extract training and validation history
    train_history = history['train']
    val_history = history['val']
    
    # Generate epoch indices
    epochs = range(1, len(train_history['loss']) + 1)
    
    # Define full paths for saving plot image and history file
    history_path = os.path.join(root, paths["history"])
    plot_image_path = os.path.join(root, paths["plot_image"])

    n_metrics = len(train_history)
    cols = 2
    rows = math.ceil(n_metrics / cols)    
    
    # Create a figure with subplots for each metric
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    # Loop through each metric (e.g., loss, accuracy)
    for i, k in enumerate(train_history):
        plt.subplot(rows, cols, i + 1)
        plt.plot(epochs, train_history[k], 'bo-', label=f'Train {k.capitalize()}')  # Training curve
        plt.plot(epochs, val_history[k], 'ro-', label=f'Val {k.capitalize()}')      # Validation curve
        plt.xlabel('Epoch')
        plt.ylabel(k.capitalize())
        plt.title(f'{k.capitalize()} over Epochs')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    
    # Save plot and history if requested
    if save:
        plt.savefig(plot_image_path)
    
    # Show the plot
    plt.show()

def init_history() -> dict[str, dict[str, list]]:
    history = {
        'train': {
            'loss': [], 'rmse': [],
            'ordinal_loss': [], 'ordinal_rmse': [],
            'cls_loss': [], 'cls_rmse': [],
            'gender_loss': [], 'gender_acc': []
        },
        'val': {
            'loss': [], 'rmse': [],
            'ordinal_loss': [], 'ordinal_rmse': [],
            'cls_loss': [], 'cls_rmse': [],
            'gender_loss': [], 'gender_acc': []
        }
    }
    return history

def seed_everything(seed):
    # Set fixed seed for reproducibility across libraries
    random.seed(seed)                          # Python random module seed
    np.random.seed(seed)                       # NumPy seed
    torch.manual_seed(seed)                    # PyTorch CPU seed
    torch.cuda.manual_seed(seed)               # PyTorch CUDA seed for single GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch CUDA seed for all GPUs if using multi-GPU
    # Ensure deterministic behavior for CUDA convolution operations
    torch.backends.cudnn.deterministic = True
    # Disable benchmark mode to prevent nondeterministic algorithm selection
    torch.backends.cudnn.benchmark = False

def load_config(path: str):
    with open(path) as f:
        config = json.load(f)
    return config