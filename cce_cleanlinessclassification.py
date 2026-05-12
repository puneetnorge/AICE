# -*- coding: utf-8 -*-
""" The Code is developed as part of the AICE project (https://aiceproject.eu/)
Puneet Sharma and Kristian Dalsbø Hindberg @ UiT-The Arctic University of Norway
AICE is a four-year project running from September 2022 to August 2026. This project has received funding from Horizon Europe (grant agreement no. 101057400).
The data used in the code & paper belongs to the AICE project
"""

# parameters
NO_OF_FOLDS = 10
VALIDATION_FRACTION = 0.15
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
BATCH_SIZE = 16
# Define the pruning fractions explicitly
PRUNE_FRACTIONS = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]
NO_OF_PRUNING_ITERATIONS = len(PRUNE_FRACTIONS)+1

# Place where we store the results and models
RESULTS_FOLDER = '/PATH'

# Normalization of Images
IMG_MEAN_PER_CHANNEL = [0.55709905, 0.38253729, 0.23618910]
IMG_STD_DEV_PER_CHANNEL  = [0.18622870, 0.15077082, 0.12357164]

"""
    CCE Quality poor --> 0, fair --> 1, good --> 2 and excellent --> 3,

There are 14 raters for each fo the 500 images.

"""

# Include all the libraries here
import os
from PIL import Image
from glob import glob
from os.path import exists
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import ConcatDataset
from torchvision import transforms, models
from sklearn.model_selection import KFold
import timm
from PIL import Image
from collections import Counter
from torch.optim import lr_scheduler
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.model_selection import train_test_split
from torchvision import transforms
import itertools
from scipy import stats as st
import sys
import math


# Move tensor to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data scaling
def rescale_to_range(data, new_min, new_max):
    data_min = min(data)
    data_max = max(data)
     # Avoid division by zero in case data_min == data_max
    if data_min == data_max:
        print('Wrong values in Input')
        return 0
    scaled_data = [(new_max - new_min) * (x - data_min) / (data_max - data_min) + new_min for x in data]
    return scaled_data


# Set random seed
random.seed(64)

# Define main folder
fn_main = "/Cleanliness/"

# Extract list of image paths and image numbers
im_list = sorted(glob(f'{fn_main}500 Images/[0-9]*.jpg'))
im_num = [os.path.basename(fn)[0:-4] for fn in im_list]

# Load cleanliness data - Leighton-Rex class
cleany_cat = pd.read_csv(f'{fn_main}LeightonRexClass.txt', sep = "\t", dtype=str)
cleany_cat.set_index('Image', inplace=True)

median_perc = []
clean_catVal = []

for i, immy in enumerate(cleany_cat.index):



    cat_curr = cleany_cat.iloc[i].values

    cat_vals = []
    '''  Categorize quality from 0 to 3, where 0 is Poor and 2 is Excellent '''
    for catty in cat_curr:
        if catty =="Poor" :
            cat_vals.append(0)
        if catty =="Fair":
            cat_vals.append(1)
        if catty =="Good":
            cat_vals.append(2)
        if catty =="Excellent":
            cat_vals.append(3)
    ''' Use the mode value '''
    q_val = st.mode(cat_vals)[0]
    clean_catVal.append(q_val)

X = []
# Make dataframe
X = pd.DataFrame({'file_paths': im_list, 'im_num': im_num, 'CleanCat': clean_catVal})


# Shuffle the DataFrame
X = X.sample(frac=1).reset_index(drop=True)
del catty, cat_vals, cat_curr

import torch

def get_pruned_model_for_inference(checkpoint_path: str):
    """
    Load a pruned model checkpoint, extract only the final pruned weights,
    and return a model ready for inference.

    Args:
        checkpoint_path (str): Path to the pruned model checkpoint.

    Returns:
        dict: State dictionary of the pruned model ready for inference.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state_dict
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    # Create a new state_dict with only final pruned weights
    final_state_dict = {}

    for key in state_dict:
        if "orig" in key:  # Structured pruning uses "orig" and "mask"
            base_key = key.replace("_orig", "")  # Get base parameter name
            mask_key = key.replace("orig", "mask")  # Get corresponding mask key

            if mask_key in state_dict:
                pruned_weight = state_dict[key] * state_dict[mask_key]  # Apply structured mask
                final_state_dict[base_key] = pruned_weight
            else:
                final_state_dict[base_key] = state_dict[key]  # If no mask, keep original weights

        elif "mask" not in key:  # Ignore mask keys
            final_state_dict[key] = state_dict[key]

    return final_state_dict

""" Count the number of samples available per class """
def Count_No_of_Samples_Per_Class(dataset):
  ''' Find and Count the number of unique samples in the dataset and print the results'''
  ## Collect all labels
  cat_labels = [la for im, la in dataset]
  # Print the number of unique labels
  cat_labels = np.array(cat_labels)

  # Find unique labels and their counts
  unique_labels, counts = np.unique(cat_labels, return_counts=True)
  # Print the unique labels and their counts
  for label, count in zip(unique_labels, counts):
      print(f"Label '{label}': {count} elements")
  return unique_labels, counts

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision import transforms
# Prepare the dataset by loading images from df and transforming to tensors
# Transforms for ordinary images of the dataset
""" Unnormalize a tensor image with mean and standard deviation. """
def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN_PER_CHANNEL, std=IMG_STD_DEV_PER_CHANNEL)
])
''' Map the data in the dataframe to image and labels for Tensor '''
class CustomImageDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        # Select the first column of the dataframe as image name
        img = Image.open(self.dataframe.iloc[idx, 0]).convert("RGB")
        # Select the 3rd column of the dataframe as labels (3rd is CleanCat)
        label = self.dataframe.iloc[idx, 2]  # 3rd column for CleanCat [MAIN CATEGORY]
        img = self.transform(img)
        return img, label
# Define transformations for data augmentation
augmentations_minorityclass = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Large changes can produce strange-looking images
    transforms.Resize((224, 224)),  # Resized to 224 by 224
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.1, 0.2), ratio=(1, 1)),
    transforms.Normalize(mean=IMG_MEAN_PER_CHANNEL, std=IMG_STD_DEV_PER_CHANNEL)
])
''' Class and methods for augmenting the dataset '''
class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, minority_class, max_no_augmented_samples):
        self.original_dataset = original_dataset
        self.minority_class = minority_class
        self.max_no_augmented_samples = max_no_augmented_samples
        self.augmented_data = []
        self.create_augmented_data()
    def create_augmented_data(self):
        count = 0
        # Iterate through the dataset to generate augmented data
        for img, label in self.original_dataset:
            if label == self.minority_class:
                count += 1
                if count <= self.max_no_augmented_samples:
                    # Apply unnormalize function
                    unnormalized_tensor = unnormalize(img.clone(), mean=IMG_MEAN_PER_CHANNEL, std=IMG_STD_DEV_PER_CHANNEL)
                    pil_image = transforms.ToPILImage()(unnormalized_tensor)
                    augmented_img = augmentations_minorityclass(pil_image)
                    self.augmented_data.append((augmented_img, label))
                else:
                    # Exit the loop once the required number of samples is generated
                    break
    def __len__(self):
        return len(self.original_dataset) + len(self.augmented_data)
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            # Corrected the index calculation for augmented data
            return self.augmented_data[idx - len(self.original_dataset)]
""" Display a batch of images using matplotlib. """
def show_images_in_batch(images_batch, labels1_batch, num_columns=2):
    """Display a batch of grayscale images using matplotlib."""
    if images_batch.ndimension() == 4 and images_batch.size(1) == 1:
        images_batch = images_batch.squeeze(1)  # Remove the channel dimension if it's single-channel
    elif images_batch.ndimension() == 3:
        pass  # Already a 3D tensor (grayscale)
    num_images = images_batch.size(0)
    num_rows = (num_images + num_columns - 1) // num_columns  # Calculate number of rows
    # Convert tensor images to numpy arrays
    images_np = [img.cpu().numpy() for img in images_batch]
    images_labels = []
    ''' Convert each tensor of labels to a numpy array and get the corresponding labels '''
    for labels in labels1_batch:
        images_labels.append(labels)  # Append the result to images_labels
    # Create a matplotlib figure
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))
    axes = axes.flatten()
    # Display images
    for i in range(num_images):
        axes[i].imshow(images_np[i], cmap='gray')
        # Get the original labels using get_label
        axes[i].set_title(images_labels[i])
        axes[i].axis('off')  # Hide the axis
    # Remove any unused axes
    for ax in axes[num_images:]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()
# Get the dataset from X
orig_dataset = CustomImageDataset(X)
def round_up_to_nearest_10(n):
    return math.ceil(n / 10) * 10
''' Split the dataset into training and testing sets e.g., 85% test and rest for val '''
def Data_Split_And_Augment(orig_dataset, train_indices, val_indices):
    # Create subsets for training and testing
    train_dataset = Subset(orig_dataset, train_indices)
    test_dataset = Subset(orig_dataset, val_indices)
    print(' ***---------Test dataset-------***')
    unique_labels = []
    counts = []
    [unique_labels, counts] = Count_No_of_Samples_Per_Class(test_dataset)
    print(' ***---------Train dataset-------***')
    unique_labels = []
    counts = []
    [unique_labels, counts] = Count_No_of_Samples_Per_Class(train_dataset)
    augmented_train_dataset = train_dataset
    for label, count in zip(unique_labels, counts):
        # For each label calculate the number of samples to augment
        no_of_augmented_samples = round_up_to_nearest_10(max(counts) + 20) - count
        print(f'No of elements to augment = {no_of_augmented_samples}')
        augmented_train_dataset = AugmentedDataset(augmented_train_dataset, minority_class=label, max_no_augmented_samples=no_of_augmented_samples)
    print(' ***---------Augmented Train dataset-------***')
    unique_labels = []
    counts = []
    [unique_labels, counts] = Count_No_of_Samples_Per_Class(augmented_train_dataset)
    return augmented_train_dataset, test_dataset

import torch
import torch.nn as nn
import timm  # If timm is used for creating models

class ResNetCustom(nn.Module):
    def __init__(self, base_model_name, num_classes_category, dropout_rate):
        super(ResNetCustom, self).__init__()

        # Load the pre-trained model, remove the classifier
        self.base_model = timm.create_model(base_model_name, pretrained=True, num_classes=0)  # Set num_classes=0 to remove the original classifier

        # Automatically get the number of output features from the base model
        num_features = self.base_model.num_features

        # Add an adaptive pooling layer and a fully connected layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Optional Dropout layer to help prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Final fully connected layer for classification
        self.fc = nn.Linear(num_features, num_classes_category)

    def forward(self, x):
        # Forward pass through the base model (extracting features)
        x = self.base_model.forward_features(x)  # Assumes timm model supports forward_features

        # Apply the pooling layer
        x = self.pool(x)

        # Flatten the pooled output
        x = torch.flatten(x, 1)

        # Apply dropout before the final classification layer
        x = self.dropout(x)

        # Final output through fully connected layer
        out = self.fc(x)

        return out


# Load a pre-trained ResNet model and customize it for a 4-class classification problem

num_classes_category = 4
dropout_rate = 0.5  # Custom dropout rate

model = ResNetCustom(base_model_name='resnet18', num_classes_category = num_classes_category, dropout_rate = dropout_rate)

"""Show a batch of images for visualization"""

import cv2
''' imshow method'''
def imshow(img, **kwargs):
  img = np.array(img)
  if img.shape[0] == 3:
    img = img.transpose(1, 2, 0)


  img -= img.min()
  img /= img.max()
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Assuming image is in BGR format, convert to RGB
  plt.imshow(img, **kwargs); plt.axis('off')

import copy
import torch.nn.utils as utils
import pickle

criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for classification

''' Training and Validation with Early Stopping '''
def training_and_validation(model, fold_no, train_loader, val_loader, optimizer, num_epochs, pruning_iteration):
      # early stopping criterion for each fold
      early_stopping = EarlyStopping(patience=5, delta=0.0001, fold_number = fold_no, prun_iter = pruning_iteration)
      train_loss = []
      val_loss = []
      train_acc = []
      val_acc = []

      for epoch in range(num_epochs):

        model.train()
        run_training_loss = 0.0

        tr_correct = 0
        tr_total= 0

        for inputs, labels  in train_loader:

            inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU if available
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            t_loss = criterion(outputs,labels)
            t_loss.backward()

            optimizer.step()
            run_training_loss += t_loss.item()
            # Record training accuracy
            _, predicted = torch.max(outputs, 1)
            tr_total += labels.size(0)
            tr_correct += (predicted == labels).sum().item()

        train_loss.append(run_training_loss / tr_total)
        train_acc.append(100 * tr_correct / tr_total)
        print(f' Epoch {epoch+1},  Training Loss: {train_loss[-1]:.4f}, Training Accuracy: {train_acc[-1]:.2f}%')

        # Validation loop per epoch
        model.eval()
        va_correct = 0
        va_total = 0
        run_validation_loss = 0.0

        ''' validation metrics '''

        with torch.no_grad():
          for inputs, labels in val_loader:
              inputs, labels = inputs.cuda(), labels.cuda()  # Move to GPU if available
              outputs = model(inputs)

              v_loss = criterion(outputs, labels)
              run_validation_loss += v_loss.item()

              # Record validation accuracy
              _, val_predicted = torch.max(outputs, 1)
              va_total += labels.size(0)
              va_correct += (val_predicted == labels).sum().item()
        # Calculate average validation loss
        avg_val_loss = run_validation_loss / va_total
        val_loss.append(avg_val_loss)
        avg_val_acc = 100 * va_correct / va_total
        val_acc.append(avg_val_acc)

        print(f'Epoch {epoch+1}, Validation Loss: {val_loss[-1]:.4f}, Validation Accuracy: {val_acc[-1]:.2f}%')
        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
          print("Early stopping")
          break

      # Plotting the loss and accuracy metrics
      plt.figure(figsize=(12, 5))

      # Plot loss
      plt.subplot(1, 2, 1)
      plt.plot(train_loss, label='Training Loss')
      plt.plot(val_loss, label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.legend()
      plt.grid()
      plt.title('Loss vs. Epochs')

      # Plot accuracy
      plt.subplot(1, 2, 2)
      plt.plot(train_acc, label='Training Accuracy')
      plt.plot(val_acc, label='Validation Accuracy')
      plt.xlabel('Epoch')
      plt.ylabel('Accuracy')
      plt.legend()
      plt.title('Accuracy vs. Epochs')
      plt.grid()
      plt.show()

      return train_loss, val_loss, train_acc, val_acc

'''Confusion matrix calculations'''
def calculate_confusion_matrix(model, inference_loader, fold_number, prun_iter):
  # For Confusion matrix calculations
  qa_preds = []
  qa_labels = []

  checkpoint_name = f"Prun_iter_{prun_iter}_fold_no_{fold_number}_checkpoint.pth"
  '''load the weights from saved checkpoint '''
  model.load_state_dict(torch.load(checkpoint_name, weights_only=True))

  with torch.no_grad():  # Disable gradient calculation for inference
    for res_inputs, res_labels in inference_loader:
            res_inputs, res_labels = res_inputs.cuda(), res_labels.cuda()  # Move to GPU if available
            res_outputs = model(res_inputs)
            _, res_pred = torch.max(res_outputs, 1)

            # Collect predictions and labels for confusion matrix
            qa_preds = res_pred.cpu().numpy()
            qa_labels = res_labels.cpu().numpy()



  print(qa_preds.size)
  # Compute the confusion matrix
  num_classes_qa = 4


  class_names = ['Poor','Fair','Good','Excellent']  # List of class names

  cm_qa = confusion_matrix(qa_preds, qa_labels, num_classes_qa)
  plot_confusion_matrix(cm_qa, class_names,'Quality')
  return cm_qa

def cross_validate_then_prune_iteratively(orig_dataset, val_perc, k_folds, num_epochs, learning_rate, num_pruning_iterations, dropout_rate):
    # Define the k-fold Stratified cross-validator
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=None)
    results_list = [] # list
    sparsity_per_iteration = []  # List to track sparsity metrics across iterations
    labels = [la for im, la in orig_dataset]
    imgs = [im for im, la in orig_dataset]
    for fold, (train_idx, val_idx) in enumerate(kfold.split(imgs, labels)):
        # Initialize model for each fold
        fold_model = ResNetCustom(base_model_name='resnet18', num_classes_category=num_classes_category, dropout_rate=dropout_rate)
        for iteration in range(num_pruning_iterations):
            print(f'Fold Number = {fold}, Pruning Iteration = {iteration}')

            # Initialize sparsity tracking for this fold and iteration
            overall_sparsity_per_fold = []
            layer_sparsity_per_fold = []
            if iteration == 0:
                print('First iteration: no pruning, use the original model')
                fold_model.load_state_dict(model.state_dict())
            else:
                # Load the checkpoint from the previous pruning iteration
                checkpoint_path = f'Prun_iter_{iteration - 1}_fold_no_{fold}_checkpoint.pth'
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                fold_model.load_state_dict(checkpoint)
                print(f'Loaded model from checkpoint: {checkpoint_path}')

                # Apply structured pruning
                prune_fraction = PRUNE_FRACTIONS[iteration - 1]
                fold_model = structured_pruning(fold_model, prune_fraction=prune_fraction)

            fold_model = fold_model.to(device)  # Move to GPU if available
            [train_dataset, val_dataset] = Data_Split_And_Augment(orig_dataset, train_idx, val_idx)
            # Define data loaders for training and validation
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            inference_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=2)
            # Define optimizer
            optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate, weight_decay=0.0005)
            # Train and validate the model
            train_loss, val_loss, train_acc, val_acc = training_and_validation(
                fold_model, fold, train_loader, val_loader, optimizer, num_epochs, iteration
            )
            # Calculate confusion matrix
            cm_qa = calculate_confusion_matrix(fold_model, inference_loader, fold, iteration)
            # Save the model checkpoint for the next pruning iteration
            checkpoint_save_path = f'Prun_iter_{iteration}_fold_no_{fold}_checkpoint.pth'
            torch.save(fold_model.state_dict(), checkpoint_save_path)
            print(f'Model checkpoint saved to: {checkpoint_save_path}')
            ## Also save the model for inference (without the mask)
            model_state_dict = get_pruned_model_for_inference(checkpoint_save_path)
            model_name = []
            model_name = checkpoint_save_path
            model_name_inf = model_name.replace(".pth", "_inf.pth")  # Add _inf before .
            print(f'Model checkpoint saved to: {model_name_inf}')
            # Save the model state dictionary (pruned and masks removed)
            torch.save(model_state_dict, model_name_inf)
            # Calculate sparsity
            overall_sparsity, layer_sparsity = calculate_sparsity(fold_model)
            # Store results for this fold and iteration in the list
            results_list.append({
                "fold": fold,
                "iteration": iteration,
                "val_loss": min(val_loss),
                "train_loss": min(train_loss),
                "val_acc": max(val_acc),
                "train_acc": max(train_acc),
                "confusion_matrix": cm_qa,
                "overall_sparsity": overall_sparsity,
                "layer_sparsity": layer_sparsity,
                "model_name": checkpoint_save_path
            })

            # Print confirmation of results storage
            print(f"Stored results for Fold {fold}, Iteration {iteration}: {results_list[-1]}")
    # Save the results list to a .pkl file
    with open('results_list.pkl', 'wb') as pkl_file:
        pickle.dump(results_list, pkl_file)
    print(f"Results saved to {'results_list.pkl'}")

# Function to calculate sparsity
def calculate_sparsity(model):
    total_params = 0
    total_zeros = 0

    # Calculate sparsity for each layer after pruning
    layer_sparsity = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):  # Typically interested in Conv2d or Linear layers
            # Access the weight parameter
            weight = module.weight.data
            num_elements = weight.numel()  # Total number of elements in the weight tensor
            num_zeros = torch.sum(weight == 0).item()  # Number of zero elements



            # Calculate sparsity for this layer
            sparsity = 100. *num_zeros / num_elements
             # Append the layer name and sparsity to the list for this iteration
            layer_sparsity.append((name, sparsity))

            # Update total counts
            total_params += num_elements
            total_zeros += num_zeros

    # Overall sparsity
    overall_sparsity = 100. *total_zeros / total_params
    print(f"Overall Sparsity: {overall_sparsity:.4f} ({total_zeros}/{total_params})")
    return overall_sparsity, layer_sparsity
overall_sparsity, layer_sparsity = calculate_sparsity(model)
print(layer_sparsity)

'''  Confusion matrix for different classes or categories'''
def confusion_matrix(preds, labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    # zip() method takes iterable containers and returns a single iterator object
    for t, p in zip(labels, preds):
        matrix[t, p] += 1
    return matrix


# Visualize the confusion matrix
def plot_confusion_matrix(cm, class_names,title_text):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title_text)
    plt.show()

class EarlyStopping:
    """
    Initializes the early stopping mechanism.

    Args:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
        fold_number (int): Current fold number for checkpoint naming (cross-validation).
        prun_iter (int): Current pruning iteration number for checkpoint naming.
    """
    def __init__(self, patience, delta, fold_number, prun_iter):
        self.patience = patience
        self.delta = delta
        self.fold_number = fold_number
        self.prun_iter = prun_iter
        self.best_score = None  # Tracks the best loss (lower is better)
        self.early_stop = False  # Flag for early stopping
        self.counter = 0  # Counter for epochs without improvement
        self.best_loss = np.inf  # Best loss starts as infinity

    def __call__(self, val_loss, model):
        """
        Check if training should be stopped early based on validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        score = -val_loss  # Loss is used as the score (lower is better, so we negate it)

        # If this is the first epoch or loss improved significantly
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            # No significant improvement (loss hasn't decreased enough)
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement in loss
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset the counter if loss improves

    def save_checkpoint(self, val_loss, model):
        """
        Save the model when validation loss improves.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (torch.nn.Module): Model to save if validation loss improves.
        """
        if val_loss < self.best_loss:  # We save if loss decreased
            self.best_loss = val_loss
            ''' Save the checkpoint with Pruning iteration number and fold number '''
            checkpoint_name = f"Prun_iter_{self.prun_iter}_fold_no_{self.fold_number}_checkpoint.pth"
            torch.save(model.state_dict(), checkpoint_name)
            print(f'Model saved as {checkpoint_name}')

def best_acc_model(results):
  best_acc = []
  best_index = []
  # Extract accuracies into a list
  acc_val = [results[i][1] for i in range(len(results))]

  # Find the index of the best accuracy
  best_index = acc_val.index(max(acc_val))

  # Best accuracy value
  best_acc = acc_val[best_index]

  print(f'Best accuracy = {best_acc}')
  print(f'Best index = {best_index}')

  return best_acc, best_index

import torch.nn.utils.prune as prune

def structured_pruning(model, prune_fraction):

  # Iterate through the named modules in the model
  for name, module in model.named_modules():
      # Check if the module is a Conv2d and skip the first conv layer
      if isinstance(module, nn.Conv2d) and name != "conv1":

          # Apply structured pruning on X percentage of the weights based on L1 norm
          # Use dim=0 for pruning output channels
          prune.ln_structured(module, name="weight", amount = prune_fraction, n=1, dim=0)



  return model

def remove_pruning(model):

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if hasattr(module, 'weight_orig'):  # Check if pruning exists
                prune.remove(module, 'weight')  # Remove pruning reparameterization

import csv
import time
t = time.time()
cross_validate_then_prune_iteratively(orig_dataset=orig_dataset,
                                      val_perc = VALIDATION_FRACTION,
                                      k_folds = NO_OF_FOLDS,
                                      num_epochs = MAX_EPOCHS,
                                      learning_rate = LEARNING_RATE,
                                      num_pruning_iterations=NO_OF_PRUNING_ITERATIONS,
                                      dropout_rate= 0.40)

elapsed = time.time() - t
print(elapsed)