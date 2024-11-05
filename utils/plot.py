import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import yaml
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import logging
import sys
import codecs
from sklearn.preprocessing import StandardScaler
import glob
import joblib
from datetime import datetime


# ===========================
# 7. Utility Functions
# ===========================
def plot_train_val_profits(train_profits, val_profits, log_path):
    plt.figure(figsize=(10,5))
    plt.plot(train_profits, label='Training Profit', color='blue')
    plt.plot(val_profits, label='Validation Profit', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Total Profit")
    plt.title("Training and Validation Profit Over Epochs")
    plt.legend()
    profit_path = os.path.join(log_path, "train_val_profit.png")
    plt.savefig(profit_path)
    plt.close()
    logging.info(f"Training and Validation profits plotted at {profit_path}.")

def plot_validation_accuracy(val_accuracies, log_path):
    plt.figure(figsize=(10,5))
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.legend()
    acc_path = os.path.join(log_path, "validation_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    logging.info(f"Validation accuracy plotted at {acc_path}.")

def plot_validation_f1(val_f1_scores, log_path):
    plt.figure(figsize=(10,5))
    plt.plot(val_f1_scores, label='Validation F1-Score', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.title("Validation F1-Score Over Epochs")
    plt.legend()
    f1_path = os.path.join(log_path, "validation_f1_score.png")
    plt.savefig(f1_path)
    plt.close()
    logging.info(f"Validation F1-score plotted at {f1_path}.")

def plot_test_profit(test_profits, test_accuracy, test_f1, log_path):
    plt.figure(figsize=(12,6))
    plt.plot(test_profits, label='Test Profit per Sample', color='purple')
    plt.xlabel("Sample")
    plt.ylabel("Profit")
    plt.title("Test Profit Over Samples")
    plt.legend()
    
    # Annotate accuracy and F1-score on the plot
    plt.text(0.95, 0.95, f'Accuracy: {test_accuracy:.4f}\nF1-Score: {test_f1:.4f}', 
             horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    
    profit_path = os.path.join(log_path, "test_profit.png")
    plt.savefig(profit_path)
    plt.close()
    logging.info(f"Test profit plotted at {profit_path}.")

def get_latest_checkpoint(checkpoint_dir):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not checkpoint_files:
        logging.error(f"No checkpoint files found in {checkpoint_dir}.")
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}.")
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint