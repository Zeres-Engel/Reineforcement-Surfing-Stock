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


class Preprocessing:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit_transform(self, df, features):
        df_scaled = df.copy()
        self.scaler.fit(df_scaled[features])
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        # Remove the warning by either adjusting the threshold or commenting out the check
        # Here, we comment out the warning
        # if (df_scaled[features] < -5).any().any() or (df_scaled[features] > 5).any().any():
        #     logging.warning(f"Normalized values exceed reasonable range in columns: {features}")
        return df_scaled
    
    def transform(self, df, features):
        df_scaled = df.copy()
        df_scaled[features] = self.scaler.transform(df_scaled[features])
        # Remove the warning by either adjusting the threshold or commenting out the check
        # Here, we comment out the warning
        # if (df_scaled[features] < -5).any().any() or (df_scaled[features] > 5).any().any():
        #     logging.warning(f"Normalized values exceed reasonable range in columns: {features}")
        return df_scaled
    
    def save_scaler(self, filepath):
        joblib.dump(self.scaler, filepath)
        logging.info(f"Scaler saved at {filepath}.")
    
    def load_scaler(self, filepath):
        self.scaler = joblib.load(filepath)
        logging.info(f"Scaler loaded from {filepath}.")