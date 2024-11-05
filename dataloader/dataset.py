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
# 4. Dataset and Preprocessing
# ===========================
class Dataset:
    def __init__(self, config):
        self.config = config
        # Load CSV and parse only relevant columns
        columns_to_use = ['time'] + config['data']['features']
        self.data = pd.read_csv(config['data']['data_path'], usecols=columns_to_use, parse_dates=["time"])
        
        # Check and verify time column
        if 'time' not in self.data.columns:
            raise KeyError("The 'time' column is missing in the dataset. Please check the CSV file format.")
        
        # Display columns for verification
        logging.info(f"Columns in dataset: {self.data.columns.tolist()}")
        
        # Ensure time column is correctly processed
        self.data['time'] = pd.to_datetime(self.data['time'], dayfirst=True, errors='coerce')
        if self.data['time'].isnull().any():
            logging.warning("Some 'time' entries could not be parsed and will be dropped.")
            self.data = self.data.dropna(subset=['time'])

    def aggregate_to_daily(self, df):
        """
        Aggregate data to daily frequency.
        """
        df = df.set_index('time')
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna().reset_index()
        return daily_df

    def load_data(self, start_date, end_date):
        # Filter data by start and end date
        mask = (self.data['time'] >= pd.to_datetime(start_date)) & (self.data['time'] <= pd.to_datetime(end_date))
        filtered_data = self.data.loc[mask].reset_index(drop=True)
        if filtered_data.empty:
            logging.error(f"No data found between {start_date} and {end_date}. Please check your date ranges.")
            raise ValueError(f"No data found between {start_date} and {end_date}.")
        # Aggregate data to daily
        daily_data = self.aggregate_to_daily(filtered_data)
        logging.info(f"Aggregated data from {start_date} to {end_date}: {daily_data.shape[0]} days.")
        return daily_data