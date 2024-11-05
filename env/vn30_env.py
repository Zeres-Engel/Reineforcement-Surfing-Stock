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
# 5. Environment Definition
# ===========================
class TradingEnv:
    def __init__(self, data, features_dim, action_dim, initial_balance, transaction_fee=0.001):
        self.data = data.drop(columns=['time'], errors='ignore').reset_index(drop=True)
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.total_profit = 0
        self.positions = 0  # 0: not holding, 1: holding
        self.buy_price = 0
        state = self._get_state()
        return state

    def _get_state(self):
        if self.current_step >= len(self.data):
            logging.error("Attempted to access data beyond available range.")
            raise IndexError("single positional indexer is out-of-bounds")
        state = self.data.iloc[self.current_step].values.astype(np.float32)
        state = np.nan_to_num(state)
        return state

    def step(self, action):
        done = False
        reward = 0
        info = {}
        
        # Get current and next close prices for reward calculation
        current_close = self.data['close'].iloc[self.current_step]
        if self.current_step < len(self.data) - 1:
            next_close = self.data['close'].iloc[self.current_step + 1]
        else:
            next_close = current_close  # If last step
        
        # Action: Buy (action > 0.1), Sell (action < -0.1), Hold (action = 0)
        action = action[0]
        if action > 0.1 and self.positions == 0:
            # Buy
            self.positions = 1
            self.buy_price = current_close
            self.balance -= current_close * (1 + self.transaction_fee)
            logging.info(f"Bought at price {current_close:.2f}")
        elif action < -0.1 and self.positions == 1:
            # Sell
            self.positions = 0
            profit = (current_close - self.buy_price) * (1 - self.transaction_fee)
            self.total_profit += profit
            self.balance += current_close * (1 - self.transaction_fee)
            reward = profit
            logging.info(f"Sold at price {current_close:.2f}, Profit: {profit:.2f}")
        
        # Calculate reward based on profit
        if self.positions == 1:
            reward = (next_close - current_close)  # No scaling factor
        
        # Check for unreasonable reward values
        if reward < -100 or reward > 100:  # Limit reward
            logging.warning(f"Unusual reward: {reward}")
            reward = 0
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        next_state = self._get_state() if not done else np.zeros(self.features_dim, dtype=np.float32)
        return next_state, reward, done, info
