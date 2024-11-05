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
from model.actor_critic import ActorCritic

# ===========================
# 6. PPO Components
# ===========================
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        
class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr_actor, lr_critic, gamma, epochs, batch_size, device="cpu", checkpoint_dir=None):
        self.device = device
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, action_std, device).to(self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.best_val_profit = -np.inf  # Initialize best validation profit
        self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, "last_model.pth")
    
    def select_action(self, state, store=True):
        state = torch.FloatTensor(state).to(self.device)
        action, logprob, state_value = self.policy.act(state)
        if store:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(logprob)
            self.buffer.state_values.append(state_value)
        return action.cpu().numpy()

    def update(self, current_val_profit):
        if len(self.buffer.rewards) == 0:
            logging.warning("Rollout buffer is empty, skipping update.")
            return

        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Convert list to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert list to tensor
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()

        # Evaluate actions
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        
        # Calculate advantages
        advantages = rewards - state_values.squeeze().detach()
        
        # PPO loss
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values.squeeze(), rewards) - 0.01 * dist_entropy.mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Gradient Clipping
        self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()

        # Save last model
        if self.checkpoint_dir:
            self.save_checkpoint(self.last_checkpoint_path, is_best=False)
            logging.info(f"Last model saved at {self.last_checkpoint_path}")

        # Save best model if current_val_profit is better
        if current_val_profit > self.best_val_profit:
            self.best_val_profit = current_val_profit
            self.save_checkpoint(self.best_checkpoint_path, is_best=True)
            logging.info(f"Best model updated and saved at {self.best_checkpoint_path}")

    def save_checkpoint(self, filepath, is_best=False):
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            checkpoint = {
                'state_dict': self.policy.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, filepath)
            if is_best:
                logging.info(f"Best checkpoint saved at {filepath}")
            else:
                logging.info(f"Last checkpoint saved at {filepath}")

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f"Checkpoint loaded from {checkpoint_path}")
        else:
            logging.error(f"Checkpoint file {checkpoint_path} does not exist.")
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")