# model/ppo.py
import numpy as np
import torch
import torch.nn as nn
import os
import logging
from model.actor_critic import ActorCritic

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
        self.best_val_profit = -np.inf

        # Define paths for best and last model checkpoints
        if self.checkpoint_dir:
            self.best_model_dir = os.path.join(self.checkpoint_dir, "best_model")
            self.last_model_dir = os.path.join(self.checkpoint_dir, "last_model")
            os.makedirs(self.best_model_dir, exist_ok=True)
            os.makedirs(self.last_model_dir, exist_ok=True)
            self.best_checkpoint_path = os.path.join(self.best_model_dir, "best_model.pth")
            self.last_checkpoint_path = os.path.join(self.last_model_dir, "last_model.pth")
        else:
            self.best_checkpoint_path = None
            self.last_checkpoint_path = None

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

        # Compute discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Convert buffer lists to tensors
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()

        # Evaluate current log probabilities and state values
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

        # Compute advantages
        advantages = rewards - state_values.squeeze().detach()

        # Compute ratios for PPO clipping
        ratios = torch.exp(logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values.squeeze(), rewards) - 0.01 * dist_entropy.mean()

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Gradient Clipping
        self.optimizer.step()
        
        # Clear the buffer after updating
        self.buffer.clear()

        # Save last checkpoint
        if self.last_checkpoint_path:
            self.save_checkpoint(self.last_checkpoint_path, is_best=False)
            logging.info(f"Last model saved at {self.last_checkpoint_path}")

        # Save best checkpoint if current validation profit is better
        if current_val_profit > self.best_val_profit:
            self.best_val_profit = current_val_profit
            if self.best_checkpoint_path:
                self.save_checkpoint(self.best_checkpoint_path, is_best=True)
                logging.info(f"Best model updated and saved at {self.best_checkpoint_path}")

    def save_checkpoint(self, filepath, is_best=False):
        if filepath:
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
