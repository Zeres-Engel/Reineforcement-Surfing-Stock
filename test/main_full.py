# main.py
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
# 0. Reset stdout and stderr to support utf-8
# ===========================
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# ===========================
# 1. Argument Parser
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate stock prediction model using PPO")
    parser.add_argument("--config", type=str, default="configs/FPT.yaml", help="Path to the config file (YAML format)")
    return parser.parse_args()

# ===========================
# 2. Load Configuration
# ===========================
def load_config(config_path):
    with open(config_path, "r", encoding='utf-8') as file:
        return yaml.safe_load(file)

# ===========================
# 3. Setup Logging
# ===========================
def setup_logging(base_log_path, ticket, log_file):
    # Create subdirectory with ticket name
    log_path = os.path.join(base_log_path, ticket)
    os.makedirs(log_path, exist_ok=True)
    
    # Create checkpoints directory inside log_path
    checkpoint_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers in successive runs
    if not logger.handlers:
        # Create FileHandler with utf-8 encoding
        file_handler = logging.FileHandler(os.path.join(log_path, log_file), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create StreamHandler to log to console only warnings and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return log_path, checkpoint_path

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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init=0.5, device="cpu"):
        super(ActorCritic, self).__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.action_var = torch.full((action_dim,), action_std_init**2).to(self.device)

    def act(self, state):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var))
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), self.critic(state).detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag_embed(self.action_var.expand_as(action_mean)))
        action_logprobs = dist.log_prob(action)
        state_values = self.critic(state)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

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

# ===========================
# 8. Training and Evaluation
# ===========================
def main():
    # Parse arguments and load config
    args = parse_args()
    config = load_config(args.config)
    
    # Get mode from config
    mode = config['agent']['mode']
    if mode not in ['train', 'eval']:
        print("Invalid mode. Choose either 'train' or 'eval'.")
        sys.exit(1)
    
    # Get ticket name from config
    ticket = config['data']['ticket']
    
    # Setup logging and checkpoint path
    base_log_path = config['agent']['save']['loggers']
    log_file = "training_log.txt"
    log_path, checkpoint_path = setup_logging(base_log_path, ticket, log_file)
    logging.info(f"Logs will be saved to: {log_path}")
    logging.info(f"Checkpoints will be saved to: {checkpoint_path}")
    
    # Set seed for reproducibility
    seed = config['agent']['train'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize Dataset and Preprocessing
    dataset = Dataset(config)
    preprocessing = Preprocessing()
    
    if mode == 'train':
        # Load and preprocess training data
        try:
            train_data = dataset.load_data(config['data']['start_train_date'], config['data']['end_train_date'])
            train_data = preprocessing.fit_transform(train_data, config['data']['features'])
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        # Load and preprocess validation data
        try:
            val_data = dataset.load_data(config['data']['start_validation_date'], config['data']['end_validation_date'])
            val_data = preprocessing.transform(val_data, config['data']['features'])
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        # Load and preprocess test data (will be used for potential future use)
        try:
            test_data = dataset.load_data(config['data']['start_test_date'], config['data']['end_test_date'])
            test_data = preprocessing.transform(test_data, config['data']['features'])
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        logging.info("Data loaded and normalized.")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Validation data shape: {val_data.shape}")
        logging.info(f"Test data shape: {test_data.shape}")
        
        # Check for NaN values after normalization
        if train_data.isnull().values.any():
            logging.warning("Train data contains NaN values after normalization.")
        if val_data.isnull().values.any():
            logging.warning("Validation data contains NaN values after normalization.")
        if test_data.isnull().values.any():
            logging.warning("Test data contains NaN values after normalization.")
        
        # Initialize environment and PPO model
        env = TradingEnv(
            data=train_data,
            features_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            initial_balance=config['environment']['initial_balance'],
            transaction_fee=config['environment']['transaction_fee']
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppo = PPO(
            state_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            action_std=config['agent']['train']['action_std'],
            lr_actor=config['agent']['train']['lr_actor'],
            lr_critic=config['agent']['train']['lr_critic'],
            gamma=config['agent']['train']['gamma'],
            epochs=config['agent']['train']['K_epochs'],
            batch_size=config['agent']['train']['batch_size'],
            device=device,
            checkpoint_dir=checkpoint_path
        )
        
        # Initialize metrics lists
        train_profits = []
        val_profits = []
        val_accuracies = []
        val_f1_scores = []
        
        # Initialize environment for validation
        env_val = TradingEnv(
            data=val_data,
            features_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            initial_balance=config['environment']['initial_balance'],
            transaction_fee=config['environment']['transaction_fee']
        )
        
        # Path to save scaler
        scaler_path = os.path.join(log_path, "scaler.pkl")
        
        # Training loop with tqdm
        for epoch in tqdm(range(1, config['agent']['train']['episodes'] + 1), desc="Training Progress"):
            state = env.reset()
            done = False
            while not done:
                try:
                    action = ppo.select_action(state, store=True)
                    next_state, reward, done, info = env.step(action)
                    ppo.buffer.rewards.append(reward)
                    ppo.buffer.is_terminals.append(done)
                    state = next_state
                except IndexError as e:
                    logging.error(e)
                    done = True
            # Do not call ppo.update() here with a placeholder
            # Instead, perform validation first and then call ppo.update()
            
            # Ghi lại profit training
            logging.info(f"Epoch {epoch}/{config['agent']['train']['episodes']} completed. Total Profit: {env.total_profit:.2f}")
            train_profits.append(env.total_profit)
            
            # Validation after each epoch
            state_val = env_val.reset()
            done_val = False
            total_profit_val = 0
            all_predictions = []
            all_trues = []
            while not done_val:
                try:
                    action_val = ppo.select_action(state_val, store=False)
                    next_state_val, reward_val, done_val, info_val = env_val.step(action_val)
                    total_profit_val += reward_val
                    # Prediction: 1 for Buy, 0 for Sell/Hold
                    prediction = 1 if action_val[0] > 0 else 0
                    all_predictions.append(prediction)
                    if env_val.current_step < len(val_data):
                        true_label = 1 if val_data['close'].iloc[env_val.current_step] > val_data['close'].iloc[env_val.current_step - 1] else 0
                        all_trues.append(true_label)
                    state_val = next_state_val
                except IndexError as e:
                    logging.error(e)
                    done_val = True
            logging.info(f"Validation after Epoch {epoch}: Total Profit: {total_profit_val:.2f}")
            val_profits.append(total_profit_val)
            
            # Calculate metrics
            if len(all_trues) > 0:
                accuracy = accuracy_score(all_trues, all_predictions)
                f1 = f1_score(all_trues, all_predictions, average='weighted')
                val_accuracies.append(accuracy)
                val_f1_scores.append(f1)
                logging.info(f"Validation Accuracy: {accuracy:.4f}, Validation F1 Score: {f1:.4f}")
            else:
                val_accuracies.append(0)
                val_f1_scores.append(0)
                logging.warning("No true labels available for validation metrics.")
            
            # Now, call PPO.update() with the actual validation profit
            ppo.update(current_val_profit=total_profit_val)
        
        # Lưu scaler sau khi huấn luyện
        preprocessing.save_scaler(scaler_path)
        logging.info(f"Scaler saved at {scaler_path}.")
        
        logging.info("Training completed.")
        
        # Plot Training and Validation Metrics
        plot_train_val_profits(train_profits, val_profits, log_path)
        plot_validation_accuracy(val_accuracies, log_path)
        plot_validation_f1(val_f1_scores, log_path)
    
    elif mode == 'eval':
        # Load and preprocess test data
        try:
            test_data = dataset.load_data(config['data']['start_test_date'], config['data']['end_test_date'])
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        logging.info("Test data loaded.")
        logging.info(f"Test data shape: {test_data.shape}")
        
        # Initialize Preprocessing and load scaler
        preprocessing = Preprocessing()
        scaler_path = os.path.join(log_path, "scaler.pkl")
        try:
            preprocessing.load_scaler(scaler_path)
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)
        
        # Transform test data
        test_data = preprocessing.transform(test_data, config['data']['features'])
        
        # Check for NaN values after normalization
        if test_data.isnull().values.any():
            logging.warning("Test data contains NaN values after normalization.")
        
        # Initialize environment and PPO model
        env_test = TradingEnv(
            data=test_data,
            features_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            initial_balance=config['environment']['initial_balance'],
            transaction_fee=config['environment']['transaction_fee']
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ppo = PPO(
            state_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            action_std=config['agent']['train']['action_std'],  # Use same action_std as during training
            lr_actor=config['agent']['train']['lr_actor'],
            lr_critic=config['agent']['train']['lr_critic'],
            gamma=config['agent']['train']['gamma'],
            epochs=config['agent']['train']['K_epochs'],
            batch_size=config['agent']['train']['batch_size'],
            device=device,
            checkpoint_dir=checkpoint_path
        )
        
        # Load best or last checkpoint
        try:
            # Tải best model nếu tồn tại, nếu không thì tải last model
            if os.path.exists(ppo.best_checkpoint_path):
                checkpoint_to_load = ppo.best_checkpoint_path
                logging.info(f"Loading best model from {checkpoint_to_load}.")
            else:
                checkpoint_to_load = ppo.last_checkpoint_path
                logging.info(f"Loading last model from {checkpoint_to_load}.")
            ppo.load_checkpoint(checkpoint_to_load)
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)
        
        # Evaluation phase
        logging.info("Starting evaluation...")
        state = env_test.reset()
        done = False
        all_rewards = []
        all_predictions = []
        all_trues = []
        test_profits = []
        profit_over_time = []
        cumulative_profit = []
        cum_profit = 0
        actual_prices = []
        predicted_prices = []
        
        while not done:
            try:
                action = ppo.select_action(state, store=False)
                next_state, reward, done, info = env_test.step(action)
                all_rewards.append(reward)
                # Prediction: 1 for Buy, 0 for Sell/Hold
                prediction = 1 if action[0] > 0 else 0
                all_predictions.append(prediction)
                # Record actual price
                actual_price = test_data['close'].iloc[env_test.current_step - 1] if env_test.current_step - 1 < len(test_data) else test_data['close'].iloc[-1]
                actual_prices.append(actual_price)
                # Predicted price (could be next day's price if holding, or current)
                predicted_price = test_data['close'].iloc[env_test.current_step] if env_test.current_step < len(test_data) else test_data['close'].iloc[-1]
                predicted_prices.append(predicted_price)
                # Record profit over time
                cum_profit += reward
                profit_over_time.append(reward)
                cumulative_profit.append(cum_profit)
                if env_test.current_step < len(test_data):
                    true_label = 1 if test_data['close'].iloc[env_test.current_step] > test_data['close'].iloc[env_test.current_step - 1] else 0
                    all_trues.append(true_label)
                state = next_state
            except IndexError as e:
                logging.error(e)
                done = True
        
        # If holding a position at the end, sell it
        if env_test.positions == 1:
            try:
                action = ppo.select_action(state, store=False)
                next_state, reward, done, info = env_test.step(action)
                all_rewards.append(reward)
                logging.info(f"Sold at price {test_data['close'].iloc[-1]:.2f}, Profit: {reward:.2f}")
                # Update profit tracking
                cum_profit += reward
                profit_over_time.append(reward)
                cumulative_profit.append(cum_profit)
                if reward > 0:
                    all_predictions.append(1)
                    all_trues.append(1)
                elif reward < 0:
                    all_predictions.append(0)
                    all_trues.append(0)
            except IndexError as e:
                logging.error(e)
        
        # Calculate total profit and transactions
        total_profit = env_test.total_profit
        # Assuming each buy/sell pair is a transaction
        total_transactions = len([r for r in all_rewards if r != 0]) // 2
        # Count profitable and unprofitable transactions
        profitable_transactions = sum([1 for r in all_rewards if r > 0])
        unprofitable_transactions = sum([1 for r in all_rewards if r < 0])
        
        # Calculate accuracy and F1 score
        if len(all_trues) > 0:
            accuracy = accuracy_score(all_trues, all_predictions)
            f1 = f1_score(all_trues, all_predictions, average='weighted')
            logging.info(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
        else:
            logging.warning("No true labels available for evaluation metrics.")
            accuracy = 0
            f1 = 0
        
        # Log profit and transactions
        logging.info(f"Total Profit: {total_profit:.2f}")
        logging.info(f"Total Transactions: {total_transactions}")
        logging.info(f"Profitable Transactions: {profitable_transactions}")
        logging.info(f"Unprofitable Transactions: {unprofitable_transactions}")
        
        # Log metrics
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")
        
        # Collect test profits per sample
        test_profits = all_rewards
        
        # Plot Test Profit with Accuracy and F1-score
        plot_test_profit(test_profits, accuracy, f1, log_path)
        
        logging.info("Evaluation completed.")

# ===========================
# 9. Entry Point
# ===========================
if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main()
