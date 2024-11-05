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

from dataloader.dataset import Dataset
from dataloader.preprocessing import Preprocessing
from env.vn30_env import TradingEnv
from model.ppo import PPO
from utils.plot import plot_train_val_profits, plot_validation_accuracy, plot_validation_f1, plot_test_profit

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
