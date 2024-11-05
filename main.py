import numpy as np
import torch
import yaml
import os
import logging
import sys
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from dataloader.dataset import Dataset
from dataloader.preprocessing import Preprocessing
from env.vn30_env import TradingEnv
from model.ppo import PPO
from utils.plot import plot_train_val_profits, plot_validation_accuracy, plot_validation_f1, plot_test_profit
from utils.logging import setup_logging
from utils.configs import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train or Evaluate stock prediction model using PPO")
    parser.add_argument("--config", type=str, default="configs/FPT.yaml", help="Path to the config file (YAML format)")
    return parser.parse_args()



def main():
    args = parse_args()
    config = load_config(args.config)
    
    mode = config['agent']['mode']
    if mode not in ['train', 'eval']:
        print("Invalid mode. Choose either 'train' or 'eval'.")
        sys.exit(1)
    
    ticket = config['data']['ticket']
    
    base_log_path = config['agent']['save']['loggers']
    log_file = "training_log.txt"
    log_path, checkpoint_path = setup_logging(base_log_path, ticket, log_file)
    logging.info(f"Logs will be saved to: {log_path}")
    logging.info(f"Checkpoints will be saved to: {checkpoint_path}")
    
    seed = config['agent']['train'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = Dataset(config)
    
    if mode == 'train':
        try:
            train_data = dataset.prepare_data(
                start_date=config['data']['start_train_date'],
                end_date=config['data']['end_train_date'],
                features=config['data']['features'],
                strategy='advanced',
                denoise=True
            )
            
            actual_features = len(train_data.columns) - 1
            logging.info(f"Actual number of features after processing: {actual_features}")
            
            config['environment']['features_dim'] = actual_features
            
            val_data = dataset.prepare_data(
                start_date=config['data']['start_validation_date'],
                end_date=config['data']['end_validation_date'],
                features=config['data']['features'],
                strategy='advanced',
                denoise=True
            )
            
            test_data = dataset.prepare_data(
                start_date=config['data']['start_test_date'],
                end_date=config['data']['end_test_date'],
                features=config['data']['features'],
                strategy='advanced',
                denoise=True
            )
            
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        logging.info("Data loaded and normalized.")
        logging.info(f"Train data shape: {train_data.shape}")
        logging.info(f"Validation data shape: {val_data.shape}")
        logging.info(f"Test data shape: {test_data.shape}")
        
        if train_data.isnull().values.any():
            logging.warning("Train data contains NaN values after normalization.")
        if val_data.isnull().values.any():
            logging.warning("Validation data contains NaN values after normalization.")
        if test_data.isnull().values.any():
            logging.warning("Test data contains NaN values after normalization.")
        
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
        
        train_profits = []
        val_profits = []
        val_accuracies = []
        val_f1_scores = []
        
        env_val = TradingEnv(
            data=val_data,
            features_dim=config['environment']['features_dim'],
            action_dim=config['environment']['action_dim'],
            initial_balance=config['environment']['initial_balance'],
            transaction_fee=config['environment']['transaction_fee']
        )
        
        scaler_path = os.path.join(log_path, "scaler.pkl")
        
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
            
            logging.info(f"Epoch {epoch}/{config['agent']['train']['episodes']} completed. Total Profit: {env.total_profit:.2f}")
            train_profits.append(env.total_profit)
            
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
            
            ppo.update(current_val_profit=total_profit_val)
        
        preprocessing.save_scaler(scaler_path)
        logging.info(f"Scaler saved at {scaler_path}.")
        
        logging.info("Training completed.")
        
        plot_train_val_profits(train_profits, val_profits, log_path)
        plot_validation_accuracy(val_accuracies, log_path)
        plot_validation_f1(val_f1_scores, log_path)
    
    elif mode == 'eval':
        try:
            test_data = dataset.load_data(config['data']['start_test_date'], config['data']['end_test_date'])
        except ValueError as e:
            logging.error(e)
            sys.exit(1)
        
        logging.info("Test data loaded.")
        logging.info(f"Test data shape: {test_data.shape}")
        
        preprocessing = Preprocessing()
        scaler_path = os.path.join(log_path, "scaler.pkl")
        try:
            preprocessing.load_scaler(scaler_path)
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)
        
        test_data = preprocessing.transform(test_data, config['data']['features'])
        
        if test_data.isnull().values.any():
            logging.warning("Test data contains NaN values after normalization.")
        
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
            action_std=config['agent']['train']['action_std'],
            lr_actor=config['agent']['train']['lr_actor'],
            lr_critic=config['agent']['train']['lr_critic'],
            gamma=config['agent']['train']['gamma'],
            epochs=config['agent']['train']['K_epochs'],
            batch_size=config['agent']['train']['batch_size'],
            device=device,
            checkpoint_dir=checkpoint_path
        )
        
        try:
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
                prediction = 1 if action[0] > 0 else 0
                all_predictions.append(prediction)
                actual_price = test_data['close'].iloc[env_test.current_step - 1] if env_test.current_step - 1 < len(test_data) else test_data['close'].iloc[-1]
                actual_prices.append(actual_price)
                predicted_price = test_data['close'].iloc[env_test.current_step] if env_test.current_step < len(test_data) else test_data['close'].iloc[-1]
                predicted_prices.append(predicted_price)
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
        
        if env_test.positions == 1:
            try:
                action = ppo.select_action(state, store=False)
                next_state, reward, done, info = env_test.step(action)
                all_rewards.append(reward)
                logging.info(f"Sold at price {test_data['close'].iloc[-1]:.2f}, Profit: {reward:.2f}")
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
        
        total_profit = env_test.total_profit
        total_transactions = len([r for r in all_rewards if r != 0]) // 2
        profitable_transactions = sum([1 for r in all_rewards if r > 0])
        unprofitable_transactions = sum([1 for r in all_rewards if r < 0])
        
        if len(all_trues) > 0:
            accuracy = accuracy_score(all_trues, all_predictions)
            f1 = f1_score(all_trues, all_predictions, average='weighted')
            logging.info(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")
        else:
            logging.warning("No true labels available for evaluation metrics.")
            accuracy = 0
            f1 = 0
        
        logging.info(f"Total Profit: {total_profit:.2f}")
        logging.info(f"Total Transactions: {total_transactions}")
        logging.info(f"Profitable Transactions: {profitable_transactions}")
        logging.info(f"Unprofitable Transactions: {unprofitable_transactions}")
        
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        logging.info(f"Test F1 Score: {f1:.4f}")
        
        test_profits = all_rewards
        
        plot_test_profit(test_profits, accuracy, f1, log_path)
        
        logging.info("Evaluation completed.")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main()