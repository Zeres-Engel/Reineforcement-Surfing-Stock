# evaluator.py
import os
import logging
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from env.vn30_env import TradingEnv
from model.ppo import PPO
from dataloader.dataset import Dataset
from dataloader.preprocessing import Preprocessing
from utils.plot import plot_test_profit
from utils.logging import setup_logging

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.ticket = config['data']['ticket']
        self.base_log_path = config['agent']['save']['loggers']
        self.log_file = "evaluation_log.txt"
        self.log_path, self.checkpoint_path = setup_logging(self.base_log_path, self.ticket, self.log_file)
        logging.info(f"Logs will be saved to: {self.log_path}")
        logging.info(f"Checkpoints will be saved to: {self.checkpoint_path}")

        seed = config['agent']['train'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.dataset = Dataset(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self):
        try:
            test_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_test_date'],
                end_date=self.config['data']['end_test_date'],
                features=self.config['data']['features'],
                strategy='advanced',
                denoise=True,
                n_features=self.config['environment']['features_dim']
            )

            return test_data
        except ValueError as e:
            logging.error(e)
            sys.exit(1)

    def evaluate(self):
        test_data = self.prepare_data()

        logging.info("Test data loaded.")
        logging.info(f"Test data shape: {test_data.shape}")

        preprocessing = Preprocessing()
        scaler_path = os.path.join(self.log_path, "scaler.pkl")
        try:
            preprocessing.load_scaler(scaler_path)
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)

        # Transform the test data
        selected_features = test_data.columns.drop('close').tolist()
        test_features = selected_features[:self.config['environment']['features_dim']]
        test_data = preprocessing.transform(test_data, test_features)

        if test_data.isnull().values.any():
            logging.warning("Test data contains NaN values after normalization.")

        env_test = TradingEnv(
            data=test_data,
            features_dim=self.config['environment']['features_dim'],
            action_dim=self.config['environment']['action_dim'],
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee']
        )

        ppo = PPO(
            state_dim=self.config['environment']['features_dim'],
            action_dim=self.config['environment']['action_dim'],
            action_std=self.config['agent']['train']['action_std'],
            lr_actor=self.config['agent']['train']['lr_actor'],
            lr_critic=self.config['agent']['train']['lr_critic'],
            gamma=self.config['agent']['train']['gamma'],
            epochs=self.config['agent']['train']['K_epochs'],
            batch_size=self.config['agent']['train']['batch_size'],
            device=self.device,
            checkpoint_dir=self.checkpoint_path
        )

        try:
            best_model_path = os.path.join(self.checkpoint_path, "best_model", "best_model.pth")
            last_model_path = os.path.join(self.checkpoint_path, "last_model", "last_model.pth")
            
            if os.path.exists(best_model_path):
                checkpoint_to_load = best_model_path
                logging.info(f"Loading best model from {checkpoint_to_load}.")
            elif os.path.exists(last_model_path):
                checkpoint_to_load = last_model_path
                logging.info(f"Loading last model from {checkpoint_to_load}.")
            else:
                logging.error("No checkpoint found.")
                sys.exit(1)
                
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

        # Check if there's an open position at the end
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

        # Create separate directories for plots
        plot_dir = os.path.join(self.log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_test_profit(test_profits, accuracy, f1, plot_dir)

        logging.info("Evaluation completed.")
