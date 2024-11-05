# trainer.py
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from env.vn30_env import TradingEnv
from model.ppo import PPO
from dataloader.dataset import Dataset
from utils.plot import plot_train_val_profits, plot_validation_accuracy, plot_validation_f1
from utils.logging import setup_logging

class Trainer:
    def __init__(self, config):
        self.config = config
        self.ticket = config['data']['ticket']
        self.base_log_path = config['agent']['save']['loggers']
        self.log_file = "training_log.txt"
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
            train_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_train_date'],
                end_date=self.config['data']['end_train_date'],
                features=self.config['data']['features'],
                strategy='advanced',
                denoise=True,
                n_features=self.config['environment']['features_dim']
            )

            val_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_validation_date'],
                end_date=self.config['data']['end_validation_date'],
                features=self.config['data']['features'],
                strategy='advanced',
                denoise=True,
                n_features=self.config['environment']['features_dim']
            )

            test_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_test_date'],
                end_date=self.config['data']['end_test_date'],
                features=self.config['data']['features'],
                strategy='advanced',
                denoise=True,
                n_features=self.config['environment']['features_dim']
            )

            return train_data, val_data, test_data
        except ValueError as e:
            logging.error(e)
            sys.exit(1)

    def train(self):
        train_data, val_data, test_data = self.prepare_data()

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

        train_profits = []
        val_profits = []
        val_accuracies = []
        val_f1_scores = []

        env_val = TradingEnv(
            data=val_data,
            features_dim=self.config['environment']['features_dim'],
            action_dim=self.config['environment']['action_dim'],
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee']
        )

        scaler_path = os.path.join(self.log_path, "scaler.pkl")

        for epoch in tqdm(range(1, self.config['agent']['train']['episodes'] + 1), desc="Training Progress"):
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

            logging.info(f"Epoch {epoch}/{self.config['agent']['train']['episodes']} completed. Total Profit: {env.total_profit:.2f}")
            train_profits.append(env.total_profit)

            # Validation
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

        self.dataset.save_scaler(scaler_path)
        logging.info(f"Scaler saved at {scaler_path}.")

        logging.info("Training completed.")

        # Create separate directories for plots
        plot_dir = os.path.join(self.log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plot_train_val_profits(train_profits, val_profits, plot_dir)
        plot_validation_accuracy(val_accuracies, plot_dir)
        plot_validation_f1(val_f1_scores, plot_dir)
