# trainer.py

import os
import logging
import torch
import numpy as np
from tqdm import tqdm, trange
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score
from env.vn30_env import TradingEnv
from model.ppo import PPO
from dataloader.dataset import Dataset
from utils.plot import plot_test_profit
from utils.logging import setup_logging
import random

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
        random.seed(seed)  # Đặt seed cho random

        self.dataset = Dataset(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, features):
        """Prepare train, val, test data for specific feature set."""
        try:
            # Chuẩn bị dữ liệu train
            train_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_train_date'],
                end_date=self.config['data']['end_train_date'],
                features=features,
                strategy='default',  # Bạn có thể thay đổi chiến lược augment nếu cần
                denoise=True
            )

            # Chuẩn bị dữ liệu validation
            val_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_validation_date'],
                end_date=self.config['data']['end_validation_date'],
                features=features,
                strategy='default',
                denoise=True
            )

            # Chuẩn bị dữ liệu test
            test_data = self.dataset.prepare_data(
                start_date=self.config['data']['start_test_date'],
                end_date=self.config['data']['end_test_date'],
                features=features,
                strategy='default',
                denoise=True
            )

            # Chọn các feature đã chọn và thêm 'close'
            train_data = self.dataset.get_features_data(train_data, selected_features=features)
            val_data = self.dataset.get_features_data(val_data, selected_features=features)
            test_data = self.dataset.get_features_data(test_data, selected_features=features)

            return train_data, val_data, test_data
        except ValueError as e:
            logging.error(e)
            return None, None, None

    def train(self):
        # 1. Lấy tất cả features sau khi augment
        all_augmented_features, augmented_train_data = self.dataset.get_all_augmented_features(
            self.config['data']['start_train_date'],
            self.config['data']['end_train_date']
        )
        
        features_dim = self.config['environment']['features_dim']
        max_combinations = self.config.get('agent', {}).get('train', {}).get('max_combinations', 1000)
        
        logging.info(f"Total augmented features: {len(all_augmented_features)}")
        logging.info(f"Selecting combinations of {features_dim} features")
        
        # 2. Tạo tổ hợp từ features đã augment
        all_feature_combinations = list(combinations(all_augmented_features, features_dim))
        total_combinations = len(all_feature_combinations)
        if total_combinations > max_combinations:
            feature_combinations = random.sample(all_feature_combinations, max_combinations)
            logging.info(f"Randomly selected {max_combinations} feature combinations out of {total_combinations}")
        else:
            feature_combinations = all_feature_combinations
            logging.info(f"Using all {total_combinations} feature combinations")
        
        best_val_profit = -float('inf')
        best_features = None
        best_model_checkpoint = None
        
        # 3. Duyệt qua từng tổ hợp với thanh tiến trình tổng
        with tqdm(total=len(feature_combinations), desc="Total Feature Combinations", unit="combo") as combo_pbar:
            for idx, feature_set in enumerate(feature_combinations):
                feature_set = list(feature_set)
                logging.info(f"\nEvaluating combination {idx + 1}/{len(feature_combinations)}: {feature_set}")

                # Log số cột trước và sau khi augment
                augmented_columns = len(all_augmented_features)
                logging.info(f"Number of augmented features: {augmented_columns}")

                # Chuẩn bị dữ liệu cho tổ hợp feature hiện tại
                train_data, val_data, test_data = self.prepare_data(features=feature_set)
                
                if train_data is None or val_data is None:
                    combo_pbar.update(1)
                    continue

                # Log số cột dữ liệu đầu vào và sau khi augment
                original_columns = len(feature_set)
                processed_columns = len(train_data.columns)
                logging.info(f"Input columns: {original_columns}, After augment columns: {processed_columns}")

                # Khởi tạo environment với số chiều state đúng
                features_without_close = [f for f in feature_set if f != 'close']
                env = TradingEnv(
                    data=train_data,
                    features_dim=len(features_without_close),
                    action_dim=self.config['environment']['action_dim'],
                    initial_balance=self.config['environment']['initial_balance'],
                    transaction_fee=self.config['environment']['transaction_fee']
                )
                
                # Khởi tạo model
                ppo = PPO(
                    state_dim=len(features_without_close),
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
                
                # Huấn luyện mô hình với thanh tiến trình con cho các episode
                with tqdm(total=self.config['agent']['train']['episodes'], desc=f"Training Combo {idx + 1}", unit="episode", leave=False) as episode_pbar:
                    for episode in range(1, self.config['agent']['train']['episodes'] + 1):
                        state = env.reset()
                        done = False
                        while not done:
                            action = ppo.select_action(state)
                            next_state, reward, done, _ = env.step(action)
                            ppo.buffer.rewards.append(reward)
                            ppo.buffer.is_terminals.append(done)
                            state = next_state
                        
                        # Cập nhật mô hình sau mỗi episode
                        ppo.update(current_val_profit=0)  # current_val_profit sẽ được cập nhật sau
                
                        if episode % 10 == 0:
                            logging.info(f"Combination {idx + 1}, Episode {episode} completed.")
                        
                        episode_pbar.update(1)
                
                # Đánh giá mô hình trên tập validation
                val_profit, accuracy, f1 = self._validate_model(ppo, val_data, feature_set)
                
                # Cập nhật best model nếu cần
                if val_profit > best_val_profit:
                    best_val_profit = val_profit
                    best_features = feature_set
                    best_model_checkpoint = ppo.save_checkpoint(
                        os.path.join(self.checkpoint_path, f"best_model_{idx}_{episode}.pth"),
                        is_best=True
                    )
                    logging.info(f"New best model! Profit: {best_val_profit:.2f}")
                    logging.info(f"Features: {best_features}")
                    logging.info(f"Checkpoint Path: {best_model_checkpoint}")
                
                # Cập nhật thanh tiến trình tổng
                combo_pbar.update(1)
        
        # 4. Lưu thông tin best model
        self.best_features = best_features
        self.best_model_checkpoint = best_model_checkpoint
        
        # Lưu thông tin vào file
        with open(os.path.join(self.log_path, "best_model_info.txt"), "w") as f:
            f.write(f"Best Features: {best_features}\n")
            f.write(f"Best Validation Profit: {best_val_profit}\n")
            f.write(f"Checkpoint Path: {best_model_checkpoint}\n")

    def _validate_model(self, ppo, val_data, feature_set):
        """Validate the model on the validation set."""
        env_val = TradingEnv(
            data=val_data,
            features_dim=len(feature_set) - 1,  # Trừ 'close'
            action_dim=self.config['environment']['action_dim'],
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee']
        )

        total_profit_val = 0
        all_predictions = []
        all_trues = []

        state = env_val.reset()
        done = False
        while not done:
            action = ppo.select_action(state, store=False)
            next_state, reward, done, _ = env_val.step(action)
            total_profit_val += reward
            prediction = 1 if action[0] > 0 else 0
            all_predictions.append(prediction)
            if env_val.current_step < len(val_data):
                true_label = 1 if val_data['close'].iloc[env_val.current_step] > val_data['close'].iloc[env_val.current_step - 1] else 0
                all_trues.append(true_label)
            state = next_state

        if len(all_trues) > 0:
            accuracy = accuracy_score(all_trues, all_predictions)
            f1 = f1_score(all_trues, all_predictions, average='weighted')
            logging.info(f"Combination {feature_set}: Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        else:
            logging.warning(f"Combination {feature_set}: No true labels available for validation metrics.")
            accuracy = 0
            f1 = 0

        return total_profit_val, accuracy, f1

    def evaluate_best_model(self):
        """Evaluate the best model on test set using the best features"""
        if not hasattr(self, 'best_features') or not hasattr(self, 'best_model_checkpoint'):
            logging.error("No best model found. Please run training first.")
            return
            
        # Chuẩn bị test data với best features
        _, _, test_data = self.prepare_data(features=self.best_features)
        
        # Load best model
        features_without_close = [f for f in self.best_features if f != 'close']
        ppo = PPO(
            state_dim=len(features_without_close),
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
        ppo.load_checkpoint(self.best_model_checkpoint)
        
        # Evaluate trên test set
        test_env = TradingEnv(
            data=test_data,
            features_dim=len(features_without_close),
            action_dim=self.config['environment']['action_dim'],
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee']
        )
        
        # Test loop và logging metrics với thanh tiến trình
        test_metrics = self._evaluate_model(ppo, test_env)
        
        # Vẽ biểu đồ
        plot_dir = os.path.join(self.log_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_test_profit(test_metrics['profits'], test_metrics['accuracy'], test_metrics['f1'], plot_dir)
        
        # Log kết quả
        logging.info("\nTest Results:")
        logging.info(f"Best Features: {self.best_features}")
        logging.info(f"Total Profit: {test_metrics['total_profit']:.2f}")
        logging.info(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}")
        logging.info(f"Win Rate: {test_metrics['win_rate']:.2%}")

    def _evaluate_model(self, ppo, env):
        """Evaluate the model on the test environment."""
        state = env.reset()
        done = False
        total_profit = 0
        profits = []
        wins = 0
        total_trades = 0
        all_predictions = []
        all_trues = []

        # Sử dụng thanh tiến trình cho quá trình đánh giá
        with tqdm(total=len(env.data), desc="Evaluating on Test Set", unit="step") as test_pbar:
            while not done:
                action = ppo.select_action(state, store=False)
                next_state, reward, done, _ = env.step(action)
                total_profit += reward
                profits.append(total_profit)
                if reward > 0:
                    wins += 1
                total_trades += 1
                prediction = 1 if action[0] > 0 else 0
                all_predictions.append(prediction)
                if env.current_step < len(env.data):
                    true_label = 1 if env.data['close'].iloc[env.current_step] > env.data['close'].iloc[env.current_step - 1] else 0
                    all_trues.append(true_label)
                state = next_state
                test_pbar.update(1)

        # Tính Sharpe Ratio
        returns = np.array(profits)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)

        # Tính Win Rate
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Tính Accuracy và F1 Score
        if len(all_trues) > 0:
            accuracy = accuracy_score(all_trues, all_predictions)
            f1 = f1_score(all_trues, all_predictions, average='weighted')
        else:
            logging.warning("No true labels available for test metrics.")
            accuracy = 0.0
            f1 = 0.0

        return {
            'total_profit': total_profit,
            'profits': profits,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'accuracy': accuracy,
            'f1': f1
        }
