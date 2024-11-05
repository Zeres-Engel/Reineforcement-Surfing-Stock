# trainer.py

import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score
from env.vn30_env import TradingEnv
from model.ppo import PPO
from dataloader.dataset import Dataset
from utils.plot import (
    plot_validation_accuracy,
    plot_validation_f1,
    plot_combination_metrics,
    plot_cumulative_profit
)
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

        seed = self.config['agent']['train'].get('seed', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)  # Đặt seed cho random

        self.dataset = Dataset(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tạo thư mục last_model và best_model bên trong checkpoint_path
        self.last_model_dir = os.path.join(self.checkpoint_path, "last_model")
        self.best_model_dir = os.path.join(self.checkpoint_path, "best_model")

        os.makedirs(self.last_model_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)

        # Khởi tạo các thuộc tính để lưu trữ best model toàn cục
        self.global_best_val_f1 = -float('inf')
        self.global_best_val_accuracy = -float('inf')
        self.global_best_features = None
        self.global_best_model_checkpoint = None

        # Lists để lưu trữ metrics qua từng tổ hợp
        self.combination_f1_scores = []
        self.combination_accuracies = []
        self.combination_feature_sets = []

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
                    device=self.device
                )
                
                # Huấn luyện mô hình với thanh tiến trình con cho các episode
                val_accuracies = []
                val_f1_scores = []

                with tqdm(total=self.config['agent']['train']['episodes'], desc=f"Training Combo {idx + 1}", unit="episode", leave=False) as episode_pbar:
                    for episode in range(1, self.config['agent']['train']['episodes'] + 1):
                        state = env.reset()
                        done = False
                        cumulative_profit = 0  # Lợi nhuận tích lũy cho episode này
                        while not done:
                            action = ppo.select_action(state)
                            next_state, reward, done, _ = env.step(action)
                            ppo.buffer.rewards.append(reward)
                            ppo.buffer.is_terminals.append(done)
                            cumulative_profit += reward
                            state = next_state
                        
                        # Đánh giá trên tập validation sau mỗi episode
                        val_profit, accuracy, f1 = self._validate_model(ppo, val_data, feature_set)
                        ppo.update(current_val_profit=val_profit)
                        val_accuracies.append(accuracy)
                        val_f1_scores.append(f1)

                        if episode % 10 == 0 or episode == self.config['agent']['train']['episodes']:
                            logging.info(f"Combination {idx + 1}, Episode {episode} completed.")
                            logging.info(f"Val Accuracy: {accuracy:.4f}, Val F1 Score: {f1:.4f}")
                        
                        episode_pbar.update(1)
                
                # Đánh giá mô hình trên tập test
                cumulative_test_profit, test_accuracy, test_f1, sharpe_ratio, win_rate = self._evaluate_model(ppo, test_data, feature_set)
                
                # Thêm vẽ biểu đồ cho last model
                plot_cumulative_profit(
                    cumulative_test_profit,
                    self.last_model_dir,
                    model_type='last',
                    combination_idx='last'
                )

                # Lưu Last Model cho tổ hợp này (chỉ lưu checkpoint cuối cùng)
                last_model_path = os.path.join(self.last_model_dir, f"combo_last.pth")
                ppo.save_checkpoint(last_model_path)
                logging.info(f"Last model for combination {idx + 1} saved at {last_model_path}")

                # Thu thập metrics cho tổ hợp này
                self.combination_f1_scores.append(max(val_f1_scores))
                self.combination_accuracies.append(max(val_accuracies))
                self.combination_feature_sets.append(feature_set)
                
                # Vẽ và lưu biểu đồ val_accuracy và val_f1score cho last_model
                plot_validation_accuracy(val_accuracies, self.last_model_dir)
                plot_validation_f1(val_f1_scores, self.last_model_dir)

                # Kiểm tra và cập nhật Best Model toàn cục
                current_best_f1 = max(val_f1_scores)
                current_best_accuracy = max(val_accuracies)
                
                is_new_best = False
                if current_best_f1 > self.global_best_val_f1:
                    self.global_best_val_f1 = current_best_f1
                    is_new_best = True
                if current_best_accuracy > self.global_best_val_accuracy:
                    self.global_best_val_accuracy = current_best_accuracy
                    is_new_best = True
                
                if is_new_best:
                    self.global_best_features = feature_set
                    best_model_path = os.path.join(self.best_model_dir, "best_model.pth")
                    ppo.save_checkpoint(best_model_path)
                    self.global_best_model_checkpoint = best_model_path
                    
                    # Thêm vẽ biểu đồ cho best model
                    plot_cumulative_profit(
                        cumulative_test_profit,
                        self.best_model_dir,
                        model_type='best',
                        combination_idx=idx + 1
                    )
                    
                    logging.info(f"New global best model! Val F1: {self.global_best_val_f1:.4f}, Val Accuracy: {self.global_best_val_accuracy:.4f}")
                    logging.info(f"Features: {self.global_best_features}")
                    logging.info(f"Best model checkpoint saved at: {best_model_path}")


                    plot_validation_accuracy(val_accuracies, self.best_model_dir, combination_idx=idx + 1)
                    plot_validation_f1(val_f1_scores, self.best_model_dir, combination_idx=idx + 1)

                # Cập nhật thanh tiến trình tổng
                combo_pbar.update(1)
    
    # Các hàm _validate_model và _evaluate_model không thay đổi
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

    def _evaluate_model(self, ppo, test_data, feature_set):
        """Evaluate the model on the test set and return cumulative_profit, accuracy, f1, sharpe_ratio, win_rate."""
        env_test = TradingEnv(
            data=test_data,
            features_dim=len([f for f in feature_set if f != 'close']),
            action_dim=self.config['environment']['action_dim'],
            initial_balance=self.config['environment']['initial_balance'],
            transaction_fee=self.config['environment']['transaction_fee']
        )

        state = env_test.reset()
        done = False
        total_profit = 0
        profits = []
        wins = 0
        total_trades = 0
        all_predictions = []
        all_trues = []

        # Sử dụng thanh tiến trình cho quá trình đánh giá
        with tqdm(total=len(env_test.data), desc="Evaluating on Test Set", unit="step") as test_pbar:
            while not done:
                action = ppo.select_action(state, store=False)
                next_state, reward, done, _ = env_test.step(action)
                total_profit += reward
                profits.append(reward)
                if reward > 0:
                    wins += 1
                total_trades += 1
                prediction = 1 if action[0] > 0 else 0
                all_predictions.append(prediction)
                if env_test.current_step < len(env_test.data):
                    true_label = 1 if env_test.data['close'].iloc[env_test.current_step] > env_test.data['close'].iloc[env_test.current_step - 1] else 0
                    all_trues.append(true_label)
                state = next_state
                test_pbar.update(1)

        # Tính Sharpe Ratio và Win Rate
        returns = np.array(profits)
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Tính Accuracy và F1 Score
        if len(all_trues) > 0:
            accuracy = accuracy_score(all_trues, all_predictions)
            f1 = f1_score(all_trues, all_predictions, average='weighted')
        else:
            logging.warning("No true labels available for test metrics.")
            accuracy = 0.0
            f1 = 0.0

        # Tính lợi nhuận tích lũy
        cumulative_profit = np.cumsum(profits).tolist()

        return cumulative_profit, accuracy, f1, sharpe_ratio, win_rate

    def evaluate_best_model(self):
        """Evaluate the best model on test set using the best features"""
        if not self.global_best_features or not self.global_best_model_checkpoint:
            logging.error("No best model found. Please run training first.")
            return
            
        # Chuẩn bị test data với best features
        _, _, test_data = self.prepare_data(features=self.global_best_features)
        
        # Load best model
        features_without_close = [f for f in self.global_best_features if f != 'close']
        ppo = PPO(
            state_dim=len(features_without_close),
            action_dim=self.config['environment']['action_dim'],
            action_std=self.config['agent']['train']['action_std'],
            lr_actor=self.config['agent']['train']['lr_actor'],
            lr_critic=self.config['agent']['train']['lr_critic'],
            gamma=self.config['agent']['train']['gamma'],
            epochs=self.config['agent']['train']['K_epochs'],
            batch_size=self.config['agent']['train']['batch_size'],
            device=self.device
        )
        ppo.load_checkpoint(self.global_best_model_checkpoint)
        
        # Đánh giá trên test set
        cumulative_test_profit, test_accuracy, test_f1, sharpe_ratio, win_rate = self._evaluate_model(ppo, test_data, self.global_best_features)
        
        # Vẽ biểu đồ profit cuối cùng cho best model
        plot_cumulative_profit(
            cumulative_test_profit,
            self.checkpoint_path,
            model_type='final_best'
        )
        
        # Log kết quả
        logging.info("\nTest Results:")
        logging.info(f"Best Features: {self.global_best_features}")
        logging.info(f"Total Profit: {cumulative_test_profit[-1]:.2f}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logging.info(f"Win Rate: {win_rate:.2%}")
        logging.info(f"Test Accuracy: {test_accuracy:.4f}")
        logging.info(f"Test F1 Score: {test_f1:.4f}")
        
        # Tạo và lưu các biểu đồ F1-score và Accuracy qua từng tổ hợp
        plot_combination_metrics(
            self.combination_f1_scores,
            self.combination_accuracies,
            self.combination_feature_sets,
            self.checkpoint_path
        )
