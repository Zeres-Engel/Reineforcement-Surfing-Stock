# utils/plot.py

import matplotlib.pyplot as plt
import os
import logging

def plot_validation_accuracy(val_accuracies, log_path, combination_idx='last'):
    plt.figure(figsize=(10,5))
    episodes = range(1, len(val_accuracies) + 1)  # X bắt đầu từ 1 và là số nguyên
    plt.plot(episodes, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Over Episodes")
    plt.legend()
    if combination_idx is not None:
        acc_path = os.path.join(log_path, f"validation_accuracy_combo_{combination_idx}.png")
    else:
        acc_path = os.path.join(log_path, "validation_accuracy.png")
    plt.savefig(acc_path)
    plt.close()
    logging.info(f"Validation accuracy plotted at {acc_path}.")

def plot_validation_f1(val_f1_scores, log_path, combination_idx='last'):
    plt.figure(figsize=(10,5))
    episodes = range(1, len(val_f1_scores) + 1)  # X bắt đầu từ 1 và là số nguyên
    plt.plot(episodes, val_f1_scores, label='Validation F1-Score', color='red')
    plt.xlabel("Episode")
    plt.ylabel("F1-Score")
    plt.title("Validation F1-Score Over Episodes")
    plt.legend()
    if combination_idx is not None:
        f1_path = os.path.join(log_path, f"validation_f1_score_combo_{combination_idx}.png")
    else:
        f1_path = os.path.join(log_path, "validation_f1_score.png")
    plt.savefig(f1_path)
    plt.close()
    logging.info(f"Validation F1-score plotted at {f1_path}.")

def plot_combination_metrics(f1_scores, accuracies, feature_sets, log_path):
    combinations = list(range(1, len(f1_scores) + 1))  # Tổ hợp thứ tự bắt đầu từ 1

    # Plot F1-Score qua từng tổ hợp
    plt.figure(figsize=(10,5))
    plt.plot(combinations, f1_scores, marker='o', label='F1-Score', color='red')
    best_f1 = max(f1_scores)
    best_f1_idx = f1_scores.index(best_f1) + 1  # +1 vì index bắt đầu từ 0
    plt.scatter(best_f1_idx, best_f1, color='blue', label=f'Best F1-Score (Combo {best_f1_idx})')
    plt.xlabel("Feature Combination")
    plt.ylabel("F1-Score")
    plt.title("Validation F1-Score Across Feature Combinations")
    plt.legend()
    f1_combinations_path = os.path.join(log_path, "combination_f1_scores.png")
    plt.savefig(f1_combinations_path)
    plt.close()
    logging.info(f"Combination F1-scores plotted at {f1_combinations_path}.")

    # Plot Accuracy qua từng tổ hợp
    plt.figure(figsize=(10,5))
    plt.plot(combinations, accuracies, marker='o', label='Accuracy', color='green')
    best_acc = max(accuracies)
    best_acc_idx = accuracies.index(best_acc) + 1  # +1 vì index bắt đầu từ 0
    plt.scatter(best_acc_idx, best_acc, color='blue', label=f'Best Accuracy (Combo {best_acc_idx})')
    plt.xlabel("Feature Combination")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Across Feature Combinations")
    plt.legend()
    acc_combinations_path = os.path.join(log_path, "combination_accuracies.png")
    plt.savefig(acc_combinations_path)
    plt.close()
    logging.info(f"Combination Accuracies plotted at {acc_combinations_path}.")

def plot_cumulative_profit(profits, log_path, model_type='best', combination_idx=None):
    plt.figure(figsize=(12,6))
    steps = range(1, len(profits) + 1)
    plt.plot(steps, profits, label=f'Cumulative Profit', color='blue')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    plt.xlabel("Trading Steps")
    plt.ylabel("Cumulative Profit")
    
    if combination_idx is not None:
        title = f"{model_type.capitalize()} Model Cumulative Profit - Combination {combination_idx}"
        filename = f"{model_type}_model_profit_combo_{combination_idx}.png"
    else:
        title = f"{model_type.capitalize()} Model Cumulative Profit"
        filename = f"{model_type}_model_profit.png"
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(log_path, filename)
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Cumulative profit plot saved at {save_path}")
