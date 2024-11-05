# utils/plot.py
import matplotlib.pyplot as plt
import os

def plot_train_val_profits(train_profits, val_profits, plot_dir):
    plt.figure(figsize=(10,5))
    plt.plot(train_profits, label='Train Profit')
    plt.plot(val_profits, label='Validation Profit')
    plt.xlabel('Epoch')
    plt.ylabel('Profit')
    plt.title('Train and Validation Profits Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'train_val_profits.png'))
    plt.close()

def plot_validation_accuracy(val_accuracies, plot_dir):
    plt.figure(figsize=(10,5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'validation_accuracy.png'))
    plt.close()

def plot_validation_f1(val_f1_scores, plot_dir):
    plt.figure(figsize=(10,5))
    plt.plot(val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score Over Epochs')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'validation_f1_score.png'))
    plt.close()

def plot_test_profit(test_profits, accuracy, f1, plot_dir):
    plt.figure(figsize=(10,5))
    plt.plot(test_profits, label='Test Profit')
    plt.xlabel('Step')
    plt.ylabel('Profit')
    plt.title(f'Test Profit - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'test_profit.png'))
    plt.close()
