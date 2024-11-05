# main.py

import argparse
import yaml
from trainer import Trainer
import logging

def main():
    parser = argparse.ArgumentParser(description="Trading Model Training and Evaluation")
    parser.add_argument('--config', type=str, default='configs/FPT.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Initialize Trainer
    trainer = Trainer(config)

    # Start training and evaluation
    trainer.train()

    # Evaluate best model on test set
    trainer.evaluate_best_model()

if __name__ == "__main__":
    main()
