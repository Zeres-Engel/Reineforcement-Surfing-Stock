# main.py
import argparse
import sys
from utils.configs import load_config
from trainer import Trainer
from evaluator import Evaluator

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

    if mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif mode == 'eval':
        evaluator = Evaluator(config)
        evaluator.evaluate()

if __name__ == "__main__":
    main()
