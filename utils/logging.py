# utils/logging.py

import os
import logging
import sys

def setup_logging(base_log_path, ticket, log_file):
    log_path = os.path.join(base_log_path, ticket)
    os.makedirs(log_path, exist_ok=True)
    
    checkpoint_path = os.path.join(log_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File handler for INFO and above
        file_handler = logging.FileHandler(os.path.join(log_path, log_file), encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler for INFO and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return log_path, checkpoint_path
