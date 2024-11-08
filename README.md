# SURFING STOCK

<div>
  <p align="center">
    <img src="images/logo.svg" width="800"> 
  </p>
</div>

**Surfing Stock** is an application that utilizes Reinforcement Learning to predict and execute stock trades. The goal of the project is to build a model capable of learning from historical data and optimizing trading strategies to maximize profit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Reinforcement Learning](https://img.shields.io/badge/Model-PPO-orange.svg)

## Table of Contents
- [Overview](#overview)
- [Data Analysis](#data-analysis)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation Results](#evaluation-results)
- [Installation and Usage](#installation-and-usage)
- [License](#license)
- [Contact](#contact)

## Overview

The application implements a sophisticated trading system using Reinforcement Learning (PPO) with these key features:
- Automated data collection using VNStock API
- Comprehensive technical indicator generation
- Feature combination search for optimal model performance
- Multi-stage evaluation pipeline (Train → Validate → Test)

## Data Analysis

### Sample Data
Here's a sample of the stock data (FPT.csv):
```csv
time,open,high,low,close,volume
9/11/2023 9:15,98.7,99.5,98.4,98.5,356300
9/11/2023 9:20,98.5,98.5,98.2,98.3,74200
9/11/2023 9:25,98.2,98.2,97.7,97.9,117400
9/11/2023 9:30,98,98.1,97.8,97.8,94700
9/11/2023 9:35,98,98,97.9,97.9,30200
```

### Seasonal Analysis
<div>
  <p align="center">
    <img src="images/seasonal.png" width="800">
    <br>
    <em>Seasonal Decomposition of FPT Stock Price</em>
  </p>
</div>

The seasonal analysis reveals:
- Clear daily trading patterns
- Weekly cycles in trading volume
- Strong trend component indicating overall market direction
- Residual noise that requires filtering

### Periodogram Analysis
<div>
  <p align="center">
    <img src="images/periodogram.png" width="800">
    <br>
    <em>Periodogram Analysis for Window Selection</em>
  </p>
</div>

Based on the periodogram analysis, we identified significant frequencies that correspond to these optimal window sizes:
```python
windows = [419, 503, 559, 629, 839, 1007, 1258, 1678]
```

These windows were selected because:
- They capture both short-term and long-term price movements
- Correspond to natural market cycles (daily, weekly, monthly patterns)
- Show strong power spectral density in the periodogram

## Model Architecture

### Trading Environment
- **Type**: Custom OpenAI Gym environment
- **State Space**: Technical indicators and price data
- **Action Space**: Continuous [-1, 1] representing sell/hold/buy
- **Reward Function**: 
  - Trade profit/loss when position is closed
  - Price change when holding position
  - Transaction fees included

### PPO Implementation
- **Policy Network**: Actor-Critic architecture
- **Features**:
  - Advantage estimation
  - Policy clipping
  - Value function learning
  - Entropy bonus for exploration

### Action Interpretation
- Action > 0.1: Buy signal
- Action < -0.1: Sell signal
- Otherwise: Hold position

## Training Pipeline

<div>
  <p align="center">
    <img src="images/training_pipeline.png" width="800">
    <br>
    <em>Feature Selection and Training Pipeline Overview</em>
  </p>
</div>

### Training Configuration
```yaml
# Data Split Dates
training:
  start_train_date: "2023-11-09"
  end_train_date: "2024-03-31"
  start_validation_date: "2024-04-01"
  end_validation_date: "2024-07-31"
  start_test_date: "2024-08-01"
  end_test_date: "2024-11-01"
```

The pipeline implements a three-stage evaluation process:
1. **Training** (2023-11-09 to 2024-03-31): Model learning and feature combination search
2. **Validation** (2024-04-01 to 2024-07-31): Model selection and hyperparameter tuning
3. **Testing** (2024-08-01 to 2024-11-01): Final performance evaluation

### Training Process

<div>
  <p align="center">
    <img src="images/training_process.png" width="800">
    <br>
    <em>Model Training Process Workflow</em>
  </p>
</div>

The model uses a combinatorial feature selection approach, training multiple PPO agents with different feature sets to find the optimal combination. Performance is tracked using validation metrics (F1-score, Accuracy) to select the best model.

### Evaluation Process

<div>
  <p align="center">
    <img src="images/eval_process.png" width="800">
    <br>
    <em>Model Evaluation Workflow</em>
  </p>
</div>

Final evaluation is performed on unseen test data, measuring:
- Trading metrics (profit, win rate, Sharpe ratio)
- Classification metrics (accuracy, F1-score)
- Performance visualization and reporting

## Evaluation Results

### Training Progress
<div>
  <p align="center">
    <img src="logs/FPT/checkpoints/best_model/validation_accuracy_combo_best.png" width="400">
    <img src="logs/FPT/checkpoints/best_model/validation_f1_score_combo_best.png" width="400">
    <br>
    <em>Validation Accuracy and F1-Score During Training</em>
  </p>
</div>

### Trading Performance
<div>
  <p align="center">
    <img src="logs/FPT/checkpoints/best_model/best_model_profit_combo_best.png" width="800">
    <br>
    <em>Cumulative Profit Over Test Period</em>
  </p>
</div>

### Best Model Performance
- **Feature Set**: ['low', 'SMA_419', 'MACD_1007', 'STOCH_2516', 'BB_upper_1678']
- **Test Metrics**:
  - Accuracy: 52.45%
  - F1 Score: 0.5857
  - Win Rate: 12.15%
  - Sharpe Ratio: 0.01
  - Total Profit: +0.94 (+94% return)

The model shows promising results with:
- Consistent positive returns over the test period
- Stable accuracy and F1-score metrics
- Effective feature selection combining different technical indicators
- Risk-adjusted returns as measured by Sharpe Ratio

## Installation and Usage

### Requirements
- Python 3.8+
- Dependencies listed in `environment.yml`

### Installation Steps
```bash
# Clone repository
git clone https://github.com/Zeres-Engel/Reinforcement-Surfing-Stock.git
cd Reinforcement-Surfing-Stock

# Create and activate environment
conda env create -f environment.yml
conda activate surfing-stock

# Run training
python main.py --config configs/FPT.yaml
```

### Configuration
Modify `configs/FPT.yaml` to adjust:
- Training parameters
- Feature dimensions
- Data date ranges
- Model architecture

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
- Email: [ngpthanh15@gmail.com](mailto:ngpthanh15@gmail.com)
- GitHub: [Zeres-Engel](https://github.com/Zeres-Engel)
