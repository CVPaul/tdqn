# Trading-Deep-Q-Network

## Overview

This repository aims to reproduce the experimentation presented in the paper [An Application of Deep Reinforcement Learning to Algorithmic Trading](https://arxiv.org/pdf/2004.06627). The goal is to implement the Trading Deep Q-Network (TDQN) algorithm and test its performance in a simulated trading environment. The implementation will follow the methodology described in the paper, including reinforcement learning techniques, trading data preprocessing, and performance evaluation.

## Objectives

Implement the Trading Deep Q-Network (TDQN) algorithm.

Recreate the artificial data generation process for training the reinforcement learning agent.

Evaluate the TDQN strategy using historical market data.

Compare TDQN's performance with benchmark strategies like Buy and Hold (B&H), Trend Following (TF), and Mean Reversion (MR).

Analyze key metrics such as the Sharpe ratio, annualized return, and maximum drawdown.

## Features

Deep Reinforcement Learning (DRL): Implement a DQN-based trading agent with enhancements like Double DQN and Huber loss.

Trading Environment: Simulated trading environment with realistic constraints, including trading costs and order execution.

Data Processing: Preprocessing of stock market data, normalization techniques, and feature engineering.

Performance Evaluation: Implementation of standardized performance metrics to assess the TDQN strategy.

## Installation

To set up the project, clone this repository and install the required dependencies:
```
git clone https://github.com/pafruchtenreich/Trading-Deep-Q-Network.git
cd Trading-Deep-Q-Network
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
