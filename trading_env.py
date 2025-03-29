import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.prices = data['close'].values  # Extract Close prices as numpy array
        self.current_step = 0
        self.cash = 10000  # Initial cash
        self.shares = 0  # Initial shares
        self.state_size = 2  # [price, position]
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,))
    
    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        return np.array([self.prices[self.current_step], self.shares])
    
    def step(self, action):
        done = self.current_step >= len(self.prices) - 1
        reward = 0
        price = self.prices[self.current_step]
        if action == 0:  # Buy
            shares_to_buy = self.cash // price
            self.shares += shares_to_buy
            self.cash -= shares_to_buy * price
        elif action == 1:  # Sell
            self.cash += self.shares * price
            self.shares = 0
        # Reward based on portfolio value change
        new_value = self.cash + (self.shares * self.prices[self.current_step + 1])
        reward = new_value - (self.cash + (self.shares * price))
        self.current_step += 1
        return np.array([self.prices[self.current_step], self.shares]), reward, done, {}
