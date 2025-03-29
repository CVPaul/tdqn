import gymnasium as gym
import numpy as np
import pandas as pd
from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.prices = data['close'].values  # Extract Close prices as numpy array
        self.current_step = 0
        self.shares = 0  # Initial shares
        self.state_size = 2  # [price, position]
        self.volume = 10
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(2,))
    
    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        return np.array([self.prices[self.current_step], self.shares])
    
    def step(self, action):
        done = self.current_step >= len(self.prices) - 1
        volume, reward = 0, 0
        price = self.prices[self.current_step]
        if action == 0:  # Buy
            volume = self.volume - self.shares
            self.shares = self.volume
        elif action == 1:  # Sell
            volume = self.volume + self.shares
            self.shares = -self.volume
        # Reward based on portfolio value change
        nxt_price = self.prices[self.current_step + 1]
        reward = self.shares * (nxt_price / price - 1.0) - volume * 5e-4
        self.current_step += 1
        return np.array([self.prices[self.current_step], self.shares]), reward, done, {}
