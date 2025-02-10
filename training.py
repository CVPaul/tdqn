from trading_env import TradingEnv
from single_stock_data_loading import load_data
from tqdn import TDQN
import numpy as np

prices = load_data(stock_index="AAPL")
env = TradingEnv(prices)
tdqn = TDQN(state_dim=env.state_size, action_dim=env.action_space.n)

for episode in range(10):
    state = env.reset()
    total_reward = 0
    for _ in range(len(prices) - 1):
        action = np.argmax(tdqn.predict(state.reshape(1, -1)))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")