import sqlite3
import numpy as np
import pandas as pd

from tqdn import TDQN
from trading_env import TradingEnv


symbol = "DOGEUSD_PERP"
con = sqlite3.connect(f"../bcp/data/{symbol}.db")
dat = pd.read_sql("SELECT start_t, close FROM klines", con)
dat.index = pd.to_datetime(dat.pop('start_t'), unit='ms')
dat = dat.resample('8h').last()
env = TradingEnv(dat)
dqn = TDQN(state_dim=env.state_size, action_dim=env.action_space.n)
dqn.train(env, 10, dat.shape[0] - 1, 8 * 60)