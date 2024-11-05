# env/vn30_env.py
import numpy as np
import gym
from gym import spaces
import logging

class TradingEnv(gym.Env):
    """Custom Environment for Trading that follows OpenAI Gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, features_dim, action_dim, initial_balance, transaction_fee=0.001):
        super(TradingEnv, self).__init__()

        # Ensure data has exactly 'features_dim' features plus 'close'
        expected_columns = features_dim + 1  # 'close' is used for profit calculation
        if data.shape[1] != expected_columns:
            raise ValueError(f"Expected data with {expected_columns} columns (features_dim + 'close'), but got {data.shape[1]} columns.")

        self.data = data.reset_index(drop=True)
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(features_dim,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_profit = 0
        self.current_step = 0
        self.positions = 0  # 1 if holding a position, 0 otherwise
        self.buy_price = 0
        self.done = False

        return self._next_observation()

    def _next_observation(self):
        current_step_data = self.data.iloc[self.current_step]
        # Exclude 'close' from observations
        state = current_step_data[:-1].values
        return state.astype(np.float32)

    def step(self, action):
        done = False
        reward = 0
        info = {}

        current_close = self.data['close'].iloc[self.current_step]
        if self.current_step < len(self.data) - 1:
            next_close = self.data['close'].iloc[self.current_step + 1]
        else:
            next_close = current_close

        action = action[0]
        if action > 0.1 and self.positions == 0:
            # Buy signal
            self.positions = 1
            self.buy_price = current_close
            self.balance -= current_close * (1 + self.transaction_fee)
            logging.debug(f"Bought at {current_close}")
        elif action < -0.1 and self.positions == 1:
            # Sell signal
            self.positions = 0
            profit = (current_close - self.buy_price) * (1 - self.transaction_fee)
            self.total_profit += profit
            self.balance += current_close * (1 - self.transaction_fee)
            reward = profit
            logging.debug(f"Sold at {current_close}, Profit: {profit}")

        if self.positions == 1:
            # Holding a position, reward is the change in price
            reward = (next_close - current_close)

        # Limit the reward to avoid extreme values
        if reward < -100 or reward > 100:
            reward = 0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        self.net_worth = self.balance + (self.positions * self.data['close'].iloc[self.current_step])

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        obs = self._next_observation() if not done else np.zeros(self.features_dim, dtype=np.float32)

        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Net Worth: {self.net_worth}')
        print(f'Total Profit: {profit}')

    def close(self):
        pass
