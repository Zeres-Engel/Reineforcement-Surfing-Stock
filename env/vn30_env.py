# env/vn30_env.py
import numpy as np
import gym
from gym import spaces
import logging

class TradingEnv(gym.Env):
    """Custom Environment for Trading that follows OpenAI Gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, features_dim, action_dim=1, initial_balance=10000, transaction_fee=0.001):
        super().__init__()
        self.data = data.astype(np.float32)
        self.features_dim = features_dim - 1
        self.action_dim = action_dim
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee

        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(features_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_profit = 0
        self.current_step = 0
        self.positions = 0
        self.buy_price = 0
        self.done = False

        return self._next_observation()

    def _next_observation(self):
        current_step_data = self.data.iloc[self.current_step]
        # Lấy tất cả features trừ 'close'
        features = [col for col in self.data.columns if col != 'close']
        state = current_step_data[features].values
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
