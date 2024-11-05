import numpy as np

class TradingEnv:
    def __init__(self, data, features_dim, action_dim, initial_balance, transaction_fee=0.001):
        self.data = data.drop(columns=['time'], errors='ignore').reset_index(drop=True)
        self.features_dim = features_dim
        self.action_dim = action_dim
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.total_profit = 0
        self.positions = 0
        self.buy_price = 0
        state = self._get_state()
        return state

    def _get_state(self):
        current_step_data = self.data.iloc[self.current_step]
        state = current_step_data.values
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
            self.positions = 1
            self.buy_price = current_close
            self.balance -= current_close * (1 + self.transaction_fee)
        elif action < -0.1 and self.positions == 1:
            self.positions = 0
            profit = (current_close - self.buy_price) * (1 - self.transaction_fee)
            self.total_profit += profit
            self.balance += current_close * (1 - self.transaction_fee)
            reward = profit
        
        if self.positions == 1:
            reward = (next_close - current_close)
        
        if reward < -100 or reward > 100:
            reward = 0
        
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
        
        next_state = self._get_state() if not done else np.zeros(self.features_dim, dtype=np.float32)
        return next_state, reward, done, info