import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.total_steps = len(df)
        
        self.action_space = spaces.Discrete(3)
        
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(df.shape[1],), 
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.holdings = 0
        self.total_asset = self.balance
        return self._get_obs()
    
    def _get_obs(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)
    
    def step(self, action):
        done = False
        info = {}
        reward = 0
        
        current_price = self.df.iloc[self.current_step]['close']
        
        if action == 1:
            if self.balance >= current_price:
                self.holdings += 1
                self.balance -= current_price
        elif action == 2:
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += current_price
        
        self.current_step += 1
        
        if self.current_step >= self.total_steps - 1:
            done = True
        
        self.total_asset = self.balance + self.holdings * current_price
        
        reward = self.total_asset - 10000
        
        return self._get_obs(), reward, done, info
    
    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Holdings: {self.holdings}')
        print(f'Total Asset: {self.total_asset}')
