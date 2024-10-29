import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """
    Môi trường giao dịch chứng khoán cho RL.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.total_steps = len(df)
        
        # Actions: 0 = giữ, 1 = mua, 2 = bán
        self.action_space = spaces.Discrete(3)
        
        # Observations: giá hiện tại và các MA
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(df.shape[1],), 
            dtype=np.float32
        )
        
        # Khởi tạo trạng thái
        self.reset()
    
    def reset(self):
        """
        Reset môi trường về trạng thái ban đầu.
        """
        self.current_step = 0
        self.balance = 10000  # Số dư ban đầu
        self.holdings = 0  # Số cổ phiếu đang nắm giữ
        self.total_asset = self.balance
        return self._get_obs()
    
    def _get_obs(self):
        """
        Lấy quan sát tại bước hiện tại.
        """
        return self.df.iloc[self.current_step].values.astype(np.float32)
    
    def step(self, action):
        """
        Thực hiện hành động và cập nhật trạng thái.
        """
        done = False
        info = {}
        reward = 0
        
        current_price = self.df.iloc[self.current_step]['close']
        
        if action == 1:  # Mua
            if self.balance >= current_price:
                self.holdings += 1
                self.balance -= current_price
        elif action == 2:  # Bán
            if self.holdings > 0:
                self.holdings -= 1
                self.balance += current_price
        
        self.current_step += 1
        
        if self.current_step >= self.total_steps - 1:
            done = True
        
        # Tính tổng tài sản
        self.total_asset = self.balance + self.holdings * current_price
        
        # Phần thưởng là sự thay đổi của tổng tài sản
        reward = self.total_asset - 10000  # So với tài sản ban đầu
        
        return self._get_obs(), reward, done, info
    
    def render(self, mode='human', close=False):
        """
        Hiển thị trạng thái hiện tại của môi trường.
        """
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Holdings: {self.holdings}')
        print(f'Total Asset: {self.total_asset}')
