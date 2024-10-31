from utils.render import StockMarketGraph

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class VN30TrendEnv(gym.Env):

    
    metadata = {	
        "render_modes": ["human", "rgb_array"],	
        "render_fps": 10,	
    }

    def __init__(	
        self,	
        data_loader,	
        features_dim,	
        action_dim,	
        env_cfg,	
        seed,	
        ticket="ACB",	
        mode="train",	
        render_range=30,	
        render_mode="rgb_array"	
    ):	
        #Thiết lập biến đặc trưng cho môi trường	
        self.vn30_raw_dict = data_loader.get_vn30_raw_df_dict()

        self.vn30_dict = data_loader.get_vn30_df_dict()
        self.vn30_tickets = data_loader.get_tickets()
        self.trading_days = data_loader.get_trading_days()
        self.n_predictions = data_loader.get_n_predictions()
        self.seed = seed
        self.ticket = ticket
        self.mode = mode

        #Xử lý render	
        self.render_range = render_range	
        self.render_mode = render_mode	
        self.grapth = StockMarketGraph(ticket=self.ticket, render_range=self.render_range, n_predictions=self.n_predictions)	

        
        if self.mode == "train":
            self.start_day_ind, self.end_day_ind = data_loader.get_train_date_inds()
        elif self.mode == "validation":
            (
                self.start_day_ind,
                self.end_day_ind,
            ) = data_loader.get_validation_date_inds()
        else:
            self.start_day_ind, self.end_day_ind = data_loader.get_test_date_inds()

        self.floor_price_change, self.ceil_price_change = -0.07, 0.07
        self.shift_reward = env_cfg.shift_reward
        self.scale_reward = env_cfg.scale_reward
        self.info_keys = ['true', 'prediction']

        self.action_dim = action_dim
        self.action_space = spaces.Box(	
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype="float32"	
        )	
        self.features_dim = features_dim	
        self.observation_space = spaces.Box(	
            low=np.NINF, high=np.PINF, shape=(self.features_dim,), dtype="float32"	
        )

    def get_start_end_day_ind(self):	
        return (self.start_day_ind, self.end_day_ind)
    
    def get_info_keys(self):
        return self.info_keys

    def _get_state(self):
        state_days = self.trading_days[self.curr_day_ind - self.n_predictions]
        state_df = self.vn30_dict[self.ticket].loc[state_days, :]
        state_np = np.reshape(	
            state_df.to_numpy(dtype="float32"), self.observation_space.shape	
        )
        return state_np
    
    def _get_reward_and_info(self, action):
        if self.curr_day_ind == (self.end_day_ind + 1):	
            reward = 0	
            info = {}	
        else:
            action = action.squeeze()
            curr_day_price_change = ((self.vn30_dict[self.ticket].loc[self.trading_days[self.curr_day_ind], 'close']) - 1).squeeze() #scalar
            distance = (curr_day_price_change * 100 - action * 7)/ 14
            #distance = 0 if np.abs(distance) < 0.03 else np.abs(distance) 
            distance = np.abs(distance)
            reward = (self.shift_reward - distance) * self.scale_reward
            info = {}
            
            info['true'] = np.where(curr_day_price_change > 0, 1, 0) 
            info['prediction'] = np.where(action > 0, 1, 0) #scalar  

            self.score += (1 if info['true'] == info['prediction'] else -1)	
            self.score_history.append(self.score)
        return reward, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.curr_day_ind = (	
            self.n_predictions	
            if self.start_day_ind < self.n_predictions	
            else self.start_day_ind	
        )	
        #Xử lý render	
        self.pivot_day_ind = self.curr_day_ind - self.n_predictions	
        self.grapth.reset(self.pivot_day_ind)	
        self.score = 0	
        self.score_history = [self.score] * self.n_predictions

        state = self._get_state()
        info = {}
        return state, info
    
    def step(self, action):
        assert action.shape[0] == self.action_dim, f'action_shape {action.shape}, action_dim {self.action_dim}'
        assert type(action).__module__ == np.__name__

        action = np.clip(action, -1, 1)

        reward, info = self._get_reward_and_info(action)

        if self.curr_day_ind == (self.end_day_ind + 1):	
            terminated = False
            next_state, _ = self.reset()	
        else:
            terminated = False
            if self.curr_day_ind == self.end_day_ind:
                terminated = True
                
            self.curr_day_ind += 1
            next_state = self._get_state()

        return next_state, reward, terminated, False, info
    
    def length(self):	
        return (	
            self.end_day_ind	
            - self.start_day_ind	
            + 1	
            - (self.n_predictions if self.start_day_ind < self.n_predictions else 0)	
        )	
    def render(self):	
        return self.grapth.render(self.vn30_raw_dict, self.score_history, self.trading_days, self.curr_day_ind, self.render_mode)